# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import sys
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import BertTokenizer
from copy import deepcopy
sys.path.append("/itet-stor/sfurrer/net_scratch/UNITER/Robust_Contrastive_UNITER/Geometric_attack")
from greedy_attack_vilt import GreedyAttack
from torch.nn.utils.rnn import pad_sequence
from vilt_module   import vilt_module.py

"""
To improve the code : 
step1. add the function attack to the class MoCo
"""

class MoCo(pl.LightningModule):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,config ,dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys   (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.config    = config 
        self.K         = K
        self.m         = m
        self.T         = T
        self.criterion = nn.CrossEntropyLoss() 
        
        # Prepare the ViLT Models
        self.ViLTransformer_q = ViLTransformerSS.(_config)
        self.ViLTransformer_k = ViLTransformerSS.(_config)
        print("\n\nMoCo Part1: Correctly Creat Querry and Key Networks --SUCESS--.")

        for param_q, param_k in zip(self.ViLTransformer_q.parameters(), 
                                    self.ViLTransformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # Dynamic queue text-Img
        self.register_buffer("txt_img_queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.txt_img_queue, dim=0)
        self.register_buffer("txt_img_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        print("MoCo Part2: Correctly Creat the Both queues --SUCESS--.")
        
        # tokenizer and attacker
        self.tokenizer       = BertTokenizer.from_pretrained('bert-base-uncased')   
        self.greedy_attacker = GreedyAttack(args         = self.config,
                                            n_candidates = config["n_candidates"],
                                            max_loops    = config["n_candidates"],    
                                            tokenizer    = self.tokenizer)
             
        print("MoCo Part3: MoCo Initialisation --SUCESS--.")
        print("-------------------------------------------")
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.ViLTransformer_q.parameters(), 
                                    self.ViLTransformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_txt,keys_img,dist):
        """ dequeue and euqueue the new batch of text AND image representation"""
        # gather keys before updating queue
        
        keys_txt = concat_all_gather(keys_txt)
        keys_img = concat_all_gather(keys_img)
        batch_size = keys_txt.size(0)

        ptr = int(self.txt_img_queue_ptr)

        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.txt_img_queue[:, ptr:ptr + batch_size] = keys_txt.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        
        self.txt_img_queue[:, ptr:ptr + batch_size] = keys_img.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        
        self.txt_img_queue_ptr[0] = ptr

    def forward(self, batch,pgd_parameters,opts,device):
        """
        Input:
            batch_q: a batch of query text-Image pair
            batch_k: a batch of key text-Image pair
        Do   :
        Step1. Get the key and querry of the batch || ok
        Step2. Compute the loss function           || ok
        Step3. Enqueue - Dequeue                   || ok
        ------------------------------
        Step2*. Implement n steps of attack with 
                1. PGD on Images                   || ok
                2. Geometric on Text               || ok
                3. Geom + Pgd                      || ok
        -------------------------------
        Step4. Get the loss and the "targets"      || ok
                
        Output:
            logits, targets
        """

        print("\nPASS FLAG 5. : Entering inside forward of Moco")
        # compute key features
        with torch.no_grad():                    # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            infer             = self.ViLTransformer_k.infer(batch)
            cls_text          = infer1["text_feats"][:,0]
            cls_img           = infer1["image_feats"][:,0] 
            img_rep, text_rep = self.ViLTransformer_k.projector(cls_img,cls_text)
            txt_k             = nn.functional.normalize(text_rep, dim=1)# keys: NxC 
            img_k             = nn.functional.normalize(img_rep, dim=1)# keys: NxC 
        
        ############ Attacks
        # PGD Attack
        print("Entering pgd attacks")
        img_delta,logs = pgd_attack(txt_k,
                                    self.txt_img_queue.clone().detach(),
                                    batch)            
        print("\nPASS FLAG 6. : PGD on Info Loss")
        
        # Geometric attack
        batch,txt_original_attacked = geometric(batch,img_k,self.txt_img_queue.clone().detach())
        
        infer             = self.ViLTransformer_q.infer(batch)
        cls_text          = infer1["text_feats"][:,0]
        cls_img           = infer1["image_feats"][:,0] 
        img_rep, text_rep = self.ViLTransformer_q.projector(cls_img,cls_text)
        txt_q_attacked    = nn.functional.normalize(text_rep, dim=1)# keys: NxC 
        img_q_attacked    = nn.functional.normalize(img_rep, dim=1) # keys: NxC         
        
        print("\nPASS FLAG 7. : Geometric Inspired attack on Info Loss")
                
        # Build the InfoNCE component
        l_pos_txt  = torch.einsum('nc,nc->n', [txt_q_attacked, img_k]).unsqueeze(-1)
        l_pos_img  = torch.einsum('nc,nc->n', [img_q_attacked, txt_k]).unsqueeze(-1)
        # negative logits: NxK        
        l_neg_txt  = torch.einsum('nc,ck->nk', [txt_q_attacked, self.txt_img_queue.clone().detach()])
        l_neg_img  = torch.einsum('nc,ck->nk', [img_q_attacked, self.txt_img_queue.clone().detach()])
        # logits: Nx(1+K)
        logits_txt = torch.cat([l_pos_txt, l_neg_txt], dim=1)
        logits_img = torch.cat([l_pos_img, l_neg_img], dim=1)
        # apply temperature
        logits_txt /= self.T
        logits_img /= self.T

        # labels: positive key indicators
        labels_txt = torch.zeros(logits_txt.shape[0], dtype=torch.long).cuda()
        labels_img = torch.zeros(logits_img.shape[0], dtype=torch.long).cuda()        
        # dequeue and enqueue
        # Tricks to make it work with non distributed : 
        logits = {"txt" : logits_txt,
                  "img" : logits_img}
        labels = {"txt" : labels_txt,
                  "img" : labels_img}
        
        dist = False
        self._dequeue_and_enqueue(txt_k,img_k,dist)
        print("\nPASS FLAG 8. : Correctly compute param of Loss and Dequeue-enqueue")
        return logits, labels ,logs , txt_original_attacked
        
    def pgd(self,txt_k,txt_img_queue,batch,) : 
        
        # Need device
        
        ## hyper parameter PGD
        adv_steps_img    = self.config["adv_steps_img"]
        adv_lr_img       = self.config["adv_lr_img"]
        adv_max_norm_img = self.config["adv_max_norm_img"]    
        # Get the original img
        img_init = batch['image'][0]
        # Initialize the delta as zero vectors
        img_delta = torch.zeros_like(img_init)
        for astep in range(adv_steps_img):                
            #print("This is the step : ", astep)
            # Need to get the gradient for each batch of image features 
            img_delta.requires_grad_(True)                
            # Get all answer of model with adv_delta added to img_feat
            try :
                batch['image'][0] = (img_init + img_delta)#.to(pl_module.device)
                infer             = self.ViLTransformer_q.infer(batch)
                cls_text          = infer1["text_feats"][:,0]
                cls_img           = infer1["image_feats"][:,0] 
                img_rep, text_rep = self.ViLTransformer_q.projector(cls_img,cls_text)
                img_q_attacke     = nn.functional.normalize(img_rep, dim=1)# keys: NxC 

            except:
                print("problem in step ",astep)
                sys.exit("STOPP")
            # Creat InfoNCE loss
            l_pos = torch.einsum('nc,nc->n', [img_q_attacke, txt_k]).unsqueeze(-1)
            # negative logits: NxK        
            l_neg = torch.einsum('nc,ck->nk', [img_q_attacke, txt_img_queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.T
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss   = self.criterion(logits, labels)
            # calculate x.grad
            loss.backward(retain_graph=True)
            # Get gradient
            img_delta_grad = img_delta.grad.clone().detach().float()
            # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
            denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1,1)

            # Clip gradient to Lower Bound
            denorm = torch.clamp(denorm, min=1e-8)
            # calculate delta_step  with format img_delta
            img_delta_step = (adv_lr_img * img_delta_grad / denorm).to(img_delta)
            # Add the calculated step to img_delta (The perturbation)
            img_delta = (img_delta + img_delta_step).detach()
            # clip the delta if needed
            if adv_max_norm_img > 0:
                img_delta = torch.clamp(img_delta, -adv_max_norm_img, adv_max_norm_img).detach()                      
        return batch

    def geometric(self, batch,img_k,txt_img_queue) : 
    
        attack_words = \
        self.greedy_attacker.adv_attack_samples(self.ViLTransformer_q, 
                                               batch
                                               img_k
                                               txt_img_queue       
                                              ) 
        print("This is the Real versus attacked sentences : ")

        for i in range(len(batch["text"])):
            print("Real sentence----: ",batch["text"][i])
            print("Attacked sentence: ",attack_words["attacked_words"][i])

        txt_original_attacked   = {"original": batch["text"],
                                   "attacked": attack_words["text"]
                                  }
        batch["text"]         = attack_words["text"]
        batch["text_ids"]     = attack_words["text_ids"]
        batch["text_masks"]   = attack_words["text_masks"]

        return batch, txt_original_attacked

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss
    
    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)    
    
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

    def add_element(modality,dict, key1,key2, value):
        if key1 not in dict:
            if modality == "image" : 
                dict[key1] = {"loss_img"  : []}
            if modality == "text" : 
                dict[key1] = {"loss_txt"  : []}            
        dict[key1][key2].append(value)
        
