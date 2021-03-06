import pickle 
import sys
import time
from copy import deepcopy, copy#

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from vilt.modules.dist_utils import all_gather
from TSNE_vizualisation import TSNE_projection
import torchvision.transforms as T

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance

def text_augmentation(pl_module, batch):
    batch = pl_module.text_augmentation.augmentation(pl_module, batch)
    return batch
   
def image_augmentation(pl_module, batch):    
    new_images = pl_module.image_augmentation.augmentation(batch)
    new_images[0] = new_images[0].to(pl_module.device)
    batch["image"] = new_images
    return batch

# To save the image 
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# To save the image 
def show(imgs_clean,imgs_attack1,imgs_attack2=None,imgs_delta=None):
    """Open an tensor and save it in PNG"""
    save_exemple = "./attacks_analysis_vilt/PGD/exemple"
    save_exemple_augm = "./ViLT/attacks_analysis_vilt/AUGM/exemple"
    unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    imgs_clean = imgs_clean.to('cpu')
    imgs_attack1= imgs_attack1.to('cpu') 
    if imgs_attack2 is not None : 
        imgs_attack2= imgs_attack2.to('cpu') 
        imgs_delta = imgs_delta.to('cpu')
    if imgs_attack2 is not None :
        for i, (img_clean, img_attack1,img_attack2,img_delta) in enumerate(zip(imgs_clean,imgs_attack1,imgs_attack2, imgs_delta)):
            img_clean  = unorm(img_clean)    
            img_attack1 = unorm(img_attack1)
            img_attack2 = unorm(img_attack2)
            img_delta = unorm(img_delta)
            img_clean = T.ToPILImage()(img_clean)
            img_attack1 = T.ToPILImage()(img_attack1)
            img_attack2 = T.ToPILImage()(img_attack2)
            img_delta = T.ToPILImage()(img_delta)
            img_clean.save(os.path.join(save_exemple,"img_clean{}.png".format(i)),"PNG")
            img_attack1.save(os.path.join(save_exemple,"img_attack1_{}.png".format(i)),"PNG")
            img_attack2.save(os.path.join(save_exemple,"img_attack2_{}.png".format(i)),"PNG")
            img_delta.save(os.path.join(save_exemple,"img_delta{}.png".format(i)),"PNG")
    else : 
        for i, (img_clean, img_augm) in enumerate(zip(imgs_clean,imgs_attack1)):
            img_clean  = unorm(img_clean)    
            img_augm = unorm(img_augm)     
            img_clean = T.ToPILImage()(img_clean)
            img_augm = T.ToPILImage()(img_augm)   
            img_clean.save(os.path.join(save_exemple_augm,"img_clean{}.png".format(i)),"PNG")
            img_augm.save(os.path.join(save_exemple_augm,"img_augm{}.png".format(i)),"PNG")            
    sys.exit("Stop")
    
def compute_pgd(pl_module, batch, loss_name, k_modality=None):
    verbose = False
    if verbose == True :      
        if loss_name == "nlvr2_attacked":
            image_clean0 = batch["image_0"][0]   
            image_clean1 = batch["image_1"][0]      
    img_delta = pl_module.pgd_attacker.pgd_attack(pl_module, batch, k_modality = k_modality)
    if loss_name == "nlvr2_attacked":  
        image0 = batch["image_0"][0] + img_delta[0]    
        image1 = batch["image_1"][0] + img_delta[1]  
        if verbose == True :
            show(image_clean0, image0,image1,img_delta[0])
        
        batch["image_0"][0] = batch["image_0"][0] + img_delta[0]
        batch["image_1"][0] = batch["image_1"][0] + img_delta[1]
    else:
        batch["image"][0] = batch["image"][0] + img_delta
    
    phase = "train" if pl_module.training else "val"
    if loss_name == "nlvr2_attacked":
        delta_range_0 = torch.linalg.norm(img_delta[0], dim=1).mean()
        delta_range_1 = torch.linalg.norm(img_delta[1], dim=1).mean()
        delta_range = (delta_range_0 + delta_range_1) / sum(pl_module.attack_idx)
    else:
        delta_range = torch.linalg.norm(img_delta, dim=1).mean()
    # print("delta:", delta_range)
    pl_module.log(f"{loss_name}_attack/{phase}/delta", delta_range)
    
    return batch
    
def compute_geometric(pl_module, batch, loss_name, k_modality=None):
    
    real_sentence = batch["text"]
    attack_words = pl_module.greedy_attacker.adv_attack_samples(pl_module,batch,k_modality) 
    
    #if attack_words["Problem"]:
    #    print("This is changes",attack_words['changes_verification'])
    #    print("This is the Real versus attacked sentences : ")
        
    #for i in range(len(batch["text"])):
    #    print("Real sentence----: ",real_sentence[i])
    #    print("Attacked sentence: ",attack_words["text"][i])

     #   txt_original_attacked   = {"original": real_sentence,
     #                              "attacked": attack_words["text"]
     #                             }
        
    batch["text"] = attack_words["text"]
    batch["text_ids"] = attack_words["txt_input_ids"]
    batch["text_masks"] = attack_words["text_masks"]

    phase = "train" if pl_module.training else "val"
    pl_module.log(f"{loss_name}_attack/{phase}/num_changes", attack_words["num_changes"])
    pl_module.log(f"{loss_name}_attack/{phase}/change_rate", attack_words["change_rate"])
    
    return batch 

def compute_moco_contrastive(pl_module, batch):
    
    def _momentum_update_key_layer(em, q_layer, k_layer):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(q_layer.parameters(), k_layer.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
    
    def _concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
                
        output = torch.cat(tensors_gather, dim=0)
        return output

    #def _dequeue_and_enqueue(keys, queue_type):
    def _dequeue_and_enqueue(keys):
        """ dequeue and euqueue the new batch of joint representation"""
        keys = _concat_all_gather(keys)
        batch_size = keys.shape[0]
        if not (pl_module.per_step_bs == batch_size):
            return
        ptr = int(pl_module.proj_queue_ptr)
        # assert pl_module.num_negative % batch_size == 0
        pl_module.proj_queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % pl_module.num_negative
        pl_module.proj_queue_ptr[0] = ptr        

    loss = 0
    loss_num = 0
    loss_fct = nn.CrossEntropyLoss()
    ret = {}
    phase = "train" if pl_module.training else "val"
    
    # momentum update key encoder
    _momentum_update_key_layer(pl_module.momentum, pl_module.text_embeddings, pl_module.k_text_embeddings)
    _momentum_update_key_layer(pl_module.momentum, pl_module.token_type_embeddings, pl_module.k_token_type_embeddings)
    _momentum_update_key_layer(pl_module.momentum, pl_module.transformer, pl_module.k_transformer)
    _momentum_update_key_layer(pl_module.momentum, pl_module.moco_head, pl_module.k_moco_head)

    with torch.no_grad():
        infer_k = pl_module.infer_k(batch, mask_text=False, mask_image=False)
        projection_cls_feats_k = pl_module.k_moco_head(infer_k["cls_feats"])
        k = nn.functional.normalize(projection_cls_feats_k, dim=1)

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    projection_cls_feats = pl_module.moco_head(infer["cls_feats"])
    q_original = nn.functional.normalize(projection_cls_feats, dim=1)
    neg_k = pl_module.proj_queue.clone().detach()
    l_pos = torch.einsum('nc,nc->n', [q_original, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q_original, neg_k])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= pl_module.temperature
    prediction_original = logits.argmax(-1)
   
    if pl_module.text_view:
        if pl_module.augmentation : 
            augmented_batch = text_augmentation(pl_module, deepcopy(batch))
        else :   
            attacked_words = {}
            augmented_batch = compute_geometric(pl_module, deepcopy(batch), "moco", k_modality = k)
            attacked_words["text"] = deepcopy(augmented_batch["text"]) 
            attacked_words["txt_input_ids"] = deepcopy(augmented_batch["text_ids"])
            attacked_words["text_masks"] = deepcopy(augmented_batch["text_masks"])        
        
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)       
        projection_cls_feats = pl_module.moco_head(infer["cls_feats"])
        q_txt_attack = nn.functional.normalize(projection_cls_feats, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q_txt_attack, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_txt_attack, neg_k])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature
    
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
        if phase == "train":
             pl_module.log(f"moco_attack/Geom_success_rate", (~(logits.argmax(-1) == prediction_original)).sum() / logits.shape[0])        
        ret["pos_dist_attacked_txt"] = torch.linalg.norm(q_txt_attack - k, dim=1).mean() 
        ret["pos_cosine_attacked_txt"] = pl_module.cosine(q_txt_attack,k).mean()
        ret["pos_dot_attacked_txt"] = torch.sum(q_txt_attack * k, dim=1).mean() 
        dist = 0
        cosine = 0
        dot = 0
        for sub_q in q_txt_attack:
            dist += torch.linalg.norm(sub_q - neg_k.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_k.T).mean()
            dot += torch.sum(torch.unsqueeze(sub_q,0) * neg_k.T, dim=1).mean()
        ret["neg_dist_attacked_txt"] = dist / q_txt_attack.shape[0] 
        ret["neg_cosine_attacked_txt"] = cosine / q_txt_attack.shape[0] 
        ret["neg_dot_attacked_txt"] = dot / q_txt_attack.shape[0] 

        loss_attacked_text = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/attacked_txt_loss", loss_attacked_text)
        loss = loss + loss_attacked_text
        loss_num += 1        
   
    if pl_module.image_view:
        if pl_module.augmentation :            
            augmented_batch = image_augmentation(pl_module, deepcopy(batch))
        else : 
            augmented_batch = compute_pgd(pl_module, deepcopy(batch), "moco", k_modality=k)
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        projection_cls_feats = pl_module.moco_head(infer["cls_feats"])
        q_img_attack = nn.functional.normalize(projection_cls_feats, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q_img_attack, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_img_attack, neg_k])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature
    
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)        
        if phase == "train":
            pl_module.log(f"moco_attack/PGD_success_rate", (~(logits.argmax(-1) == prediction_original)).sum()/logits.shape[0])
        ret["pos_dist_attacked_img"] = torch.linalg.norm(q_img_attack-k, dim=1).mean() 
        ret["pos_cosine_attacked_img"] = pl_module.cosine(q_img_attack,k).mean()       
        ret["pos_dot_attacked_img"] = torch.sum(q_img_attack * k, dim=1).mean()        
        dist = 0
        cosine = 0
        dot = 0
        for sub_q in q_img_attack:
            dist += torch.linalg.norm(sub_q-neg_k.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_k.T).mean()
            dot +=  torch.sum(torch.unsqueeze(sub_q,0) * neg_k.T, dim=1).mean()
        ret["neg_dist_attacked_img"] = dist / q_img_attack.shape[0] 
        ret["neg_cosine_attacked_img"] = cosine / q_img_attack.shape[0]  
        ret["neg_dot_attacked_img"] = dot / q_img_attack.shape[0]   
        
        loss_attacked_img = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/attacked_img_loss", loss_attacked_img)
        loss = loss + loss_attacked_img
        loss_num += 1     
        
    if pl_module.image_view and pl_module.text_view and not pl_module.augmentation: 
        
        augmented_batch["text"] = attacked_words["text"]
        augmented_batch["text_ids"] = attacked_words["txt_input_ids"]
        augmented_batch["text_masks"] = attacked_words["text_masks"]
        
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)       
        projection_cls_feats = pl_module.moco_head(infer["cls_feats"])
        q_both_attack = nn.functional.normalize(projection_cls_feats, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q_both_attack, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q_both_attack, neg_k])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature
    
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
        if phase == "train":
             pl_module.log(f"moco_attack/Both_success_rate", (~(logits.argmax(-1) == prediction_original)).sum() / logits.shape[0])        
        ret["pos_dist_attacked_both"] = torch.linalg.norm(q_both_attack - k, dim=1).mean() 
        ret["pos_cosine_attacked_both"] = pl_module.cosine(q_both_attack,k).mean()
        ret["pos_dot_attacked_both"] = torch.sum(q_both_attack * k, dim=1).mean() 
        dist = 0
        cosine = 0
        dot = 0
        for sub_q in q_both_attack:
            dist += torch.linalg.norm(sub_q - neg_k.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_k.T).mean()
            dot += torch.sum(torch.unsqueeze(sub_q,0) * neg_k.T, dim=1).mean()
        ret["neg_dist_attacked_both"] = dist / q_both_attack.shape[0] 
        ret["neg_cosine_attacked_both"] = cosine / q_both_attack.shape[0] 
        ret["neg_dot_attacked_both"] = dot / q_both_attack.shape[0] 

        loss_attacked_both = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/attacked_both_loss", loss_attacked_both)
        loss = loss + loss_attacked_both
        loss_num += 1         
     
    if pl_module.training:
        _dequeue_and_enqueue(k)
    
    ret["moco_loss"] = loss / loss_num
    
    loss = getattr(pl_module, f"{phase}_moco_loss")(ret["moco_loss"])
    pl_module.log(f"moco_loss/step/{phase}", loss)
    
    if pl_module.image_view:  
        # L2 distance
        pl_module.log(f"moco_dist_{phase}_L2/Pos_attacked_img",ret["pos_dist_attacked_img"]) 
        pl_module.log(f"moco_dist_{phase}_L2/Neg_attacked_img",ret["neg_dist_attacked_img"]) 
        pl_module.log(f"moco_dist_{phase}_L2/Neg-Pos_attacked_img",ret["neg_dist_attacked_img"]-ret["pos_dist_attacked_img"])         
        # Cosine distance        
        pl_module.log(f"moco_dist_{phase}_Cosine/Pos_attacked_img",ret["pos_cosine_attacked_img"]) 
        pl_module.log(f"moco_dist_{phase}_Cosine/Neg_attacked_img",ret["neg_cosine_attacked_img"])         
        pl_module.log(f"moco_dist_{phase}_Cosine/Neg-Pos_attacked_img",ret["neg_cosine_attacked_img"]-ret["pos_cosine_attacked_img"])   
        
        # dot distance
        pl_module.log(f"moco_dist_{phase}_Dot/Pos_attacked_img",ret["pos_dot_attacked_img"]) 
        pl_module.log(f"moco_dist_{phase}_Dot/Neg_attacked_img",ret["neg_dot_attacked_img"])         
        pl_module.log(f"moco_dist_{phase}_Dot/Neg-Pos_attacked_img",ret["neg_dot_attacked_img"]-ret["pos_dot_attacked_img"]) 
        
    if pl_module.text_view:
        # L2 distance
        pl_module.log(f"moco_dist_{phase}_L2/Pos_attacked_txt",ret["pos_dist_attacked_txt"]) 
        pl_module.log(f"moco_dist_{phase}_L2/Neg_attacked_txt",ret["neg_dist_attacked_txt"]) 
        pl_module.log(f"moco_dist_{phase}_L2/Neg-Pos_attacked_txt",ret["neg_dist_attacked_txt"]-ret["pos_dist_attacked_txt"])         
        # Cosine distance        
        pl_module.log(f"moco_dist_{phase}_Cosine/Pos_attacked_txt",ret["pos_cosine_attacked_txt"]) 
        pl_module.log(f"moco_dist_{phase}_Cosine/Neg_attacked_txt",ret["neg_cosine_attacked_txt"])         
        pl_module.log(f"moco_dist_{phase}_Cosine/Neg-Pos_attacked_txt",ret["neg_cosine_attacked_txt"]-ret["pos_cosine_attacked_txt"]) 
        
        # dot distance        
        pl_module.log(f"moco_dist_{phase}_Dot/Pos_attacked_txt",ret["pos_dot_attacked_txt"]) 
        pl_module.log(f"moco_dist_{phase}_Dot/Neg_attacked_txt",ret["neg_dot_attacked_txt"])         
        pl_module.log(f"moco_dist_{phase}_Dot/Neg-Pos_attacked_txt",ret["neg_dot_attacked_txt"]-ret["pos_dot_attacked_txt"])   
        
    if pl_module.image_view and pl_module.text_view and not pl_module.augmentation: 
        # L2 distance
        pl_module.log(f"moco_dist_{phase}_L2/Pos_attacked_both",ret["pos_dist_attacked_both"]) 
        pl_module.log(f"moco_dist_{phase}_L2/Neg_attacked_both",ret["neg_dist_attacked_both"]) 
        pl_module.log(f"moco_dist_{phase}_L2/Neg-Pos_attacked_both",ret["neg_dist_attacked_both"]-ret["pos_dist_attacked_both"])         
        # Cosine distance        
        pl_module.log(f"moco_dist_{phase}_Cosine/Pos_attacked_both",ret["pos_cosine_attacked_both"]) 
        pl_module.log(f"moco_dist_{phase}_Cosine/Neg_attacked_both",ret["neg_cosine_attacked_both"])         
        pl_module.log(f"moco_dist_{phase}_Cosine/Neg-Pos_attacked_both",ret["neg_cosine_attacked_both"]-ret["pos_cosine_attacked_both"]) 
        
        # dot distance        
        pl_module.log(f"moco_dist_{phase}_Dot/Pos_attacked_both",ret["pos_dot_attacked_both"]) 
        pl_module.log(f"moco_dist_{phase}_Dot/Neg_attacked_both",ret["neg_dot_attacked_both"])         
        pl_module.log(f"moco_dist_{phase}_Dot/Neg-Pos_attacked_both",ret["neg_dot_attacked_both"]-ret["pos_dot_attacked_both"])            
                     
    return ret

def compute_barlowtwins_contrastive(pl_module, batch):
    loss = 0
    loss_num = 0
    ret = {}
    
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    with torch.no_grad():
        infer = pl_module.infer(batch, mask_text=False, mask_image=False)
        k = pl_module.barlowtwins_head(infer['cls_feats'])   
        
    if pl_module.text_view:
        if pl_module.augmentation : 
            augmented_batch = text_augmentation(pl_module, deepcopy(batch))
        else : 
            attacked_words = {}
            augmented_batch = compute_geometric(pl_module, deepcopy(batch), "barlowtwins",k_modality=k)
            attacked_words["text"] = deepcopy(augmented_batch["text"]) 
            attacked_words["txt_input_ids"] = deepcopy(augmented_batch["text_ids"])
            attacked_words["text_masks"] = deepcopy(augmented_batch["text_masks"])   
            
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        q_text = pl_module.barlowtwins_head(infer['cls_feats'])
        c_2 = q_text.T @ k
        c_2.div_(pl_module.per_step_bs)
        torch.distributed.all_reduce(c_2)
        on_diag = torch.diagonal(c_2).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_2).pow_(2).sum()         
        
        loss = loss + on_diag + pl_module.adv_lr * off_diag
        loss_num += 1   
        
        ret["barlowtwins_loss_invariance_text"] = on_diag 
        ret["barlowtwins_loss_redundancy_text"] = pl_module.adv_lr * off_diag          
        
        dist_image_neg_dist = 0
        dist_text_neg_dist  = 0
        dist_image_neg_cosine = 0
        dist_text_neg_cosine  = 0
        
        ret["pos_dist_attacked_txt"] = torch.linalg.norm(q_text - k, dim=1).mean()
        ret["pos_cosine_attacked_txt"] = pl_module.cosine(q_text,k).mean() 
        ret["pos_dot_attacked_txt"] = torch.sum(q_text * k, dim=1).mean()          
   
    if pl_module.image_view:
        if pl_module.augmentation : 
            augmented_batch = image_augmentation(pl_module, deepcopy(batch))
        else : 
            augmented_batch = compute_pgd(pl_module, deepcopy(batch), "barlowtwins",k_modality=k)
            
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        q_image = pl_module.barlowtwins_head(infer['cls_feats'])
        c_1 = q_image.T @ k
        c_1.div_(pl_module.per_step_bs)
        torch.distributed.all_reduce(c_1)
        on_diag = torch.diagonal(c_1).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_1).pow_(2).sum()
         
        loss = loss + on_diag + pl_module.adv_lr * off_diag
        loss_num += 1
        ret["barlowtwins_loss_invariance_img"] = on_diag 
        ret["barlowtwins_loss_redundancy_img"] = pl_module.adv_lr * off_diag  
        
        dist_image_neg_dist = 0
        dist_text_neg_dist  = 0
        dist_image_neg_cosine = 0
        dist_text_neg_cosine  = 0 
        
        ret["pos_dist_attacked_img"] = torch.linalg.norm(q_image - k, dim=1).mean() 
        ret["pos_cosine_attacked_img"] = pl_module.cosine(q_image,k).mean() 
        ret["pos_dot_attacked_img"] = torch.sum(q_image * k, dim=1).mean()   
          
    if pl_module.image_view and pl_module.text_view and not pl_module.augmentation:    
        augmented_batch["text"] = attacked_words["text"]
        augmented_batch["text_ids"] = attacked_words["txt_input_ids"]
        augmented_batch["text_masks"] = attacked_words["text_masks"]
        
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        q_both = pl_module.barlowtwins_head(infer['cls_feats'])
        c_3 = q_both.T @ k
        c_3.div_(pl_module.per_step_bs)
        torch.distributed.all_reduce(c_3)
        on_diag = torch.diagonal(c_3).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c_3).pow_(2).sum()         
        
        loss = loss + on_diag + pl_module.adv_lr * off_diag
        loss_num += 1   
        
        ret["barlowtwins_loss_invariance_both"] = on_diag 
        ret["barlowtwins_loss_redundancy_both"] = pl_module.adv_lr * off_diag          
        
        dist_image_neg_dist = 0
        dist_text_neg_dist  = 0
        dist_image_neg_cosine = 0
        dist_text_neg_cosine  = 0
        
        ret["pos_dist_attacked_both"] = torch.linalg.norm(q_both - k, dim=1).mean()
        ret["pos_cosine_attacked_both"] = pl_module.cosine(q_both,k).mean() 
        ret["pos_dot_attacked_both"] = torch.sum(q_both * k, dim=1).mean()          
    
    ret["barlowtwins_loss"] = loss / loss_num

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_barlowtwins_loss")(ret["barlowtwins_loss"])
    pl_module.log(f"barlowtwins/{phase}/loss", loss)

  
    if pl_module.image_view:
        # L2 distance
        pl_module.log(f"barlowtwins_dist_{phase}_L2/Pos_attacked_img",ret["pos_dist_attacked_img"])
        # Cosine Distance
        pl_module.log(f"barlowtwins_dist_{phase}_Cosine/Pos_attacked_img",ret["pos_cosine_attacked_img"])
        # Dot Distance
        pl_module.log(f"barlowtwins_dist_{phase}_Dot/Pos_attacked_img",ret["pos_dot_attacked_img"])        
        
        # Loss_invariance and Loss redundancy
        loss_invariance_img = getattr(pl_module, f"{phase}_barlowtwins_loss_invariance_img")(ret["barlowtwins_loss_invariance_img"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_invariance_img", loss_invariance_img)  
        loss_redundancy_img = getattr(pl_module, f"{phase}_barlowtwins_loss_redundancy_img")(ret["barlowtwins_loss_redundancy_img"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_redundancy_img", loss_redundancy_img)         
        
    if pl_module.text_view:
        # L2 distance
        pl_module.log(f"barlowtwins_dist_{phase}_L2/Pos_attacked_txt",ret["pos_dist_attacked_txt"]) 
        # Cosine Distance   
        pl_module.log(f"barlowtwins_dist_{phase}_Cosine/Pos_attacked_txt",ret["pos_cosine_attacked_txt"])
        # Dot Distance   
        pl_module.log(f"barlowtwins_dist_{phase}_Dot/Pos_attacked_txt",ret["pos_dot_attacked_txt"])        
        
        # Loss_invariance and Loss redundancy
        loss_invariance_text = getattr(pl_module, f"{phase}_barlowtwins_loss_invariance_text")(ret["barlowtwins_loss_invariance_text"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_invariance_text", loss_invariance_text)  
        loss_redundancy_text = getattr(pl_module, f"{phase}_barlowtwins_loss_redundancy_text")(ret["barlowtwins_loss_redundancy_text"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_redundancy_text", loss_redundancy_text)      
    
    if pl_module.image_view and pl_module.text_view and not pl_module.augmentation:
        # L2 distance
        pl_module.log(f"barlowtwins_dist_{phase}_L2/Pos_attacked_both",ret["pos_dist_attacked_both"]) 
        # Cosine Distance   
        pl_module.log(f"barlowtwins_dist_{phase}_Cosine/Pos_attacked_both",ret["pos_cosine_attacked_both"])
        # Dot Distance   
        pl_module.log(f"barlowtwins_dist_{phase}_Dot/Pos_attacked_both",ret["pos_dot_attacked_both"])        
        
        # Loss_invariance and Loss redundancy
        loss_invariance_both = getattr(pl_module, f"{phase}_barlowtwins_loss_invariance_both")(ret["barlowtwins_loss_invariance_both"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_invariance_both", loss_invariance_both)  
        loss_redundancy_both = getattr(pl_module, f"{phase}_barlowtwins_loss_redundancy_both")(ret["barlowtwins_loss_redundancy_both"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_redundancy_both", loss_redundancy_both)    
        
    return ret

def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret

def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret

def compute_vqa_attack(pl_module, batch):

    if pl_module.image_view:
        attacked_batch_image = compute_pgd(pl_module, deepcopy(batch), "vqa_attacked")
        infer = pl_module.infer(attacked_batch_image, mask_text=False, mask_image=False)
    if pl_module.text_view:

        attacked_batch_text = compute_geometric(pl_module, deepcopy(batch), "vqa_attacked")
        if pl_module.image_view:
            attacked_batch_text["image"][0] = attacked_batch_image["image"][0]
        infer = pl_module.infer(attacked_batch_text, mask_text=False, mask_image=False)
    
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_attacked_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_attacked_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa_attacked/{phase}/loss", loss)
    pl_module.log(f"vqa_attacked/{phase}/score", score)

    return ret


def compute_vqa(pl_module, batch):         
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret

def compute_nlvr2_attack(pl_module, batch):
    ret = {}
    infer1 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
    infer2 = pl_module.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)
    # NlVR2 output clean
    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    ori_logits = pl_module.nlvr2_classifier(cls_feats)
    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    ori_loss = F.cross_entropy(ori_logits, nlvr2_labels)
    ret["nlvr2_original_logits"] = ori_logits
    ret["nlvr2_original_loss"] = ori_loss
    ret["nlvr2_labels"] = nlvr2_labels
    attack_batch = None
    if pl_module.image_view:   
        attacked_batch_image = compute_pgd(pl_module, deepcopy(batch), "nlvr2_attacked")
        
        infer1_a = pl_module.infer(attacked_batch_image, mask_text=False, mask_image=False, image_token_type_idx=1)
        infer2_a = pl_module.infer(attacked_batch_image, mask_text=False, mask_image=False, image_token_type_idx=2)
        
        cls_feats = torch.cat([infer1_a["cls_feats"], infer2_a["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

        ret["nlvr2_attacked_logits"] = nlvr2_logits
        ret["nlvr2_attacked_loss"] = nlvr2_loss
        
    if pl_module.text_view:
        
        attacked_batch_text = compute_geometric(pl_module, deepcopy(batch), "nlvr2_attacked")
        if pl_module.image_view:
            attacked_batch_text["image_0"][0] = attacked_batch_image["image_0"][0]
            attacked_batch_text["image_1"][0] = attacked_batch_image["image_1"][0]
            
        infer1_a = pl_module.infer(attacked_batch_text, mask_text=False, mask_image=False, image_token_type_idx=1)
        infer2_a = pl_module.infer(attacked_batch_text, mask_text=False, mask_image=False, image_token_type_idx=2)
        cls_feats = torch.cat([infer1_a["cls_feats"], infer2_a["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

        ret["nlvr2_attacked_logits"] = nlvr2_logits
        ret["nlvr2_attacked_loss"] = nlvr2_loss
        
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        # To Do
        loss = getattr(pl_module, f"{phase}_nlvr2_attacked_loss")(ret["nlvr2_attacked_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_attacked_accuracy")(
            ret["nlvr2_attacked_logits"], ret["nlvr2_attacked_labels"]
        )
        pl_module.log(f"nlvr2_attacked/{phase}/loss", loss)
        pl_module.log(f"nlvr2_attacked/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_original_loss")(
                F.cross_entropy(ret["nlvr2_original_logits"][dev_batches], ret["nlvr2_labels"][dev_batches])
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_original_accuracy")(
                ret["nlvr2_original_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2_original/dev/loss", dev_loss)
            pl_module.log(f"nlvr2_original/dev/accuracy", dev_acc)
            if pl_module.image_view or pl_module.text_view:
                dev_loss = getattr(pl_module, f"dev_nlvr2_attacked_loss")(
                    F.cross_entropy(ret["nlvr2_attacked_logits"][dev_batches], ret["nlvr2_labels"][dev_batches])
                )
                dev_acc = getattr(pl_module, f"dev_nlvr2_attacked_accuracy")(
                    ret["nlvr2_attacked_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
                pl_module.log(f"nlvr2_attacked/dev/loss", dev_loss)
                pl_module.log(f"nlvr2_attacked/dev/accuracy", dev_acc)
                change_rate_cross = getattr(pl_module, f"dev_nlvr2_attacked_change_rate_cross")(
                    ret["nlvr2_attacked_logits"][dev_batches], ret["nlvr2_original_logits"][dev_batches]
                )
                pl_module.log(f"nlvr2_attacked/dev/change_rate_cross", change_rate_cross)               

        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_original_loss")(
                F.cross_entropy(ret["nlvr2_original_logits"][test_batches], ret["nlvr2_labels"][test_batches])
            )
            test_acc = getattr(pl_module, f"test_nlvr2_original_accuracy")(
                ret["nlvr2_original_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2_original/test/loss", test_loss)
            pl_module.log(f"nlvr2_original/test/accuracy", test_acc)
            if pl_module.image_view or pl_module.text_view:
                test_loss = getattr(pl_module, f"test_nlvr2_attacked_loss")(
                    F.cross_entropy(ret["nlvr2_attacked_logits"][test_batches], ret["nlvr2_labels"][test_batches])
                )
                test_acc = getattr(pl_module, f"test_nlvr2_attacked_accuracy")(
                    ret["nlvr2_attacked_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
                pl_module.log(f"nlvr2_attacked/test/loss", test_loss)
                pl_module.log(f"nlvr2_attacked/test/accuracy", test_acc)
                change_rate_cross = getattr(pl_module, f"test_nlvr2_attacked_change_rate_cross")(
                    ret["nlvr2_attacked_logits"][test_batches], ret["nlvr2_original_logits"][test_batches]
                )
                pl_module.log(f"nlvr2_attacked/test/change_rate_cross", change_rate_cross)             

    return ret

def compute_nlvr2(pl_module, batch):
   
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )
    # NlVR2 output
    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)
    # Compute the cross-entropy
    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss   = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret

def compute_irtr_attacked(pl_module, batch):
    is_training_phase = pl_module.training
    ret = {}

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    #score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = pl_module.moco_head(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)
    ret["irtr_original_loss"] = irtr_loss
    ret["irtr_original_logits"] = score
    
    if pl_module.image_view:
        batch = compute_pgd(pl_module, deepcopy(batch), "irtr_attacked")
        _bs, _c, _h, _w = batch["image"][0].shape
        false_len = pl_module.hparams.config["draw_false_text"]
        text_ids = torch.stack(
            [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
        )
        text_masks = torch.stack(
            [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
        )
        text_labels = torch.stack(
            [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
        )

        text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
        text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
        text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
        images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

        infer = pl_module.infer(
            {
                "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
                "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
                "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
                "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
            }
        )
        score = pl_module.moco_head(infer["cls_feats"])[:, 0]
        score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
        irtr_loss = F.cross_entropy(score, answer)
        ret["irtr_attacked_loss"] = irtr_loss
        ret["irtr_attacked_logits"] = score

    if pl_module.text_view:
        batch = compute_geometric(pl_module, deepcopy(batch), "irtr_attacked")
        _bs, _c, _h, _w = batch["image"][0].shape
        false_len = pl_module.hparams.config["draw_false_text"]
        text_ids = torch.stack(
            [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
        )
        text_masks = torch.stack(
            [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
        )
        text_labels = torch.stack(
            [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
        )
    
        text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
        text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
        text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
        images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)
    
        infer = pl_module.infer(
            {
                "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
                "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
                "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
                "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
            }
        )
        score = pl_module.moco_head(infer["cls_feats"])[:, 0]
        score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
        irtr_loss = F.cross_entropy(score, answer)
        ret["irtr_attacked_loss"] = irtr_loss
        ret["irtr_attacked_logits"] = score

    phase = "train" if pl_module.training else "val"
    
    irtr_loss = getattr(pl_module, f"{phase}_irtr_original_loss")(ret["irtr_original_loss"])
    pl_module.log(f"irtr_attacked/{phase}/irtr_original_loss", irtr_loss)
    acc = getattr(pl_module, f"{phase}_irtr_original_accuracy")(ret["irtr_original_logits"], answer)
    pl_module.log(f"irtr_attacked/{phase}/original_accuracy", acc)
    
    if pl_module.text_view or pl_module.image_view:
        irtr_loss = getattr(pl_module, f"{phase}_irtr_attacked_loss")(ret["irtr_attacked_loss"])
        pl_module.log(f"irtr_attacked/{phase}/irtr_attacked_loss", irtr_loss)
        acc = getattr(pl_module, f"{phase}_irtr_attacked_accuracy")(ret["irtr_attacked_logits"], answer)
        pl_module.log(f"irtr_attacked/{phase}/attacked_accuracy", acc)
    
    return ret

def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(max_num=500)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True,
        max_num=500
    ) # Here something Weird
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)

@torch.no_grad()
def compute_attacked_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(max_num=20)
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True,
        max_num=20
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))
    
    text_preload_attacked = list()
    image_preload_attacked = list()
    if pl_module.text_view or pl_module.image_view:
        for _b in tqdm.tqdm(text_loader, desc="attacked data prefetch loop"):
            _b = {
                "text": _b["text"],
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
            }
            if pl_module.text_view:
                _b = compute_geometric(pl_module, deepcopy(_b), "irtr")
                text_preload_attacked.append(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                        "text_labels": _b["text_labels"].to(pl_module.device),
                        "img_index": _b["img_index"],
                    }
                )
            if pl_module.image_view:
                _b = compute_pgd(pl_module, deepcopy(_b), "irtr")
                (ie, im, _, _) = pl_module.transformer.visual_embed(
                    _b["image"][0],
                    max_image_len=pl_module.hparams.config["max_image_len"],
                    mask_it=False,
                )
                image_preload_attacked.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)    
    
    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)
    print(scores)
    print(iids)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Sequential):
        for sub_layer in module:
            init_weights(sub_layer)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name,config):
    image = config["image_view"]
    text  = config["text_view"]
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}_image_{image}_text_{text}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")