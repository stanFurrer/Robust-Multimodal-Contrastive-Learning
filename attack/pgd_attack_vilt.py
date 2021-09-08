import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os 
class PGDAttack:
    def __init__(self, config, contrastive_framework):
        self.contrastive_framework = contrastive_framework
        self.adv_steps_img = config["adv_steps_img"]
        self.adv_lr_img = config["adv_lr_img"]
        self.adv_max_norm_img = config["adv_max_norm_img"]
        # a mini ViLTransformerSS
        self.pl_module = None
        self.max_image_len = config["max_image_len"]
        self.text_embeddings = None
        self.transformer = None
        self.token_type_embeddings = None
        self.pooler = None
    
    
    def build_mini_vilt(self, pl_module):
        raise NotImplementedError(f"Build_mini_vilt of {self.contrastive_framework} isn't implemented.")
    def vilt_zero_grad(self):
        raise NotImplementedError(f"vilt_zero_grad of {self.contrastive_framework} isn't implemented.")
    def pgd_attack(self, pl_module, batch, k_image):
        raise NotImplementedError(f"pgd_attack of {self.contrastive_framework} isn't implemented.")

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
    
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
    
        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.max_image_len,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)
        
        x = co_embeds
        
        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
        
        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1]:],
        )
        if self.pooler is not None:
            cls_feats = self.pooler(x)
        else:
            cls_feats = None
    
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }
        
        return ret


class PGDAttack_moco(PGDAttack):
    def __init__(self, config):
        super().__init__(config, "moco")
        # a mini ViLTransformerSS
        self.moco_head = None

    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.moco_head = deepcopy(pl_module.moco_head)
        self.pooler = deepcopy(pl_module.pooler) 
        
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.moco_head.zero_grad()
        self.pooler.zero_grad() 
    
    def pgd_attack(self, pl_module, batch, k_modality = None):
        self.build_mini_vilt(pl_module)
        loss_fct = nn.CrossEntropyLoss()
        # Get the original img
        img_init = batch['image'][0]
        # Initialize the delta as zero vectors
        img_delta = torch.zeros_like(img_init)
        self.vilt_zero_grad()
        for astep in range(self.adv_steps_img):
            # Need to get the gradient for each batch of image features
            img_delta.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=False):
                with torch.enable_grad():
                    try:
                        batch['image'][0] = (img_init + img_delta) 
                        infer = self.infer(batch, mask_text=False, mask_image=False)
                        projection_cls_feats = self.moco_head(infer["cls_feats"])
                        q_img_attack = nn.functional.normalize(projection_cls_feats, dim=1)                    
                    except:
                        print("problem in step ", astep)
                        sys.exit("STOPP")
                    # RMCL Loss 
                    l_pos = torch.einsum('nc,nc->n', [q_img_attack, k_modality]).unsqueeze(-1) # k_image
                    l_neg = torch.einsum('nc,ck->nk', [q_img_attack, self.pl_module.proj_queue.clone().detach()])
                    logits = torch.cat([l_pos, l_neg], dim=1)
                    logits /= self.pl_module.temperature
                    labels = torch.zeros(logits.shape[0], dtype=torch.long)
                    labels = labels.type_as(logits)
                    loss = loss_fct(logits.float(), labels.long()) / (1.0 * self.adv_steps_img)
                    # calculate x.grad
                    loss.backward()
                # Get gradient
                img_delta_grad = img_delta.grad.clone().detach().float()
                # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
                denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1, 1)
                # Clip gradient to Lower Bound
                denorm = torch.clamp(denorm, min=1e-8)
                # calculate delta_step  with format img_delta
                img_delta_step = (self.adv_lr_img * img_delta_grad / denorm).to(img_delta)
                # Add the calculated step to img_delta (The perturbation)
                img_delta = (img_delta + img_delta_step).detach()
                # clip the delta if needed
                if self.adv_max_norm_img > 0:
                    img_delta = torch.clamp(img_delta, -self.adv_max_norm_img, self.adv_max_norm_img).detach()
                
        return img_delta

    
class PGDAttack_bartlowtwins(PGDAttack):
    def __init__(self, config):
        super().__init__(config, "barlowtwins")
        # a mini ViLTransformerSS
        self.barlowtwins_head = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.barlowtwins_head = deepcopy(pl_module.barlowtwins_head)
        self.pooler = deepcopy(pl_module.pooler) 
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.barlowtwins_head.zero_grad()
        self.pooler.zero_grad() 
        
    def pgd_attack(self, pl_module, batch, k_modality=None): 
        
        def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
        self.build_mini_vilt(pl_module)
        # Get the original img
        img_init = batch['image'][0]
        # Initialize the delta as zero vectors
        img_delta = torch.zeros_like(img_init)
        self.vilt_zero_grad()
        for astep in range(self.adv_steps_img):
            # Need to get the gradient for each batch of image features
            img_delta.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=False):
                with torch.enable_grad():
                    batch['image'][0] = (img_init + img_delta)
                    infer = self.infer(batch, mask_text=False, mask_image=False)
                    q_image = self.barlowtwins_head(infer['cls_feats'])
                    c = torch.mm(q_image.to(torch.float32).T, k_modality.to(torch.float32)) / q_image.shape[0]
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).pow_(2).sum()

                    loss = (on_diag + pl_module.adv_lr * off_diag) / self.adv_steps_img
                    loss.backward()
                # Get gradient
                img_delta_grad = img_delta.grad.clone().detach().float()
                # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
                denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1, 1)
                # Clip gradient to Lower Bound
                denorm = torch.clamp(denorm, min=1e-8)
                # calculate delta_step  with format img_delta
                img_delta_step = (self.adv_lr_img * img_delta_grad / denorm).to(img_delta)
                # Add the calculated step to img_delta (The perturbation)
                img_delta = (img_delta + img_delta_step).detach()
                # clip the delta if needed
                if self.adv_max_norm_img > 0:
                    img_delta = torch.clamp(img_delta, -self.adv_max_norm_img, self.adv_max_norm_img).detach()
        
        return img_delta
number = 0
class PGDAttack_nlvr2(PGDAttack):
    def __init__(self, config):
        super().__init__(config, "nlvr2")
        self.attack_idx = config["attack_idx"]
        # a mini ViLTransformerSS
        self.nlvr2_classifier = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.nlvr2_classifier = deepcopy(pl_module.nlvr2_classifier)
        self.pooler = deepcopy(pl_module.pooler)
        
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.nlvr2_classifier.zero_grad()
        self.pooler.zero_grad()

    def pgd_attack(self, pl_module, batch, k_modality=None):
        # To save an img_delta
        save_delta = {}
        global number
        number += 1
        self.build_mini_vilt(pl_module)

        loss_fct = nn.CrossEntropyLoss()
        # Get the original img
        img_init_0 = batch['image_0'][0]
        img_init_1 = batch['image_1'][0]
        # Initialize the delta as zero vectors
        img_delta_0 = torch.zeros_like(img_init_0)
        img_delta_1 = torch.zeros_like(img_init_1)
        self.vilt_zero_grad()
        for astep in range(self.adv_steps_img):
            # Need to get the gradient for each batch of image features
            if self.attack_idx[0]:
                img_delta_0.requires_grad_(True)
            if self.attack_idx[1]:
                img_delta_1.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=False):
                with torch.enable_grad():
                    batch['image_0'][0] = (img_init_0 + img_delta_0)
                    batch['image_1'][0] = (img_init_1 + img_delta_1)
                    infer1 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
                    infer2 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)
                    # NlVR2 output
                    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
                    nlvr2_logits = self.nlvr2_classifier(cls_feats)
                    # Compute the cross-entropy
                    nlvr2_labels = batch["answers"]
                    nlvr2_labels = torch.tensor(nlvr2_labels).to(self.pl_module.device).long()
                    loss = loss_fct(nlvr2_logits, nlvr2_labels)/self.adv_steps_img
                    # calculate x.grad
                    loss.backward() 

                if self.attack_idx[0]:
                    verbose = False
                    # Get gradient
                    img_delta_grad = img_delta_0.grad.clone().detach().float()
                    # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
                    denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1, 1)
                    # Clip gradient to Lower Bound
                    denorm = torch.clamp(denorm, min=1e-8)
                    # calculate delta_step  with format img_delta
                    img_delta_step = (self.adv_lr_img * img_delta_grad / denorm).to(img_delta_0)
                    # Add the calculated step to img_delta (The perturbation)
                    img_delta_0 = (img_delta_0 + img_delta_step).detach()
                    # clip the delta if needed
                    if self.adv_max_norm_img > 0:
                        img_delta_0 = torch.clamp(img_delta_0, -self.adv_max_norm_img, self.adv_max_norm_img).detach()

                if self.attack_idx[1]:
                    # Get gradient
                    img_delta_grad = img_delta_1.grad.clone().detach().float()
                    # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
                    denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1, 1)
                    # Clip gradient to Lower Bound
                    denorm = torch.clamp(denorm, min=1e-8)
                    # calculate delta_step  with format img_delta
                    img_delta_step = (self.adv_lr_img * img_delta_grad / denorm).to(img_delta_1)
                    # Add the calculated step to img_delta (The perturbation)
                    img_delta_1 = (img_delta_1 + img_delta_step).detach()
                    # clip the delta if needed
                    if self.adv_max_norm_img > 0:
                        img_delta_1 = torch.clamp(img_delta_1, -self.adv_max_norm_img, self.adv_max_norm_img).detach()

                    ## Save Img_delta
                    verbose = False
                    if verbose == True : 
                        save = "./SLURM/vector_norm_delta2"
                        norm_delta = torch.linalg.norm(img_delta_1, dim=1).mean().cpu()
                        save_delta["step_{}".format(astep)] = norm_delta

            if verbose == True :
                with open(os.path.join(save,"{}.pkl".format(number)), "wb") as fp:
                    pickle.dump(save_delta, fp,protocol=pickle.HIGHEST_PROTOCOL)  

        return img_delta_0, img_delta_1     

class PGDAttack_irtr(PGDAttack):
    def __init__(self, config):
        super().__init__(config, "moco")
        # a mini ViLTransformerSS
        self.moco_head = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.moco_head = deepcopy(pl_module.moco_head)
        self.pooler = deepcopy(pl_module.pooler)
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.moco_head.zero_grad()
        self.pooler.zero_grad() 
        
    def pgd_attack(self, pl_module, batch, k_modality):
        self.build_mini_vilt(pl_module)
        loss_fct = nn.CrossEntropyLoss()
        # Get the original img
        img_init = batch['image'][0]
        # Initialize the delta as zero vectors
        img_delta = torch.zeros_like(img_init)
        self.vilt_zero_grad()
        for astep in range(self.adv_steps_img):
            # Need to get the gradient for each batch of image features
            img_delta.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=False):
                with torch.enable_grad():
                    try:

                        batch['image'][0] = (img_init + img_delta)  # .to(pl_module.device)
                        infer = self.infer(batch, mask_text=False, mask_image=False)
                        projection_cls_feats = self.moco_head(infer["cls_feats"])
                        q_img_attack = nn.functional.normalize(projection_cls_feats, dim=1)                     
                    except:
                        print("problem in step ", astep)
                        sys.exit("STOPP")

                    batch_scores = []
                    batch_labels = []

                    for q_idx, q_att in enumerate(q_img_attack): ## Wrong
                        scores = torch.einsum('nc,ck->nk', [q_att.unsqueeze(0), text_representation.T])
                        batch_scores.append(scores)
                        batch_labels.append(q_idx)
                    logits = torch.cat(batch_scores).view(len(batch_labels), -1)
                    labels = torch.tensor(batch_labels).type_as(logits)
                    loss = loss_fct(logits.float(), labels.long()) / (1.0 * self.adv_steps_img)
                    # print(loss)
                    # calculate x.grad
                    loss.backward()
                # Get gradient
                img_delta_grad = img_delta.grad.clone().detach().float()
                # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
                denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1,
                                                                                                                 1)
                # Clip gradient to Lower Bound
                denorm = torch.clamp(denorm, min=1e-8)
                # calculate delta_step  with format img_delta
                img_delta_step = (self.adv_lr_img * img_delta_grad / denorm).to(img_delta)
                # Add the calculated step to img_delta (The perturbation)
                img_delta = (img_delta + img_delta_step).detach()
                # clip the delta if needed
                if self.adv_max_norm_img > 0:
                    img_delta = torch.clamp(img_delta, -self.adv_max_norm_img, self.adv_max_norm_img).detach()
        
        return img_delta       

    
class PGDAttack_vqa(PGDAttack):
    def __init__(self, config):
        super().__init__(config, "vqa")
        # a mini ViLTransformerSS
        self.vqa_classifier = None
    
    def build_mini_vilt(self, pl_module):
        self.pl_module = pl_module
        self.text_embeddings = deepcopy(pl_module.text_embeddings)
        self.token_type_embeddings = deepcopy(pl_module.token_type_embeddings)
        self.transformer = deepcopy(pl_module.transformer)
        self.pooler = deepcopy(pl_module.pooler)
        self.vqa_classifier = deepcopy(pl_module.vqa_classifier)
        
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.pooler.zero_grad()
        self.vqa_classifier.zero_grad()
        
    def pgd_attack(self, pl_module, batch, k_modality=None):
        self.build_mini_vilt(pl_module)
        loss_fct = nn.CrossEntropyLoss()
        # Get the original img
        img_init = batch['image'][0]
        # Initialize the delta as zero vectors
        img_delta = torch.zeros_like(img_init)
        self.vilt_zero_grad()
        for astep in range(self.adv_steps_img):
            # Need to get the gradient for each batch of image features
            img_delta.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=False):            
                with torch.enable_grad():
                    batch['image'][0] = (img_init + img_delta)  # .to(pl_module.device)
                    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
                    # vqa output
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
                    )                
                    vqa_loss.backward()

                # Get gradient
                img_delta_grad = img_delta.grad.clone().detach().float()
                # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
                denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1, 1)
                # Clip gradient to Lower Bound
                denorm = torch.clamp(denorm, min=1e-8)
                # calculate delta_step  with format img_delta
                img_delta_step = (self.adv_lr_img * img_delta_grad / denorm).to(img_delta)
                # Add the calculated step to img_delta (The perturbation)
                img_delta = (img_delta + img_delta_step).detach()
                # clip the delta if needed
                if self.adv_max_norm_img > 0:
                    img_delta = torch.clamp(img_delta, -self.adv_max_norm_img, self.adv_max_norm_img).detach()
        return img_delta  