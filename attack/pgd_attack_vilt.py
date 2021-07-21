import torch
from copy import deepcopy
import torch.nn as nn


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
    def pgd_attack(self, pl_module, batch, k_text):
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
            img = batch[imgkey][0]  # [0] : Because it's a list of one element
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
        # image_embeds.shape : [64 217 768] :: [batch,patch,hiddensize]
        # patch_index shape  : ([64 217 2]), (19,19)) (patch_index, (H,W))
        
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
    
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.moco_head.zero_grad()
    
    def pgd_attack(self, pl_module, batch, k_text):
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
            with torch.enable_grad():
                try:
                    batch['image'][0] = (img_init + img_delta)  # .to(pl_module.device)
                    infer = self.infer(batch, mask_text=False, mask_image=False)
                    image_representation_q, text_representation_q = self.moco_head(infer['image_feats'], infer['text_feats'])
                    q_attacked = nn.functional.normalize(image_representation_q, dim=1)
                except:
                    print("problem in step ", astep)
                    sys.exit("STOPP")
                # RMCL Loss
                l_pos = torch.einsum('nc,nc->n', [q_attacked, k_text]).unsqueeze(-1)
                l_neg = torch.einsum('nc,ck->nk', [q_attacked, self.pl_module.image_queue.clone().detach()])
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
   
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.barlowtwins_head.zero_grad()
    
    def pgd_attack(self, pl_module, batch, k_text=None):
        
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
            with torch.enable_grad():
                batch['image'][0] = (img_init + img_delta)  # .to(pl_module.device)
                infer = self.infer(batch, mask_text=False, mask_image=False)
                image_representation, text_representation = self.barlowtwins_head(infer['image_feats'], infer['text_feats'])
                # RMCL Loss
                c = image_representation.T @ text_representation

                # c.div_(pl_module.per_step_bs)
                # torch.distributed.all_reduce(c)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()

                loss = (on_diag + pl_module.adv_lr * off_diag) / self.adv_steps_img * pl_module.loss_weight
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
        self.pooler = deepcopy(pl_module.pooler)
        self.nlvr2_classifier = deepcopy(pl_module.nlvr2_classifier)
    
    def vilt_zero_grad(self):
        self.text_embeddings.zero_grad()
        self.transformer.zero_grad()
        self.token_type_embeddings.zero_grad()
        self.pooler.zero_grad()
        self.nlvr2_classifier.zero_grad()
    
    def pgd_attack(self, pl_module, batch, k_text=None):
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
            with torch.enable_grad():
                batch['image_0'][0] = (img_init_0 + img_delta_0)  # .to(pl_module.device)
                batch['image_1'][0] = (img_init_1 + img_delta_1)
                infer1 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=1)
                infer2 = self.infer(batch, mask_text=False, mask_image=False, image_token_type_idx=2)
                # NlVR2 output
                cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
                nlvr2_logits = self.nlvr2_classifier(cls_feats)
                # Compute the cross-entropy
                nlvr2_labels = batch["answers"]
                nlvr2_labels = torch.tensor(nlvr2_labels).to(self.pl_module.device).long()
                loss = loss_fct(nlvr2_logits, nlvr2_labels)
                # calculate x.grad
                loss.backward()
            
            if self.attack_idx[0]:
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
        
        return img_delta_0, img_delta_1
