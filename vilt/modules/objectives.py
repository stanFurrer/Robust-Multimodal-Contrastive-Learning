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
    
def compute_pgd(pl_module, batch, loss_name, k_modality=None):
    img_delta = pl_module.pgd_attacker.pgd_attack(pl_module, batch, k_modality = k_modality)
    # add debug code here
    if loss_name == "nlvr2_attacked":
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
        
    #    for i in range(len(batch["text"])):
    #        print("Real sentence----: ",real_sentence[i])
    #        print("Attacked sentence: ",attack_words["text"][i])

     #   txt_original_attacked   = {"original": real_sentence,
     #                              "attacked": attack_words["text"]
     #                             }
        
    batch["text"] = attack_words["text"]
    batch["text_ids"] = attack_words["txt_input_ids"]
    batch["text_masks"] = attack_words["text_masks"]

    phase = "train" if pl_module.training else "val"
    pl_module.log(f"{loss_name}_attack/{phase}/num_changes", attack_words["num_changes"])
    pl_module.log(f"{loss_name}_attack/{phase}/change_rate", attack_words["change_rate"])
    
    return batch #, txt_original_attacked

def compute_moco_contrastive(pl_module, batch,batch_idx):
    
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

    def _dequeue_and_enqueue(keys, queue_type):
        """ dequeue and euqueue the new batch of text AND image representation"""
        keys = _concat_all_gather(keys)
        batch_size = keys.shape[0]
        if not (pl_module.per_step_bs == batch_size):
            return
        if queue_type == 'text':
            ptr = int(pl_module.text_queue_ptr)
            # assert pl_module.num_negative % batch_size == 0
            pl_module.text_queue[:, ptr:ptr+batch_size] = keys.T
            ptr = (ptr + batch_size) % pl_module.num_negative
            pl_module.text_queue_ptr[0] = ptr
        if queue_type == 'image':
            ptr = int(pl_module.image_queue_ptr)
            # assert pl_module.num_negative % batch_size == 0
            pl_module.image_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % pl_module.num_negative
            pl_module.image_queue_ptr[0] = ptr

    loss = 0
    loss_num = 0
    loss_fct = nn.CrossEntropyLoss()
    ret = {}

    # momentum update key encoder
    _momentum_update_key_layer(pl_module.momentum, pl_module.text_embeddings, pl_module.k_text_embeddings)
    _momentum_update_key_layer(pl_module.momentum, pl_module.token_type_embeddings, pl_module.k_token_type_embeddings)
    _momentum_update_key_layer(pl_module.momentum, pl_module.transformer, pl_module.k_transformer)
    _momentum_update_key_layer(pl_module.momentum, pl_module.moco_head, pl_module.k_moco_head)

    with torch.no_grad():
        infer_k = pl_module.infer_k(batch, mask_text=False, mask_image=False)
        image_representation_k, text_representation_k = pl_module.k_moco_head(infer_k['image_feats'], infer_k['text_feats'])
        k_text = nn.functional.normalize(text_representation_k, dim=1)
        k_image = nn.functional.normalize(image_representation_k, dim=1)

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    image_representation, text_representation = pl_module.moco_head(infer['image_feats'], infer['text_feats'])
    original_text = nn.functional.normalize(text_representation, dim=1)
    original_image = nn.functional.normalize(image_representation, dim=1)
    neg_img = pl_module.image_queue.clone().detach()
    neg_txt = pl_module.text_queue.clone().detach()        
        
    if pl_module.image_view:
        if pl_module.augmentation :            
            augmented_batch = image_augmentation(pl_module, deepcopy(batch))
        else : 
            augmented_batch = compute_pgd(pl_module, deepcopy(batch), "moco", k_modality=k_text)
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        image_representation_q, text_representation_q = pl_module.moco_head(infer['image_feats'], infer['text_feats'])
        q = nn.functional.normalize(image_representation_q, dim=1)

        # attacked image: close to the same image before attack; away from different images.
        l_pos = torch.einsum('nc,nc->n', [q, k_image]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, neg_img])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature
    
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        ret["image_image_pos_dist"] = torch.linalg.norm(q-k_image, dim=1).mean()
        ret["image_image_pos_cosine"] = pl_module.cosine(q,k_image).mean()
        dist = 0
        cosine = 0
        for sub_q in q:
            dist += torch.linalg.norm(sub_q-neg_img.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_img.T).mean()
        ret["image_image_neg_dist"] = dist / q.shape[0]
        ret["image_image_neg_cosine"] = cosine / q.shape[0]
        
        loss_image_image = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/img_img_loss", loss_image_image)
        loss = loss + loss_image_image
        loss_num += 1
        
        # attacked image: close to the corresponding text; away from other text
        l_pos = torch.einsum('nc,nc->n', [q, k_text]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, neg_txt])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
        
        ret["image_text_pos_dist"] = torch.linalg.norm(q - k_text, dim=1).mean()
        ret["image_text_pos_cosine"] = pl_module.cosine(q,k_text).mean()
        dist = 0
        cosine = 0
        for sub_q in q:
            dist += torch.linalg.norm(sub_q - neg_txt.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_txt.T).mean()
        ret["image_text_neg_dist"] = dist / q.shape[0]
        ret["image_text_neg_cosine"] = cosine / q.shape[0]
        # ret["image_text_logits"] = logits
        # ret["image_text_labels"] = labels

        loss_image_text = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/img_txt_loss", loss_image_text)
        loss = loss + loss_image_text
        loss_num += 1             
        
    if pl_module.text_view:
        if pl_module.augmentation : 
            augmented_batch = text_augmentation(pl_module, deepcopy(batch))
        else :   
            augmented_batch = compute_geometric(pl_module, deepcopy(batch), "moco", k_modality = k_image)
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        image_representation_q, text_representation_q = pl_module.moco_head(infer['image_feats'], infer['text_feats'])
        q = nn.functional.normalize(text_representation_q, dim=1)
        
        # attacked text: close to the same text; away from different text
        neg_txt = pl_module.text_queue.clone().detach()
        l_pos = torch.einsum('nc,nc->n', [q, k_text]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, neg_txt])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature
    
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
        
        ret["text_text_pos_dist"] = torch.linalg.norm(q - k_text, dim=1).mean()
        ret["text_text_pos_cosine"] = pl_module.cosine(q,k_text).mean()
        dist = 0
        cosine = 0
        for sub_q in q:
            dist += torch.linalg.norm(sub_q - neg_txt.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_txt.T).mean()
        ret["text_text_neg_dist"] = dist / q.shape[0]
        ret["text_text_neg_cosine"] = cosine / q.shape[0]
        # ret["text_text_logits"] = logits
        # ret["text_text_labels"] = labels
        
        loss_text_text = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/txt_txt_loss", loss_text_text)
        loss = loss + loss_text_text
        loss_num += 1

        # attacked text: close to the corresponding image, away from other images
        l_pos = torch.einsum('nc,nc->n', [q, k_image]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, neg_img])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= pl_module.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)
        
        ret["text_image_pos_dist"] = torch.linalg.norm(q - k_image, dim=1).mean()
        ret["text_image_pos_cosine"] = pl_module.cosine(q,k_image).mean()
        dist = 0
        cosine = 0
        for sub_q in q:
            dist += torch.linalg.norm(sub_q - neg_img.T, dim=1).mean()
            cosine += pl_module.cosine(torch.unsqueeze(sub_q,0),neg_img.T).mean()
        ret["text_image_neg_dist"] = dist / q.shape[0]
        ret["text_image_neg_cosine"] = cosine / q.shape[0]        
        
        # ret["text_image_pos_dist"] = torch.linalg.norm(q-k_image).mean()
        # ret["text_image_neg_dist"] = torch.linalg.norm(q-neg_img, dim=1).mean()
        # ret["text_image_logits"] = logits
        # ret["text_image_labels"] = labels
        
        loss_text_image = loss_fct(logits.float(), labels.long())
        pl_module.log(f"moco_loss/txt_img__loss", loss_text_image)
        loss = loss + loss_text_image
        loss_num += 1
        
    if pl_module.training:
        _dequeue_and_enqueue(k_text, 'text')
        _dequeue_and_enqueue(k_image, 'image')

    if (batch_idx) % 1000 == 0 and pl_module.tsne_vizualisation: 
        batch_idx +=1
        print("--Computing TSNE--")
        nbr_element = 1000 * len(batch["text"])
        TSNE_projection(neg_img,neg_txt,nbr_element,batch_idx,pl_module.img_save_path)
    
    ret["moco_loss"] = loss / loss_num
    
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_moco_loss")(ret["moco_loss"])
    pl_module.log(f"moco_loss/step/{phase}", loss)
    
    pl_module.log(f"moco_dist_{phase}/Original/cosine/img_txt", pl_module.cosine(original_image,original_text).mean())
    
    if pl_module.image_view:          
        # L2 distance
        pl_module.log(f"moco_dist_{phase}_img_img/L2/Pos_img_img", ret["image_image_pos_dist"])
        pl_module.log(f"moco_dist_{phase}_img_img/L2/Neg_img_img", ret["image_image_neg_dist"])
        pl_module.log(f"moco_dist_{phase}_img_img/L2/Neg-Pos_img_img", ret["image_image_neg_dist"] - ret["image_image_pos_dist"])

        pl_module.log(f"moco_dist_{phase}_img_txt/L2/Pos_img_txt", ret["image_text_pos_dist"])
        pl_module.log(f"moco_dist_{phase}_img_txt/L2/Neg_img_txt", ret["image_text_neg_dist"])
        pl_module.log(f"moco_dist_{phase}_img_txt/L2/Neg-Pos_img_txt", ret["image_text_neg_dist"] - ret["image_text_pos_dist"])
        # Cosine distance
        pl_module.log(f"moco_dist_{phase}_img_img/cosine/Pos_img_img", ret["image_image_pos_cosine"])
        pl_module.log(f"moco_dist_{phase}_img_img/cosine/Neg_img_img", ret["image_image_neg_cosine"])
        pl_module.log(f"moco_dist_{phase}_img_img/cosine/Neg-Pos_img_img", ret["image_image_neg_cosine"] - ret["image_image_pos_cosine"])

        pl_module.log(f"moco_dist_{phase}_img_txt/cosine/Pos_img_txt", ret["image_text_pos_cosine"])
        pl_module.log(f"moco_dist_{phase}_img_txt/cosine/Neg_img_txt", ret["image_text_neg_cosine"])
        pl_module.log(f"moco_dist_{phase}_img_txt/cosine/Neg-Pos_img_txt", ret["image_text_neg_cosine"] - ret["image_text_pos_cosine"])        
        
    if pl_module.text_view:     
        # L2 distance
        pl_module.log(f"moco_dist_{phase}_txt_txt/L2/Pos_txt_txt", ret["text_text_pos_dist"])
        pl_module.log(f"moco_dist_{phase}_txt_txt/L2/Neg_txt_txt", ret["text_text_neg_dist"])
        pl_module.log(f"moco_dist_{phase}_txt_txt/L2/Neg-Pos_txt_txt", ret["text_text_neg_dist"] - ret["text_text_pos_dist"])

        pl_module.log(f"moco_dist_{phase}_txt_img/L2/Pos_txt_img", ret["text_image_pos_dist"])
        pl_module.log(f"moco_dist_{phase}_txt_img/L2/Neg_txt_img", ret["text_image_neg_dist"])
        pl_module.log(f"moco_dist_{phase}_txt_img/L2/Neg-Pos_txt_img", ret["text_image_neg_dist"] - ret["text_image_pos_dist"])
        # Cosine distance
        pl_module.log(f"moco_dist_{phase}_txt_txt/cosine/Pos_txt_txt", ret["text_text_pos_cosine"])
        pl_module.log(f"moco_dist_{phase}_txt_txt/cosine/Neg_txt_txt", ret["text_text_neg_cosine"])
        pl_module.log(f"moco_dist_{phase}_txt_txt/cosine/Neg-Pos_txt_txt", ret["text_text_neg_cosine"] - ret["text_text_pos_cosine"])

        pl_module.log(f"moco_dist_{phase}_txt_img/cosine/Pos_txt_img", ret["text_image_pos_cosine"])
        pl_module.log(f"moco_dist_{phase}_txt_img/cosine/Neg_txt_img", ret["text_image_neg_cosine"])
        pl_module.log(f"moco_dist_{phase}_txt_img/cosine/Neg-Pos_txt_img", ret["text_image_neg_cosine"] - ret["text_image_pos_cosine"])
        
    return ret

def compute_barlowtwins_contrastive(pl_module, batch,batch_idx):
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
        original_image, original_text = pl_module.barlowtwins_head(infer['image_feats'], infer['text_feats'])
    
    if pl_module.image_view:
        if pl_module.augmentation : 
            augmented_batch = image_augmentation(pl_module, deepcopy(batch))
        else : 
            if pl_module.multimodal :
                augmented_batch = compute_pgd(pl_module, deepcopy(batch), "barlowtwins",k_modality=original_text)
            else : 
                augmented_batch = compute_pgd(pl_module, deepcopy(batch), "barlowtwins",k_modality=original_image)
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        image_representation, text_representation = pl_module.barlowtwins_head(infer['image_feats'], infer['text_feats'])
        
        if pl_module.multimodal : 
            # Image, Text
            c_1 = image_representation.T @ original_text
        else : 
            # Image, Image
            c_1 = image_representation.T @ original_image
        
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
        for sub_img in image_representation:
            dist_image_neg_dist += torch.linalg.norm(sub_img - original_image, dim=1).mean()
            dist_text_neg_dist  += torch.linalg.norm(sub_img - original_text, dim=1).mean() 
            dist_image_neg_cosine += pl_module.cosine(torch.unsqueeze(sub_img,0),original_image).mean()
            dist_text_neg_cosine  += pl_module.cosine(torch.unsqueeze(sub_img,0),original_text).mean()
        
        ret["image_image_neg_dist"] = dist_image_neg_dist / image_representation.shape[0]
        ret["image_text_neg_dist"]  = dist_text_neg_dist / image_representation.shape[0]
        ret["image_image_neg_cosine"] = dist_image_neg_cosine / image_representation.shape[0]
        ret["image_text_neg_cosine"] = dist_text_neg_cosine / image_representation.shape[0]
        
        ret["image_image_dist"] = torch.linalg.norm(image_representation - original_image, dim=1).mean()
        ret["image_text_dist"] = torch.linalg.norm(image_representation - original_text, dim=1).mean()
        ret["image_image_cosine"] = pl_module.cosine(image_representation,original_image).mean()
        ret["image_text_cosine"] = pl_module.cosine(image_representation,original_text).mean()

    if pl_module.text_view:
        if pl_module.augmentation : 
            augmented_batch = text_augmentation(pl_module, deepcopy(batch))
        else : 
            if pl_module.multimodal :
                augmented_batch = compute_geometric(pl_module, deepcopy(batch), "barlowtwins",k_modality=original_image)
            else : 
                augmented_batch = compute_geometric(pl_module, deepcopy(batch), "barlowtwins",k_modality=original_text)
                
        infer = pl_module.infer(augmented_batch, mask_text=False, mask_image=False)
        image_representation, text_representation = pl_module.barlowtwins_head(infer['image_feats'], infer['text_feats'])
         
        if pl_module.multimodal :
            # Text, Image
            c_2 = text_representation.T @ original_image   
        else : 
            # Text, Text
            c_2 = text_representation.T @ original_text

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
        for sub_text in text_representation:
            dist_image_neg_dist += torch.linalg.norm(sub_text - original_image, dim=1).mean()
            dist_text_neg_dist  += torch.linalg.norm(sub_text - original_text, dim=1).mean() 
            dist_image_neg_cosine += pl_module.cosine(torch.unsqueeze(sub_text,0),original_image).mean()
            dist_text_neg_cosine += pl_module.cosine(torch.unsqueeze(sub_text,0),original_text).mean()             
        
        ret["text_image_neg_dist"] = dist_image_neg_dist / text_representation.shape[0]
        ret["text_text_neg_dist"]  = dist_text_neg_dist / text_representation.shape[0]
        ret["text_image_neg_cosine"] = dist_image_neg_cosine / text_representation.shape[0]
        ret["text_text_neg_cosine"]  = dist_text_neg_cosine / text_representation.shape[0]
        
        ret["text_text_dist"] = torch.linalg.norm(text_representation - original_text, dim=1).mean()
        ret["text_image_dist"] = torch.linalg.norm(text_representation - original_image, dim=1).mean()
        ret["text_text_cosine"] = pl_module.cosine(text_representation,original_text).mean()
        ret["text_image_cosine"] = pl_module.cosine(text_representation,original_image).mean()
        
        
    ret["barlowtwins_loss"] = loss / loss_num  # * pl_module.loss_weight

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_barlowtwins_loss")(ret["barlowtwins_loss"])
    pl_module.log(f"barlowtwins/{phase}/loss", loss)
    
    pl_module.log(f"barlowtwins_dist_{phase}/Original/cosine/img_txt", pl_module.cosine(original_image,original_text).mean())
    if pl_module.image_view:
        # L2 distance
        pl_module.log(f"barlowtwins_dist_{phase}_img_img/L2/Pos_img_img", ret["image_image_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_img/L2/Neg_img_img", ret["image_image_neg_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_img/L2/Neg-Pos_img_img", ret["image_image_neg_dist"]-ret["image_image_dist"])    
        
        pl_module.log(f"barlowtwins_dist_{phase}_img_txt/L2/Pos_img_txt", ret["image_text_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_txt/L2/Neg_img_txt", ret["image_text_neg_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_txt/L2/Neg-Pos_img_txt", ret["image_text_neg_dist"]-ret["image_text_dist"])    
        
        # Cosine Distance
        pl_module.log(f"barlowtwins_dist_{phase}_img_img/cosine/Pos_img_img", ret["image_image_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_img/cosine/Neg_img_img", ret["image_image_neg_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_img/cosine/Neg-Pos_img_img", ret["image_image_neg_cosine"]-ret["image_image_cosine"])    
        
        pl_module.log(f"barlowtwins_dist_{phase}_img_txt/cosine/Pos_img_txt", ret["image_text_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_txt/cosine/Neg_img_txt", ret["image_text_neg_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_img_txt/cosine/Neg-Pos_img_txt", ret["image_text_neg_cosine"]-ret["image_text_cosine"])           
        
        # Loss_invariance and Loss redundancy
        loss_invariance_img = getattr(pl_module, f"{phase}_barlowtwins_loss_invariance_img")(ret["barlowtwins_loss_invariance_img"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_invariance_img", loss_invariance_img)  
        loss_redundancy_img = getattr(pl_module, f"{phase}_barlowtwins_loss_redundancy_img")(ret["barlowtwins_loss_redundancy_img"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_redundancy_img", loss_redundancy_img)  
        
        
    if pl_module.text_view:
        # L2 distance
        pl_module.log(f"barlowtwins_dist_{phase}_txt_txt/L2/Pos_txt_txt", ret["text_text_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_txt_txt/L2/Neg_txt_txt", ret["text_text_neg_dist"])        
        pl_module.log(f"barlowtwins_dist_{phase}_txt_txt/L2/Neg-Pos_txt_txt", ret["text_text_neg_dist"]-ret["text_text_dist"])          
        pl_module.log(f"barlowtwins_dist_{phase}_txt_img/L2/Pos_txt_img", ret["text_image_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_txt_img/L2/Neg_txt_img", ret["text_image_neg_dist"])
        pl_module.log(f"barlowtwins_dist_{phase}_txt_img/L2/Neg-Pos_txt_img", ret["text_image_neg_dist"]-ret["text_image_dist"])
        
        # Cosine Distance   
        pl_module.log(f"barlowtwins_dist_{phase}_txt_txt/cosine/Pos_txt_txt", ret["text_text_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_txt_txt/cosine/Neg_txt_txt", ret["text_text_neg_cosine"])        
        pl_module.log(f"barlowtwins_dist_{phase}_txt_txt/cosine/Neg-Pos_txt_txt", ret["text_text_neg_cosine"]-ret["text_text_cosine"])          
        pl_module.log(f"barlowtwins_dist_{phase}_txt_img/cosine/Pos_txt_img", ret["text_image_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_txt_img/cosine/Neg_txt_img", ret["text_image_neg_cosine"])
        pl_module.log(f"barlowtwins_dist_{phase}_txt_img/cosine/Neg-Pos_txt_img", ret["text_image_neg_cosine"]-ret["text_image_cosine"])        
        
        # Loss_invariance and Loss redundancy
        loss_invariance_text = getattr(pl_module, f"{phase}_barlowtwins_loss_invariance_text")(ret["barlowtwins_loss_invariance_text"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_invariance_text", loss_invariance_text)  
        loss_redundancy_text = getattr(pl_module, f"{phase}_barlowtwins_loss_redundancy_text")(ret["barlowtwins_loss_redundancy_text"])
        pl_module.log(f"barlowtwins/{phase}/barlowtwins_loss_redundancy_text", loss_redundancy_text)  
            
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

# No more used
def compute_pgd_finetuning(pl_module,batch,loss_name) : 
    img_delta_dict     = {}
    img_init           = {} 
    for i in range(2) : 
        img_init['image_{}'.format(i)] = batch['image_{}'.format(i)][0]
        if i == 1 : 
            # Attack both image indepedently
            batch['image_0'][0] = img_init['image_0']
        img_delta = torch.zeros_like(img_init['image_{}'.format(i)])
        for astep in range(pl_module.adv_steps_img):                
            img_delta.requires_grad_(True)                
            batch['image_{}'.format(i)][0] = img_delta + img_init['image_{}'.format(i)]

            infer1 = pl_module.infer(
                batch,mask_text=False, mask_image=False, image_token_type_idx=1)
            infer2 = pl_module.infer(
                batch, mask_text=False, mask_image=False, image_token_type_idx=2)
  
            cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
            nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

            nlvr2_labels = torch.tensor(batch["answers"]).to(pl_module.device).long()
            loss = F.cross_entropy(nlvr2_logits, nlvr2_labels, reduction='none')
            loss = loss.mean() 
            loss.backward(retain_graph=True)
            img_delta_grad = img_delta.grad.clone().detach().float()
            # Get inf_norm gradient (It will be used to normalize the img_delta_grad)
            denorm = torch.norm(img_delta_grad.view(img_delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1,1)

            # Clip gradient to Lower Bound
            denorm = torch.clamp(denorm, min=1e-8)
            img_delta_step = (pl_module.adv_lr_img * img_delta_grad / denorm).to(img_delta)
            img_delta = (img_delta + img_delta_step).detach()
            if pl_module.adv_max_norm_img > 0:
                img_delta = torch.clamp(img_delta, -pl_module.adv_max_norm_img, 
                                        pl_module.adv_max_norm_img).detach()           
        if i == 0 : 
            img_delta_0 = img_delta
        # Get imgdelta on CPU for analysis purposes 
        img_delta_cpu = img_delta.cpu()
        img_delta_dict['image_{}'.format(i)] = img_delta_cpu
    
    batch['image_0'][0] = img_init["image_0"] + img_delta_0        
    batch['image_1'][0] = img_init["image_1"] + img_delta

    phase = "train" if pl_module.training else "test"
    delta_range = getattr(pl_module, f"{phase}_{loss_name}_delta")(torch.linalg.norm(img_delta, dim=1).mean())
    pl_module.log(f"{loss_name}/{phase}/delta", delta_range)
    return batch,img_delta_dict    

# No more used
def compute_geometric_finetuning(pl_module, batch,loss_name) : 
    
    real_sentence = batch["text"]
    attack_words = \
    pl_module.greedy_attacker.adv_attack_samples(pl_module,batch) 
    
    #if attack_words["Problem"] == True : 
    #    print("This is the Real versus attacked sentences : ")

    #    for i in range(len(batch["text"])):
    #        print("Real sentence----: ",real_sentence[i])
    #        print("Attacked sentence: ",attack_words["text"][i])
    
    txt_original_attacked   = {"original": real_sentence,
                               "attacked": attack_words["text"]
                                }
    
    batch["text"]          = attack_words["text"]
    batch["text_ids"]      = attack_words["txt_input_ids"]
    batch["text_masks"]    = attack_words["text_masks"]

    phase = "train" if pl_module.training else "test"
    num_changes = getattr(pl_module, f"{phase}_{loss_name}_num_changes")(attack_words["num_changes"])
    change_rate = getattr(pl_module, f"{phase}_{loss_name}_change_rate")(attack_words["change_rate"])
    #pl_module.log(f"{loss_name}/{phase}/num_changes", num_changes)
    #pl_module.log(f"{loss_name}/{phase}/change_rate", change_rate)    
    
    return batch,txt_original_attacked

saving = 99
def compute_nlvr2_attack(pl_module, batch):
    global saving
    saving +=1
    ret = {}
    
    infer1 = pl_module.infer(batch,mask_text=False, mask_image=False, image_token_type_idx=1)
    infer2 = pl_module.infer(batch,mask_text=False, mask_image=False, image_token_type_idx=2)
    # NlVR2 output clean
    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    ori_logits = pl_module.nlvr2_classifier(cls_feats)
    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    ori_loss = F.cross_entropy(ori_logits, nlvr2_labels)
    ret["nlvr2_original_logits"] = ori_logits
    ret["nlvr2_original_loss"] = ori_loss
    ret["nlvr2_attacked_labels"] = nlvr2_labels

    attack_batch = None    
    if pl_module.image_view : 
        
        #batch_attacked,img_delta_dict = \
        #    compute_pgd_finetuning(pl_module,deepcopy(batch),"nlvr2_attacked")    
        attacked_batch = compute_pgd(pl_module, deepcopy(batch), "nlvr2_attacked")
        
        infer1_a = pl_module.infer(
            attacked_batch,mask_text=False, mask_image=False,image_token_type_idx=1
        )
        infer2_a = pl_module.infer(
            attacked_batch,mask_text=False, mask_image=False,image_token_type_idx=2
        )
        cls_feats = torch.cat([infer1_a["cls_feats"], infer2_a["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)
        

        ret["nlvr2_attacked_logits"] = nlvr2_logits
        ret["nlvr2_attacked_loss"] = nlvr2_loss
        
        ## Save img_delta
        #save_path ="/itet-stor/sfurrer/net_scratch/UNITER/ViLT/attacks_analysis/PGD"
        #if saving %100 == 0 : 
        #    with open(os.path.join(save_path,'img_delta_dict{}_max_norm_{}_lr_{}.pkl'\
        #        .format(saving,pl_module.adv_max_norm_img,pl_module.adv_lr_img)), 'wb') as fp:
        #                pickle.dump(img_delta_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)      
    if pl_module.text_view : 
        now_batch = attack_batch if attack_batch is not None else batch
        attacked_batch = compute_geometric(pl_module, deepcopy(now_batch), "nlvr2_attacked")
        #batch_attacked,txt_original_attacked = \
        #compute_geometric_finetuning(pl_module, deepcopy(batch),"nlvr2_attacked")
        
        infer1_a = pl_module.infer(
            attacked_batch,mask_text=False, mask_image=False,image_token_type_idx=1
        )
        infer2_a = pl_module.infer(
            attacked_batch,mask_text=False, mask_image=False,image_token_type_idx=2
        )
        cls_feats = torch.cat([infer1_a["cls_feats"], infer2_a["cls_feats"]], dim=-1)
        nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)
        nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

        ret["nlvr2_attacked_logits"] = nlvr2_logits
        ret["nlvr2_attacked_loss"] = nlvr2_loss
        
        ## Save txt_original_attacked
        #save_path ="/itet-stor/sfurrer/net_scratch/UNITER/ViLT/attacks_analysis/Geom"
        #if saving %100 == 0 : 
        #    with open(os.path.join(save_path,'txt_original_attacked{}_candidate_{}_loop_{}.pkl'\
        #        .format(saving,pl_module.n_candidates,pl_module.max_loops)), 'wb') as fp:
        #                pickle.dump(txt_original_attacked, fp, protocol=pickle.HIGHEST_PROTOCOL)     

    phase = "train" if pl_module.training else "val"
    if phase == "train":
        # there may be some bugs, but so far I just want to evaluation
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
                F.cross_entropy(ret["nlvr2_original_logits"][dev_batches], nlvr2_labels[dev_batches])
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_original_accuracy")(
                ret["nlvr2_original_logits"][dev_batches], nlvr2_labels[dev_batches]
            )
            pl_module.log(f"nlvr2_original/dev/loss", dev_loss)
            pl_module.log(f"nlvr2_original/dev/accuracy", dev_acc)
            if pl_module.image_view or pl_module.text_view:
                dev_loss = getattr(pl_module, f"dev_nlvr2_attacked_loss")(
                    F.cross_entropy(ret["nlvr2_attacked_logits"][dev_batches], nlvr2_labels[dev_batches])
                )
                dev_acc = getattr(pl_module, f"dev_nlvr2_attacked_accuracy")(
                    ret["nlvr2_attacked_logits"][dev_batches], nlvr2_labels[dev_batches]
                )
                pl_module.log(f"nlvr2_attacked/dev/loss", dev_loss)
                pl_module.log(f"nlvr2_attacked/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_original_loss")(
                F.cross_entropy(ret["nlvr2_original_logits"][test_batches], nlvr2_labels[test_batches])
            )
            test_acc = getattr(pl_module, f"test_nlvr2_original_accuracy")(
                ret["nlvr2_original_logits"][test_batches], nlvr2_labels[test_batches]
            )
            pl_module.log(f"nlvr2_original/test/loss", test_loss)
            pl_module.log(f"nlvr2_original/test/accuracy", test_acc)
            if pl_module.image_view or pl_module.text_view:
                test_loss = getattr(pl_module, f"test_nlvr2_attacked_loss")(
                    F.cross_entropy(ret["nlvr2_attacked_logits"][test_batches], nlvr2_labels[test_batches])
                )
                test_acc = getattr(pl_module, f"test_nlvr2_attacked_accuracy")(
                    ret["nlvr2_attacked_logits"][test_batches], nlvr2_labels[test_batches]
                )
                pl_module.log(f"nlvr2_attacked/test/loss", test_loss)
                pl_module.log(f"nlvr2_attacked/test/accuracy", test_acc)
                
    return ret
                                      
"""
                    
    phase = "train" if pl_module.training else "val"              
    if phase == "train":
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
            dev_loss = getattr(pl_module, f"dev_nlvr2_attacked_loss")(
                F.cross_entropy(
                    ret["nlvr2_attacked_logits"][dev_batches], ret["nlvr2_attacked_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_attacked_accuracy")(
                ret["nlvr2_attacked_logits"][dev_batches], ret["nlvr2_attacked_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2_attacked/dev/loss", dev_loss)
            pl_module.log(f"nlvr2_attacked/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_attacked_loss")(
                F.cross_entropy(
                    ret["nlvr2_attacked_logits"][test_batches], ret["nlvr2_attacked_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_attacked_accuracy")(
                ret["nlvr2_attacked_logits"][test_batches], ret["nlvr2_attacked_labels"][test_batches]
            )
            change_rate_cross = getattr(pl_module, f"test_nlvr2_attacked_change_rate_cross")(
                ret["nlvr2_attacked_logits"][test_batches], ret["nlvr2_logits"][test_batches], 
                ret["nlvr2_attacked_labels"][test_batches]
            )
          
            pl_module.log(f"nlvr2_attacked/test/loss", test_loss)
            pl_module.log(f"nlvr2_attacked/test/accuracy", test_acc)
            pl_module.log(f"nlvr2_attacked/test/change_rate_cross", change_rate_cross)

    return ret
"""
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

    # print(batch["text_ids"])
    # print(batch["false_text_0_ids"])
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
    ret["irtr_original_loss"] = irtr_loss
    ret["irtr_original_logits"] = score
    
    if pl_module.image_attack:
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
        score = pl_module.rank_output(infer["cls_feats"])[:, 0]
        score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
        irtr_loss = F.cross_entropy(score, answer)
        ret["irtr_attacked_loss"] = irtr_loss
        ret["irtr_attacked_logits"] = score

    if pl_module.text_attack:
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
        score = pl_module.rank_output(infer["cls_feats"])[:, 0]
        score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
        irtr_loss = F.cross_entropy(score, answer)
        ret["irtr_attacked_loss"] = irtr_loss
        ret["irtr_attacked_logits"] = score

    phase = "train" if pl_module.training else "val"
    
    irtr_loss = getattr(pl_module, f"{phase}_irtr_original_loss")(ret["irtr_original_loss"])
    pl_module.log(f"irtr_attacked/{phase}/irtr_original_loss", irtr_loss)
    acc = getattr(pl_module, f"{phase}_irtr_original_accuracy")(ret["irtr_original_logits"], answer)
    pl_module.log(f"irtr_attacked/{phase}/original_accuracy", acc)
    
    if pl_module.text_attack or pl_module.image_attack:
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
    if pl_module.text_attack or pl_module.image_attack:
        for _b in tqdm.tqdm(text_loader, desc="attacked data prefetch loop"):
            _b = {
                "text": _b["text"],
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
            }
            if pl_module.text_attack:
                _b = compute_geometric(pl_module, deepcopy(_b), "irtr")
                text_preload_attacked.append(
                    {
                        "text_ids": _b["text_ids"].to(pl_module.device),
                        "text_masks": _b["text_masks"].to(pl_module.device),
                        "text_labels": _b["text_labels"].to(pl_module.device),
                        "img_index": _b["img_index"],
                    }
                )
            if pl_module.image_attack:
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
                infer = pl_module.infer(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    },
                    image_embeds=ie,
                    image_masks=im,
                )
                img_representation, txt_representation = pl_module.moco_head(infer['image_feats'], infer['text_feats'])
                img_q = nn.functional.normalize(img_representation, dim=1)
                txt_q = nn.functional.normalize(txt_representation, dim=1)
                score = torch.einsum('nc,nc->n', [img_q, txt_q])  # .unsqueeze(-1)
                # print(txt_batch["img_index"], score)

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


def vqa_test_wrapup(outs, model_name):
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
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
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