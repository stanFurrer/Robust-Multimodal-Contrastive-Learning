from attack.greedy_attack_vilt import GreedyAttack_moco, GreedyAttack_barlowtwins, GreedyAttack_nlvr2,GreedyAttack_irtr, GreedyAttack_vqa
from attack.pgd_attack_vilt import PGDAttack_moco, PGDAttack_bartlowtwins, PGDAttack_nlvr2, PGDAttack_irtr, PGDAttack_vqa
from augmentation.image_augmentation import ImageAugmentation
from augmentation.text_augmentation import TextAugmentation
import os 
import time
from copy import deepcopy
from collections import OrderedDict 
from transformers import BertTokenizer
from Geometric_attack.greedy_attack_vilt_cross_entropy import GreedyAttack_cross_entropy 


import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()
        self.config = config
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            # Load itm head from other model
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        if config["loss_names"]["moco"] > 0:
            self.tsne_vizualisation = config["TSNE_vizualisation"]
            self.img_save_path = config["img_save_path"]
            self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
            self.multimodal = config["Multimodal"]
            self.per_step_bs = config["num_gpus"] * config["num_nodes"] * config["per_gpu_batchsize"]
            self.k_text_embeddings = BertEmbeddings(bert_config)
            self._shadow_layer(self.text_embeddings, self.k_text_embeddings)
            self.k_token_type_embeddings = nn.Embedding(2, config["hidden_size"])
            self._shadow_layer(self.token_type_embeddings, self.k_token_type_embeddings)
            self.k_transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
            self._shadow_layer(self.transformer, self.k_transformer)
            self.moco_head = heads.MOCOHead(config["hidden_size"], config["hidden_size"], 128)
            self.moco_head.apply(objectives.init_weights)
            self.k_moco_head = heads.MOCOHead(config["hidden_size"], config["hidden_size"], 128)
            self._shadow_layer(self.moco_head, self.k_moco_head)
            self.momentum = config["momentum"]
            self.temperature = config["temperature"]
            self.text_view = config["text_view"]
            self.augmentation= config["augmentation"]
            self.image_view = config["image_view"]
            self.num_negative = config["num_negative"]
            self.register_buffer("proj_queue", torch.randn(128, self.num_negative))
            self.register_buffer("proj_queue_ptr", torch.zeros(1, dtype=torch.long))
            #self.proj_queue = nn.functional.normalize(self.proj_queue, dim=0)
            
            if self.augmentation : 
                if self.text_view:
                    self.text_augmentation = TextAugmentation(config)
                if self.image_view:   
                    self.image_augmentation = ImageAugmentation(config)
            else : 
                if self.text_view:
                    print("----Loading greedy attack ----")
                    self.greedy_attacker = GreedyAttack_moco(config)
                    print("----Greedy attack Loaded ----")
                if self.image_view:
                    self.pgd_attacker = PGDAttack_moco(config)          
               
        if config["loss_names"]["barlowtwins"] > 0:
            self.tsne_vizualisation = config["TSNE_vizualisation"]
            self.img_save_path = config["img_save_path"]
            self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)    
            self.multimodal = config["Multimodal"]
            self.per_step_bs = config["num_gpus"] * config["num_nodes"] * config["per_gpu_batchsize"]
            self.barlowtwins_head = heads.BarlowTwinsHead(config["hidden_size"], [8192, 8192], 8192)
            self.text_view = config["text_view"]
            self.image_view = config["image_view"]
            self.adv_lr = config["adv_lr"]
            self.loss_weight = 0.001
            self.augmentation= config["augmentation"]
            if self.augmentation : 
                if self.text_view:
                    self.text_augmentation = TextAugmentation(config)
                if self.image_view:   
                    self.image_augmentation = ImageAugmentation(config)                    
            else :             
                if self.text_view:
                    print("----Loading greedy attack ----")
                    self.greedy_attacker = GreedyAttack_barlowtwins(config)
                    print("----Greedy attack Loaded ----")
                if self.image_view:
                    self.pgd_attacker = PGDAttack_bartlowtwins(config)
        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            # Load MLM or ITM head
            if config["loss_names"]["mlm"] > 0:
                if os.path.isfile("models_weight/vilt_200k_mlm_itm.ckpt") :
                    ckpt2 = torch.load("models_weight/vilt_200k_mlm_itm.ckpt", map_location="cpu")
                else :
                    ckpt2 = torch.load("../models_weight/vilt_200k_mlm_itm.ckpt", map_location="cpu")                 
                ckpt["state_dict"]['mlm_score.bias'] = ckpt2["state_dict"]['mlm_score.bias'] 
                ckpt["state_dict"]['mlm_score.transform.dense.weight'] = ckpt2["state_dict"]['mlm_score.transform.dense.weight']
                ckpt["state_dict"]['mlm_score.transform.dense.bias'] = ckpt2["state_dict"]['mlm_score.transform.dense.bias']
                ckpt["state_dict"]['mlm_score.transform.LayerNorm.weight'] = ckpt2["state_dict"]['mlm_score.transform.LayerNorm.weight']
                ckpt["state_dict"]['mlm_score.transform.LayerNorm.bias'] = ckpt2["state_dict"]['mlm_score.transform.LayerNorm.bias']
                ckpt["state_dict"]['mlm_score.decoder.weight'] = ckpt2["state_dict"]['mlm_score.decoder.weight']
            if config["loss_names"]["itm"] > 0: 
                if os.path.isfile("models_weight/vilt_200k_mlm_itm.ckpt") :
                    ckpt2 = torch.load("models_weight/vilt_200k_mlm_itm.ckpt", map_location="cpu")
                else :
                    ckpt2 = torch.load("../models_weight/vilt_200k_mlm_itm.ckpt", map_location="cpu")                 
                ckpt["state_dict"]['itm_score.fc.weight'] = ckpt2["state_dict"]['itm_score.fc.weight']
                ckpt["state_dict"]['itm_score.fc.bias']   = ckpt2["state_dict"]['itm_score.fc.bias']
                
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["vqa_attacked"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)            
            #param attacks
            self.image_view = config["image_view"]
            self.text_view  = config["text_view"]
            if config["text_view"]:
                print("----Loading GreedyAttack_cross_entropy ----")
                self.greedy_attacker = GreedyAttack_vqa(config)
                print("----Greedy GreedyAttack_cross_entropy DONE ----")
            if config["image_view"]:
                self.pgd_attacker = PGDAttack_vqa(config)              
            
        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]
            
        if self.hparams.config["loss_names"]["nlvr2_attacked"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]
            #param attacks
            self.image_view = config["image_view"]
            self.text_view  = config["text_view"]
            if config["text_view"]:
                print("----Loading GreedyAttack_cross_entropy ----")
                self.greedy_attacker = GreedyAttack_nlvr2(config)
                print("----Greedy GreedyAttack_cross_entropy DONE ----")
            if config["image_view"]:
                self.attack_idx = config["attack_idx"]
                self.pgd_attacker = PGDAttack_nlvr2(config)            
               
        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        if self.hparams.config["loss_names"]["irtr_attacked"] > 0:
            self.moco_head = heads.MOCOHead(config["hidden_size"], config["hidden_size"], 128)
            self.image_view = config["image_view"]
            self.text_view = config["text_view"]
            if config["text_view"]:
                print("----Loading GreedyAttack_cross_entropy ----")
                self.greedy_attacker = GreedyAttack_irtr(config)
                print("----Greedy GreedyAttack_cross_entropy DONE ----")
            if config["image_view"]:
                self.pgd_attacker = PGDAttack_irtr(config)        
       
        vilt_utils.set_metrics(self)
        self.current_tasks = list()        
        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            #Load the ITM head
            if config["loss_names"]["itm"] > 0: 
                if os.path.isfile("models_weight/vilt_200k_mlm_itm.ckpt") :
                    ckpt2 = torch.load("models_weight/vilt_200k_mlm_itm.ckpt", map_location="cpu")
                else :
                    ckpt2 = torch.load("../models_weight/vilt_200k_mlm_itm.ckpt", map_location="cpu")                    
                ckpt["state_dict"]['itm_score.fc.weight'] = ckpt2["state_dict"]['itm_score.fc.weight']
                ckpt["state_dict"]['itm_score.fc.bias']   = ckpt2["state_dict"]['itm_score.fc.bias']
       
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
                     
    def _shadow_layer(self, q_layer, k_layer):
        for param_q, param_k in zip(q_layer.parameters(), k_layer.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
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
            img = batch[imgkey][0] # [0] : Because it's a list of one element
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
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
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

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
    
    def infer_k(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]

        text_masks = batch[f"text_masks"]
        text_embeds = self.k_text_embeddings(text_ids)

        img = batch[imgkey][0]
        (
            image_embeds,
            image_masks,
            patch_index,
            image_labels,
        ) = self.k_transformer.visual_embed(
            img,
            max_image_len=self.hparams.config["max_image_len"],
            mask_it=mask_image,
        )

        text_embeds, image_embeds = (
            text_embeds + self.k_token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.k_token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds


        for i, blk in enumerate(self.k_transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.k_transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret
    
    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Visual Question Answering
        if "vqa_attacked" in self.current_tasks:
            ret.update(objectives.compute_vqa_attack(self, batch))            
            
        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))
            
        # Natural Language for Visual Reasoning 2
        if "nlvr2_attacked" in self.current_tasks:
            ret.update(objectives.compute_nlvr2_attack(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))
        
        # Image Retrieval and Text Retrieval
        if "irtr_attacked" in self.current_tasks:
            ret.update(objectives.compute_irtr_attacked(self, batch))
            
        # MoCo Contrasive framework
        if "moco" in self.current_tasks:
            ret.update(objectives.compute_moco_contrastive(self, batch))
        
        if "barlowtwins" in self.current_tasks:
            ret.update(objectives.compute_barlowtwins_contrastive(self, batch))
            
        return ret


    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0 or self.hparams.config["loss_names"]["vqa_attacked"] > 0 :
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0 or self.hparams.config["loss_names"]["vqa_attacked"] > 0:
            objectives.vqa_test_wrapup(outs, model_name,self.config)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)    