import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from vilt.modules.dist_utils import all_gather
from vilt.modules.objectives import compute_irtr_recall , compute_attacked_irtr_recall
from vilt.gadgets.my_metrics import Accuracy, VQAScore, Scalar,change_rate

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "vqa_attacked":
                setattr(pl_module, f"{split}_vqa_attacked_score", VQAScore())
                setattr(pl_module, f"{split}_vqa_attacked_loss", Scalar())                
                
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            
            elif k == "nlvr2_attacked":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                    if pl_module.image_view:
                        setattr(pl_module, f"train_{k}_delta", Scalar())
                    if pl_module.text_view:
                        setattr(pl_module, f"train_{k}_num_changes", Scalar())
                        setattr(pl_module, f"train_{k}_change_rate", Scalar())  
                else:  
                    setattr(pl_module, f"dev_nlvr2_original_accuracy", Accuracy())
                    setattr(pl_module, f"dev_nlvr2_original_loss", Scalar())
                    setattr(pl_module, f"dev_nlvr2_attacked_accuracy", Accuracy())
                    setattr(pl_module, f"dev_nlvr2_attacked_loss", Scalar())
                    setattr(pl_module, f"test_nlvr2_original_accuracy", Accuracy())
                    setattr(pl_module, f"test_nlvr2_original_loss", Scalar())
                    setattr(pl_module, f"test_nlvr2_attacked_accuracy", Accuracy())
                    setattr(pl_module, f"test_nlvr2_attacked_loss", Scalar())    
                    
                    setattr(pl_module, f"test_nlvr2_attacked_change_rate_cross", change_rate())    
                    setattr(pl_module, f"dev_nlvr2_attacked_change_rate_cross", change_rate())                       
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "irtr_attacked":
                setattr(pl_module, f"{split}_irtr_original_loss", Scalar())
                setattr(pl_module, f"{split}_irtr_attacked_loss", Scalar())
                setattr(pl_module, f"{split}_irtr_original_accuracy", Accuracy())
                setattr(pl_module, f"{split}_irtr_attacked_accuracy", Accuracy())            
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
            
            elif k == "moco" or k == "barlowtwins":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                if k == "barlowtwins":
                    if pl_module.image_view:
                        setattr(pl_module, f"{split}_{k}_loss_invariance_img", Scalar())
                        setattr(pl_module, f"{split}_{k}_loss_redundancy_img", Scalar())
                    if pl_module.text_view:
                        setattr(pl_module, f"{split}_{k}_loss_invariance_text", Scalar())
                        setattr(pl_module, f"{split}_{k}_loss_redundancy_text", Scalar())          
                    if pl_module.text_view and pl_module.image_view and not pl_module.augmentation:
                        setattr(pl_module, f"{split}_{k}_loss_invariance_both", Scalar())
                        setattr(pl_module, f"{split}_{k}_loss_redundancy_both", Scalar())                           
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        if pl_module.hparams.config["loss_names"]["irtr_attacked"] > 0:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_attacked_irtr_recall(pl_module)
        else:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "vqa_attacked" : 
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()        
        
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        
        elif loss_name == "nlvr2_attacked":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
                
                if pl_module.image_view:
                    value = getattr(pl_module, f"train_{loss_name}_delta").compute()
                    pl_module.log(f"{loss_name}/train/img_delta_epoch", value)
                    getattr(pl_module, f"train_{loss_name}_delta").reset()
                if pl_module.text_view:
                    value = getattr(pl_module, f"train_{loss_name}_num_changes").compute()
                    pl_module.log(f"{loss_name}/train/txt_num_changes", value)
                    getattr(pl_module, f"train_{loss_name}_num_changes").reset()
                    value = getattr(pl_module, f"train_{loss_name}_change_rate").compute()
                    pl_module.log(f"{loss_name}/train/txt_change_rate", value)
                    getattr(pl_module, f"train_{loss_name}_change_rate").reset()                
            else:
                value = getattr(pl_module, f"dev_nlvr2_original_accuracy").compute()
                pl_module.log(f"nlvr2_original/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_nlvr2_original_accuracy").reset()
                pl_module.log(
                    f"nlvr2_original/dev/loss_epoch",
                    getattr(pl_module, f"dev_nlvr2_original_loss").compute(),
                )
                getattr(pl_module, f"dev_nlvr2_original_loss").reset()
                if pl_module.image_view or pl_module.text_view:
                    value = getattr(pl_module, f"dev_nlvr2_attacked_accuracy").compute()
                    pl_module.log(f"nlvr2_attacked/dev/accuracy_epoch", value)
                    getattr(pl_module, f"dev_nlvr2_attacked_accuracy").reset()
                    pl_module.log(
                        f"nlvr2_attacked/dev/loss_epoch",
                        getattr(pl_module, f"dev_nlvr2_attacked_loss").compute(),
                    )
                    getattr(pl_module, f"dev_nlvr2_attacked_loss").reset()
                    
                    value = getattr(pl_module, f"dev_nlvr2_attacked_change_rate_cross").compute()
                    pl_module.log(f"nlvr2_attacked/dev/change_rate_cross_epoch", value)
                    getattr(pl_module, f"dev_nlvr2_attacked_change_rate_cross").reset()                 
                    
                value = getattr(pl_module, f"test_nlvr2_original_accuracy").compute()
                pl_module.log(f"nlvr2_original/test/accuracy_epoch", value)
                getattr(pl_module, f"test_nlvr2_original_accuracy").reset()
                pl_module.log(
                    f"nlvr2_original/test/loss_epoch",
                    getattr(pl_module, f"test_nlvr2_original_loss").compute(),
                )
                getattr(pl_module, f"test_nlvr2_original_loss").reset()
                if pl_module.image_view or pl_module.text_view:
                    value = getattr(pl_module, f"test_nlvr2_attacked_accuracy").compute()
                    pl_module.log(f"nlvr2_attacked/test/accuracy_epoch", value)
                    getattr(pl_module, f"test_nlvr2_attacked_accuracy").reset()
                    pl_module.log(
                        f"nlvr2_attacked/test/loss_epoch",
                        getattr(pl_module, f"test_nlvr2_attacked_loss").compute(),
                    )
                    getattr(pl_module, f"test_nlvr2_attacked_loss").reset()    
                    
                    value = getattr(pl_module, f"test_nlvr2_attacked_change_rate_cross").compute()
                    pl_module.log(f"nlvr2_attacked/test/change_rate_cross_epoch", value)
                    getattr(pl_module, f"test_nlvr2_attacked_change_rate_cross").reset()                 
 
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "irtr_attacked":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_original_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_original_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_original_loss").reset()
            
            value = getattr(pl_module, f"{phase}_irtr_original_accuracy").compute()
            pl_module.log(f"irtr_attacked/{phase}/original_accuracy_epoch", value)
            getattr(pl_module, f"{phase}_irtr_original_accuracy").reset()
            
            if pl_module.text_attack or pl_module.image_attack:
                pl_module.log(
                    f"{loss_name}/{phase}/irtr_attacked_loss_epoch",
                    getattr(pl_module, f"{phase}_irtr_attacked_loss").compute(),
                )
                getattr(pl_module, f"{phase}_irtr_attacked_loss").reset()
            
                value = getattr(pl_module, f"{phase}_irtr_attacked_accuracy").compute()
                pl_module.log(f"irtr_attacked/{phase}/attacked_accuracy_epoch", value)
                getattr(pl_module, f"{phase}_irtr_attacked_accuracy").reset()        
        
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        
        elif loss_name == "moco" or loss_name == "barlowtwins":
            value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}_loss/epoch/{phase}", value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            if loss_name == "barlowtwins" : 
                if pl_module.image_view : 
                    value_invariance_img = getattr(pl_module, f"{phase}_{loss_name}_loss_invariance_img").compute()
                    pl_module.log(f"{loss_name}_loss_invariance_img/epoch/{phase}", value_invariance_img)
                    getattr(pl_module, f"{phase}_{loss_name}_loss_invariance_img").reset()     
                    
                    value_redundancy_img = getattr(pl_module, f"{phase}_{loss_name}_loss_redundancy_img").compute()
                    pl_module.log(f"{loss_name}_loss_redundancy_img/epoch/{phase}", value_redundancy_img)
                    getattr(pl_module, f"{phase}_{loss_name}_loss_redundancy_img").reset()     
                    
                if pl_module.text_view : 
                    value_invariance_text = getattr(pl_module, f"{phase}_{loss_name}_loss_invariance_text").compute()
                    pl_module.log(f"{loss_name}_loss_invariance_text/epoch/{phase}", value_invariance_text)
                    getattr(pl_module, f"{phase}_{loss_name}_loss_invariance_text").reset()     
                    
                    value_redundancy_text = getattr(pl_module, f"{phase}_{loss_name}_loss_redundancy_text").compute()
                    pl_module.log(f"{loss_name}_loss_redundancy_text/epoch/{phase}", value_redundancy_text)
                    getattr(pl_module, f"{phase}_{loss_name}_loss_redundancy_text").reset()             
            
                if pl_module.image_view  and pl_module.text_view : 
                    value_invariance_both = getattr(pl_module, f"{phase}_{loss_name}_loss_invariance_both").compute()
                    pl_module.log(f"{loss_name}_loss_invariance_both/epoch/{phase}", value_invariance_both)
                    getattr(pl_module, f"{phase}_{loss_name}_loss_invariance_both").reset()     
                    
                    value_redundancy_both = getattr(pl_module, f"{phase}_{loss_name}_loss_redundancy_both").compute()
                    pl_module.log(f"{loss_name}_loss_redundancy_both/epoch/{phase}", value_redundancy_text)
                    getattr(pl_module, f"{phase}_{loss_name}_loss_redundancy_both").reset()                 

        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)



def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return

def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier", "moco_head", "barlowtwinshead"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )
