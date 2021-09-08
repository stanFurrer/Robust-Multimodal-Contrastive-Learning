import pickle
import os
import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules.vilt_module import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule

# Solve issue : #5486 in huggingface/transformer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    print("\n------------------------------")
    print("NAME EXPERIENCE : ", _config["exp_name"])
    print("The Multimodal is set to ", _config["Multimodal"])
    if _config["augmentation"] : 
        print("Doing augmentation") 
        print("Text_view set to ", _config["text_view"])
        print("Image_view set to ",_config["image_view"] )
    else : 
        if _config["image_view"] : 
            print("Hyper parameter for pgd")
            print("adv_lr_img :",_config["adv_lr_img"])
            print("adv_max_norm_img :",_config["adv_max_norm_img"])
        if _config["text_view"] :
            print("\nHyper parameter for Geometric")
            print("n_candidates :",_config["n_candidates"])
            print("max_loops :",_config["max_loops"])        
    print("------------------------------\n")    

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    test = False #-----------------------------------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if test == True : 
        # For test Purposes : Other
        logger = pl.loggers.TensorBoardLogger(
            _config["log_dir"],
            name= "Other_new"
            #name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
        )         
    else : 
        if _config["exp_save"] !="vilt" : 
            name = _config["exp_save"]
        else : 
            name = "Pretraining_moco"#"finetuning_irtr_flick_barlowtwins"
            #name = "MCL/Finetuning_IRTR_flick_moco"
            #name    = "IRTR/finetuning_irtr_barlowtwins" # NLVR2/Finetuning_nlvr2_moco
        version = "Version-3component" 
        # For experimence Purposes 
        logger = pl.loggers.TensorBoardLogger(
            _config["log_dir"],
            version=version,
            name= name
        )          

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    # callbacks = [checkpoint_callback, lr_callback]
    callbacks = [lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print("This is grad_steps",grad_steps)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        print("-----------This is for test only--------------- ")
        results = trainer.test(model, datamodule=dm)
