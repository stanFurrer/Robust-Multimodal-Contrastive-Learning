import os
import copy
import pytorch_lightning as pl

from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    # Adding some parameter to _congig
    # Geometric inspired attack
    #_config["n_candidates"]   = 20
    #_config["max_loops"]      = 10
    #_config["sim_thred"]      = 0.5
    #_config["cos_sim"]        = True
    #_config["synonym"]        = 'cos_sim'
    #_config["embedding_path"] = './Geometric_attack/counter-fitted-vectors.txt'
    #_config["sim_path"]       = 'cos_sim_counter_fitting.npy'
    

    # Pgd attack
    #_config["adv_steps_img"]    = 5
    #_config["adv_lr_img"]       = 0.5
    #_config["adv_max_norm_img"] = 0.1

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
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
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    
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
        trainer.test(model, datamodule=dm)
