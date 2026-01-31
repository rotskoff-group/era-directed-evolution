import os
import hydra
from pera.nn import (create_dataset_from_path,
                    create_lightning_model,
                    create_dataloaders)
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf
import lightning.pytorch.strategies



@hydra.main(version_base="1.3", config_path="../cfgs", config_name="train_transformer")
def main(cfg):
    nn_config = cfg.nn
    train_config = cfg.train
    global_args = cfg.global_args

    L.seed_everything(**train_config.seed_args)
    prot_model = create_lightning_model(nn_config=nn_config,
                                            train_config=train_config)
    
    dataset = create_dataset_from_path(cfg.global_args.dataset_filename,
                                       nn_config)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset,
                                                                           nn_config)

    resume_training_path = train_config["resume_training_path"]
    every_epoch_checkpoint_callback = ModelCheckpoint(
        **train_config["every_epoch_checkpoint_args"]
    )
    best_checkpoint_callback = ModelCheckpoint(
        **train_config["best_checkpoint_args"]
    )
    
    if train_config["strategy"] is not None:
        strategy = getattr(lightning.pytorch.strategies, train_config["strategy"])
        strategy = strategy(**train_config["strategy_args"])

    if train_config["logger"]["loggertype"] == "TensorBoard":
        logger = TensorBoardLogger(save_dir="./", version=train_config["logger"]["logger_args"]["version"], name="lightning_logs")

        trainer = L.Trainer(callbacks=[
                                    every_epoch_checkpoint_callback,
                                    best_checkpoint_callback],
                            strategy=strategy,
                            logger=logger,
                            **train_config["trainer_args"])
    else:
        trainer = L.Trainer(callbacks=[
                                    every_epoch_checkpoint_callback,
                                    best_checkpoint_callback],
                            strategy=strategy,
                            **train_config["trainer_args"])

    if trainer.global_rank == 0:
        train_folder_name = trainer.logger.log_dir
        os.makedirs(train_folder_name, exist_ok=True)
        OmegaConf.save(cfg, f"{train_folder_name}/config.yaml")

    if nn_config["load_model"] is not None:
        if os.path.isfile(nn_config["load_model"]):
            prot_model.load_model_from_ckpt(nn_config["load_model"])
        else:
            raise NotImplemented

    trainer.fit(prot_model, train_dataloader, val_dataloader, ckpt_path=resume_training_path if resume_training_path is not None else None)


if __name__ == "__main__":
    main()
