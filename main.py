import os
import os.path as osp
import argparse

import torch
import torchaudio
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data import MelDM
from src.train_module import MelModul


def main(n_mels = 80, n_groups=1):
    EXPERIMENTS_PATH = "experiments"
    experiment_name = f'mels_{n_mels}_n_groups_{n_groups}'
    experiment_save_path = osp.join(EXPERIMENTS_PATH, experiment_name)

    torchaudio.datasets.SPEECHCOMMANDS(root="./dataset", download=True)

    datamodule = MelDM(n_mels=n_mels)
    model = MelModul(n_mels=n_mels, n_groups=n_groups)
    os.makedirs(experiment_save_path, exist_ok=True)
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    logger = TensorBoardLogger(
        save_dir=EXPERIMENTS_PATH,
        name=experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
        filename="{epoch:02d}-{valid_loss:.3f}",
    )

    trainer = L.Trainer(
        max_epochs=40,
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=1,
        gradient_clip_val=1.0,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)
    trainer.validate(
        ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_groups", type=int, default=1)
    args = parser.parse_args()
    main(n_mels=args.n_mels, n_groups=args.n_groups)

