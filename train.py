from pytorch_lightning.callbacks import ModelCheckpoint

from traffic_signs_datamodule import TrafficSignDataModule
from model import Model
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn

GCN_IN = 3
GCN_OUT = 64
NTN_k = 10

torch.set_float32_matmul_precision('high')


def train():
    dm = TrafficSignDataModule(batch_size=1)
    model = Model(GCN_IN, GCN_OUT, NTN_k, is_softmax=False)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_acc",
        mode="max",
        dirpath=f"lightning_logs/Sub-GMN/TrafficSign/checkpoints",
        filename=f"{str(datetime.now()).replace(' ', '_').replace(':','_')}" + "-{epoch:02d}",
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=10_000,
        gradient_clip_val=1,
        callbacks=[checkpoint_callback],
        default_root_dir=f"lightning_logs/Sub-GMN/TrafficSign/checkpoints",
        overfit_batches=1,
        check_val_every_n_epoch=10_000,
       # precision='16-mixed',
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    train()
