from pytorch_lightning.callbacks import ModelCheckpoint

from traffic_signs_datamodule import TrafficSignDataModule
from model import Model
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn

GCN_IN = 3
GCN_OUT = 128
NTN_k = 32
GCN_k = 7

torch.set_float32_matmul_precision('medium')


def train():
    size = (1000, 1000)
    dm = TrafficSignDataModule(batch_size=15, crop_size=size, data_graph_seg_size=300)
    model = Model(GCN_IN, GCN_OUT, NTN_k, GCN_k)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_acc",
        mode="max",
        dirpath=f"lightning_logs/Sub-GMN/TrafficSign/checkpoints",
        filename=f"{str(datetime.now()).replace(' ', '_').replace(':','_')}" + f"{GCN_OUT=}--{NTN_k=}--size={size[0]}" + "-{epoch:02d}--{val_AUC:.02f}",
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100_000,
        gradient_clip_val=1,
        callbacks=[checkpoint_callback],
        default_root_dir=f"lightning_logs/Sub-GMN/TrafficSign/checkpoints",
        #overfit_batches=1,
        check_val_every_n_epoch=250,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    train()
