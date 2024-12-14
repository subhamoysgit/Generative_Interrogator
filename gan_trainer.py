import torch
from dataset import SHARPdataset
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import wandb
from enc_dec import ResNet18Enc, ResNet18Dec, SimEnc, SimDec
from matplotlib.pyplot import imshow, figure
from gan import GAN
from gan_plotter import GANplotter


def main():
    pl.seed_everything(23)
    torch.set_float32_matmul_precision('high')

    image_size = 128
    bs = 32
    ep = 300
    ld = 100
    lr = 0.0005
    st = 1
    crop = False
    arch = 'simple'
    name = f'GAN_sharp_experiment_ld_{ld}_bs_{bs}_lr_{lr}_stride_{st}_crop_{crop}_arch_{arch}_'
    fname = name + "{epoch:02d}"
    DATA_DIR = '/d0/kvandersande/sharps_hdf5/'
    MODEL_DIR = '/d0/subhamoy/models/gan/sharps/'
    model = GAN(latent_dim=ld, lr=lr, arch=arch)
    wandb_logger = WandbLogger(entity="sc8473",
                                # Set the project where this run will be logged
                                project="GAN_sharp",
                                name = name,
                                # Track hyperparameters and run metadata
                                config={
                                    "learning_rate": lr,
                                    "epochs": ep,
                                    "batch_size": bs,
                                    "latent_dim": ld
                                })

    checkpoint_callback = ModelCheckpoint(dirpath=MODEL_DIR,
                                        filename=fname, #'{epoch}-{name}',
                                        save_top_k=-1,
                                        verbose=True,
                                        #monitor='g_loss',
                                        mode='min')

    plotter = GANplotter()

    trainer = pl.Trainer(max_epochs=ep,
                        accelerator='gpu',
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, plotter],
                        devices=[0],
                        log_every_n_steps=10)

    sharp_dataset = SHARPdataset(DATA_DIR, image_size=image_size,
                                    crop=crop, include_sharp=False,
                                    data_stride=st, use_first=True)
    dataLoader = torch.utils.data.DataLoader(sharp_dataset,
                                                batch_size=bs,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=8) 
    trainer.fit(model, dataLoader)
    
if __name__=='__main__':
    main()