import torch
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import wandb
from enc_dec import ResNet18Enc, ResNet18Dec, SimEnc, SimDec
from matplotlib.pyplot import imshow, figure

class GANplotter(pl.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = 128


    def on_train_epoch_end(self, trainer, pl_module):
        """
        logs 6 GAN generated images at the end of every epoch using wandb
        """        
        figure(figsize=(8, 3), dpi=300)

        # Z COMES FROM NORMAL(0, 1)
        self.val_z = torch.randn((6, pl_module.hparams.latent_dim), device=pl_module.device)

        with torch.no_grad():
            sample_imgs = pl_module.generator(self.val_z).cpu()

        delta = 0.0
        fig, axes = plt.subplots(1,6, figsize=(12,2), constrained_layout=True)
        ax = axes.ravel()

        for i in range(6):
            im = ax[i].imshow(sample_imgs.detach()[i,0,:,:],
                                cmap='gray',
                                vmin= -1 + delta,
                                vmax=1 - delta)
            ax[i].axis('off')
            ax[i].set_title('Generated')   

        wandb.log({"Generated SHARP": wandb.Image(fig)})