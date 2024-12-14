import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from dataset import SHARPdataset
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import wandb
from enc_dec import ResNet18Enc, ResNet18Dec, SimEnc, SimDec
from matplotlib.pyplot import imshow, figure

class Discriminator(nn.Module):
    def __init__(self, arch = 'simple'):
        """_summary_

        Args:
            arch (str, optional): _description_. Defaults to 'simple'.
        """        
        super().__init__()
        if arch=='simple':
            self.block1 = SimEnc(latent_dim=100, nc = 1)
        else:
            self.block1 = nn.Sequential(ResNet18Enc(nc = 1),
                                        nn.Linear(512, 100),
                                        nn.LeakyReLU(0.2))
        self.block2 = nn.Linear(100, 1)
  
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return torch.sigmoid(x)
    
    
# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim, arch = 'simple'):
        """_summary_

        Args:
            latent_dim (_type_): _description_
            arch (str, optional): _description_. Defaults to 'simple'.
        """        
        super().__init__()
        if arch=='simple':
            self.bl = SimDec(latent_dim=latent_dim, nc = 1)
        else:
            self.bl = ResNet18Dec(z_dim=latent_dim, nc = 1)

    def forward(self, x):
        return self.bl(x)
    
    
class GAN(pl.LightningModule):
    def __init__(self, arch='simple', latent_dim = 100, lr = 0.0005):
        """_summary_

        Args:
            arch (str, optional): _description_. Defaults to 'simple'.
            latent_dim (int, optional): _description_. Defaults to 100.
            lr (float, optional): _description_. Defaults to 0.0005.
        """        
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim=self.hparams.latent_dim, arch=arch)
        self.discriminator = Discriminator(arch=arch)
        
    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):     
        real_imgs, _ = batch
        optimizer_g, optimizer_d = self.optimizers()
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)
            
        self.toggle_optimizer(optimizer_g)
        fake_imgs = self(z)
        y_hat = self.discriminator(fake_imgs)
        
        y = torch.ones(real_imgs.size(0), 1)
        y = y.type_as(real_imgs)
        
        g_loss = self.adversarial_loss(y_hat, y)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
            
            
        self.toggle_optimizer(optimizer_d)
        y_hat_fake = self.discriminator(self(z).detach())
        y_fake = torch.zeros(real_imgs.size(0), 1)
        y_fake = y_fake.type_as(real_imgs)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
        fake_mask = y_hat_fake > 0.5
        fake_acc = (y_fake == fake_mask).sum().item() / y_fake.size(0)
        
        y_hat_real = self.discriminator(real_imgs)
        y_real = torch.ones(real_imgs.size(0), 1)
        y_real = y_real.type_as(real_imgs)
        real_loss = self.adversarial_loss(y_hat_real, y_real)
        real_mask = y_hat_real > 0.5
        real_acc = (y_real == real_mask).sum().item() / y_real.size(0)
        
        d_loss = (fake_loss + real_loss)/2
        
        log_dct = {"d_loss": d_loss,
                    "fake_acc":fake_acc,
                    "real_acc":real_acc}
        
        self.log_dict(log_dct, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d], []