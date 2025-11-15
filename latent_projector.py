import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from gan import GAN
import numpy as np
import pandas as pd
import cv2
import h5py
import glob
import pickle

def image_to_GAN_latent(filename, gan_idx=81, iter=1000, seed=1):
    """_summary_

    Args:
        filename (str): corresponds to image to be project to GAN latent space
        gan_idx (int, optional): Index of trained GAN models. Defaults to 81.
        iter (int, optional): Number of iterations for choosing best reconstruction. Defaults to 1000.
        seed (int, optional): To initialize GAN latent vector. Defaults to 1.

    Returns:
        reconstruction loss, reconstructed image, latent vector, target image
    """    
    ### load image
    file = h5py.File(filename)
    key = list(file.keys())[0]
    data = np.array(file[key])
    data = np.clip(data, -1000, 1000)/1000
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=1, neginf=-1)
 
 
    ### load GAN
    ep = str(gan_idx)
    name = f'GAN_sharp_experiment_ld_100_bs_32_lr_0.0005_stride_1_crop_False_arch_simple_epoch={ep}.ckpt'
    MODEL_DIR = '/d0/subhamoy/models/gan/sharps/'
    model = GAN.load_from_checkpoint(MODEL_DIR + name)
    model.to('cuda:1')

    ### optimize GAN latent such that generated image matches real image
    for param in model.parameters():
        param.requires_grad = False
        
    loss_fn = torch.nn.MSELoss() #L1Loss() #

    m = torch.nn.Linear(1,100)
    bias = torch.nn.Parameter(torch.zeros(100), requires_grad=False)
    m.bias = bias
    
    torch.manual_seed(seed)
    m.weight = torch.nn.Parameter(torch.randn(100,1, dtype=torch.float32))
    
    m.to('cuda:1')

            
    optimizer = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    #optimizer = torch.optim.Adam(m.parameters(), lr=0.01)#, momentum=0.9)
    init_point = torch.ones((1,1), requires_grad=True, dtype=torch.float32, device='cuda:1')
    
    img = cv2.resize(data[0, :, :], (128, 128),interpolation = cv2.INTER_AREA)
    
    mask = 0.5+ 0.5*((img>0.05).astype(float) + (img<-0.05).astype(float))
    target = torch.tensor(img[None,None,:,:], requires_grad=False, device = 'cuda:1', dtype=torch.float32)
    mask = torch.tensor(mask[None,None,:,:], requires_grad=False, device = 'cuda:1', dtype=torch.float32)

  
    l, imgs, weights = [], [], []
    for _ in range(iter):
        predicted = model(m(init_point))
        loss = loss_fn(predicted*mask, target*mask) 
        loss.to('cuda:1')
        loss.backward()
        optimizer.step()
        img = predicted.detach()
        l.append(loss.detach().cpu().numpy())
        weights.append(m.weight.detach().cpu().numpy())
        imgs.append(img.cpu().numpy()[0,0,:,:])
        
    idx = np.argmin(np.array(l))
    
    return l[idx], imgs[idx], weights[idx], target[0,0,:,:].cpu()

def save_GAN_latent():
    DIR = '/d0/kvandersande/sharps_hdf5/'
    filenames = sorted(glob.glob(DIR+'*.h5'))
    df = {'files':filenames, 'latent_GAN':[]}
    
    for j in range(len(filenames)):
        loss, weights  = [], []
        for i in range(10):
            l, _, weight, _ = image_to_GAN_latent(filenames[j], iter=1000, gan_idx=81, seed=i)
            loss.append(l)
            weights.append(weight)
        
        idx = np.argmin(np.array(loss))
        df['latent_GAN'].append(weights[idx])
        print(j, filenames[j])
        pickle.dump(df,open('/d0/subhamoy/models/gan/sharps/latent_space.p', 'wb'))

def main():
    save_GAN_latent()
    ep = '81'
    name = f'GAN_sharp_experiment_ld_100_bs_32_lr_0.0005_stride_1_crop_False_arch_simple_epoch={ep}.ckpt'
    MODEL_DIR = '/d0/subhamoy/models/gan/sharps/'
    model = GAN.load_from_checkpoint(MODEL_DIR + name)
    model.to('cuda:1')
    p = pickle.load(open('/d0/subhamoy/models/gan/sharps/latent_space.p', 'rb'))
    df = pd.read_csv('/d0/kvandersande/index_sharps.csv')
    files, lons, lats, latents, losses = [], [], [], [], []
    pp = {'files': files, 'lons': lons, 'lats': lats, 'latents':latents, 'losses': losses}
    loss_fn = torch.nn.MSELoss()
    for i,f in enumerate(p['files']):
        file = h5py.File(f)
        if f in list(df['file']):
            pp['files'].append(f)
            lmax = df['lon_max'][df['file']==f].values[0]
            lmin = df['lon_min'][df['file']==f].values[0]
            ltmax = df['lat_max'][df['file']==f].values[0]
            ltmin = df['lat_min'][df['file']==f].values[0]
            pp['lons'].append((lmin+lmax)/2)
            pp['lats'].append((ltmin+ltmax)/2)
            pp['latents'].append(p['latent_GAN'][i].reshape(1,-1))
            key = list(file.keys())[0]
            data = np.array(file[key])
            data = np.clip(data, -1000, 1000)/1000
            data = np.nan_to_num(data, copy=False, nan=0.0, posinf=1, neginf=-1)
            predicted = model(torch.tensor(p['latent_GAN'][i].reshape(1,-1)).to('cuda:1'))
            img = cv2.resize(data[0, :, :], (128, 128),interpolation = cv2.INTER_AREA)
            mask = 0.5+ 0.5*((img>0.05).astype(float) + (img<-0.05).astype(float))
            target = torch.tensor(img[None,None,:,:], requires_grad=False, device = 'cuda:1', dtype=torch.float32)
            mask = torch.tensor(mask[None,None,:,:], requires_grad=False, device = 'cuda:1', dtype=torch.float32)
            loss = loss_fn(predicted*mask, target*mask) 
            pp['losses'].append(loss.detach().cpu().numpy())
            pickle.dump(pp,open('/d0/subhamoy/models/gan/sharps/longitude_vs_reconstr_loss.p', 'wb'))
            print(i, f, lmin, lmax, ltmin, ltmax, loss)
    
        
if __name__=='__main__':
    main()