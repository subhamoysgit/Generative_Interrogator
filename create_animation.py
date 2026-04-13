import torch
from dataset import SHARPdataset
import matplotlib.pyplot as plt
from gan import GAN
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
from PIL import Image
import torchvision.transforms.functional as functional
import cv2
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.offsetbox as osb
from matplotlib import rcParams as rcp
import torchvision.transforms.functional as functional
from PIL import Image
import torch
from simsiam import load_model
from sdo_augmentation.augmentation import Augmentations
from sdo_augmentation.augmentation_list import AugmentationList
import h5py
from PIL import Image
from utils import * 
from scipy import stats
from latent_projector import *
from sunpy.coordinates import sun
from datetime import datetime
from scipy.stats import pearsonr
import pandas as pd

ep = '81'
name = f'GAN_sharp_experiment_ld_100_bs_32_lr_0.0005_stride_1_crop_False_arch_simple_epoch={ep}.ckpt'
MODEL_DIR = '/d0/subhamoy/models/gan/sharps/'#'./models/'
pl_module = GAN.load_from_checkpoint(MODEL_DIR + name)
device = 'cuda:1'
pl_module.to(device)

# set random seed to 23 and generate 100 latent vector as input to GAN
torch.manual_seed(23)
val_z = torch.randn((100, pl_module.hparams.latent_dim), device=pl_module.device)

latent_space = torch.randn((10000, pl_module.hparams.latent_dim), device=pl_module.device)
gen_imgs = torch.zeros(10000, 1, 128, 128)
with torch.no_grad():
    gen_imgs = pl_module(latent_space).cpu()

gen_imgs = gen_imgs.numpy()
latent_space = latent_space.cpu().numpy()

clf_dict = pickle.load(open('dict_clf.p', 'rb'))
# keys: 'uflux','pflux','nflux','eflux','tilt','dist','area','polarity','r','mu'
clf = clf_dict['uflux']
normal1 = clf.coef_[0]/(np.sum(clf.coef_[0]**2)**0.5)
clf_r = clf_dict['r']
normal2 = clf_r.coef_[0]/(np.sum(clf_r.coef_[0]**2)**0.5)
clf_p = clf_dict['polarity']
normal3 = clf_p.coef_[0]/(np.sum(clf_p.coef_[0]**2)**0.5)
clf_d = clf_dict['dist']
normal4 = clf_d.coef_[0]/(np.sum(clf_d.coef_[0]**2)**0.5)

normal = project_boundary(normal2[None,:], normal1[None,:], normal3[None,:]) 
normal = normal/(np.sum(normal**2)**0.5)
normal = normal[0,:]
delta=0.01
idx=25#75
u = torch.from_numpy(normal).to(device)
u = u.type_as(val_z)
n = torch.from_numpy(normal1).to(device)
n = n.type_as(val_z)

for i in range(7):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    axs = ax.ravel()
    eps = i-6
    z_shift = val_z[idx, :] + eps*n

    latent_plane = hyperplane(latent_space, clf)
    embeddings_2d, mn, mx = project_latent_2d(latent_space, latent_plane, clf,type='custom')
    _, cmin, cmax, img_min, img_max = image_on_latent(gen_imgs, embeddings_2d[:10000,:], embeddings_2d[10000:,:],
                                        type='custom', title = None, mask = False, pol_connect=False, ax=axs[0], fig=fig)

    pt = z_shift.detach().cpu().numpy()

    dot = np.dot(clf.coef_[0][1:], pt[1:])
    pt_2d = pt[:2]
    pt_2d[1]  = dot/clf.coef_[0][1]
    pt_2d = (pt_2d - mn) / (mx - mn)
    plt.suptitle('Magnetic Field', y=0.92, fontsize=15)
    axs[0].plot([1-pt_2d[1], 1-pt_2d[1]], 
            [pt_2d[0], pt_2d[0]], '*b', ms=20, zorder=100)

    with torch.no_grad():
        shifted_img = pl_module(z_shift[None,:].to(device)).cpu()
    ax[1].imshow(shifted_img.detach()[0,0,:,:],
                    cmap='gray',
                    vmin= -1,
                    vmax=1)
    ax[1].axis('off')
    
    # --- Step 1: convert (x, y) in axs[0] → figure coordinates
    start_disp = axs[0].transData.transform((1-pt_2d[1], pt_2d[0]))          # data → display
    start_fig = fig.transFigure.inverted().transform(start_disp)  # display → figure

    # --- Step 2: define target point on left edge of axs[1]
    bbox = axs[1].get_position()   # in figure coords
    end_fig = (bbox.x0, (bbox.y0 + bbox.y1) / 2)  # middle of left edge

    # --- Step 3: draw arrow in figure coordinates
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)  # invisible full-figure axes
    ax.set_axis_off()

    ax.annotate(
    '',
    xy=end_fig, xycoords='figure fraction',
    xytext=start_fig, textcoords='figure fraction',
    arrowprops=dict(arrowstyle='->', color='k', linestyle = '--', lw=1))
    
    # --- compute span across both axes
    pos0 = axs[0].get_position()
    pos1 = axs[1].get_position()

    left = pos1.x0
    right = pos1.x1
    width = right - left

    # --- add a thin axis on top
    top_ax = fig.add_axes([left+0.04, pos0.y1 + 0.02, width-0.04, 0.08])

    # --- horizontal bar
    print(np.abs(shifted_img.detach()[0,0,:,:]).sum())
    s = shifted_img.detach()
    mfield = (np.abs(s[0,0,:,:]).sum() - 378.8535)/(2306.4561 - 378.8535)
    top_ax.barh([0], [mfield/2], height=0.6, color='blue')  # single bar

    # --- styling (important)
    top_ax.set_xlim(0, 1)        # adjust based on your scale
    top_ax.set_yticks([])        # hide y
    top_ax.set_xticks([])        # optional: hide x
    top_ax.spines[:].set_visible(False)
    
    st = str(i).zfill(2)
    
    plt.savefig(f'./plots/plot_{st}.png', dpi=300)
    if i==6:
        radi = torch.norm(z_shift)
    
    
    
for i in range(12):
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    axs = ax.ravel()
    eps = i/4

    z_shift = val_z[idx, :] + 3.5*eps*(u) + 0.0*n

    latent_plane = hyperplane(latent_space, clf)
    embeddings_2d, mn, mx = project_latent_2d(latent_space, latent_plane, clf,type='custom')
    _, cmin, cmax, img_min, img_max = image_on_latent(gen_imgs, embeddings_2d[:10000,:], embeddings_2d[10000:,:],
                                        type='custom', title = None, mask = False, pol_connect=False, ax=axs[0], fig=fig)

    
    
    # print(radi, torch.norm(z_shift))
    z_shift /= torch.norm(z_shift)
    z_shift *= radi
    radi *= 1.001
    pt = z_shift.detach().cpu().numpy()


    dot = np.dot(clf.coef_[0][1:], pt[1:])
    pt_2d = pt[:2]
    pt_2d[1]  = dot/clf.coef_[0][1]
    pt_2d = (pt_2d - mn) / (mx - mn)
    #plt.suptitle('Magnetic Field', y=0.92, fontsize=15)
    axs[0].plot([1-pt_2d[1], 1-pt_2d[1]], 
            [pt_2d[0], pt_2d[0]], '*r', ms=20, zorder=100)

    with torch.no_grad():
        shifted_img = pl_module(z_shift[None,:].to(device)).cpu()
    ax[1].imshow(shifted_img.detach()[0,0,:,:],
                    cmap='gray',
                    vmin= -1,
                    vmax=1)
    ax[1].axis('off')
    
    # --- Step 1: convert (x, y) in axs[0] → figure coordinates
    start_disp = axs[0].transData.transform((1-pt_2d[1], pt_2d[0]))          # data → display
    start_fig = fig.transFigure.inverted().transform(start_disp)  # display → figure

    # --- Step 2: define target point on left edge of axs[1]
    bbox = axs[1].get_position()   # in figure coords
    end_fig = (bbox.x0, (bbox.y0 + bbox.y1) / 2)  # middle of left edge

    # --- Step 3: draw arrow in figure coordinates
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)  # invisible full-figure axes
    ax.set_axis_off()

    ax.annotate(
    '',
    xy=end_fig, xycoords='figure fraction',
    xytext=start_fig, textcoords='figure fraction',
    arrowprops=dict(arrowstyle='->', color='k', linestyle = '--', lw=1))
    
    # --- compute span across both axes
    pos0 = axs[0].get_position()
    pos1 = axs[1].get_position()

    left = pos1.x0
    right = pos1.x1
    width = right - left

    # --- add a thin axis on top
    top_ax = fig.add_axes([left+0.04, pos0.y1 + 0.02, width-0.04, 0.08])
    side_ax = fig.add_axes([right+0.04, pos1.y0, 0.06, pos1.y1-pos1.y0])
    
    
    # --- horizontal bar
    print(np.abs(shifted_img.detach()[0,0,:,:]).sum())
    s = shifted_img.detach().cpu().numpy()
    #print(np.abs(shifted_img.detach()[0,0,:,:]).sum())
    mfield = (np.abs(s[0,0,:,:]).sum() - 378.8535)/(2306.4561 - 378.8535)
    top_ax.barh([0], [mfield/2], height=0.6, color='blue')
    
    dis, tilt, uflux, pflux, nflux, eflux, area, lp, R = calculate_params_(s[0,0,:,:], 0.15)
    print(R)
    complexity = (R - 339.9386)/(857.3191 - 339.9386)
    side_ax.bar([0], [complexity], width=0.4, color='red')  # single bar

    # --- styling (important)
    top_ax.set_xlim(0, 1)        # adjust based on your scale
    top_ax.set_yticks([])        # hide y
    top_ax.set_xticks([])        # optional: hide x
    top_ax.spines[:].set_visible(False)
    
    # --- styling (important)
    side_ax.set_ylim(0, 1)        # adjust based on your scale
    side_ax.set_yticks([])        # hide y
    side_ax.set_xticks([])        # optional: hide x
    side_ax.spines[:].set_visible(False)
    axs[1].text(1.03, pos1.y0-0.02, "Complexity", transform=axs[1].transAxes,rotation=90, fontsize=15)
    st = str(i+7).zfill(2)
    plt.savefig(f'./plots/plot_{st}.png', dpi=300)
    
files = sorted(glob.glob("./plots/*.png"))

frames = [Image.open(f) for f in files]
frames[0].save(
    "./plots/output.gif",
    save_all=True,
    append_images=frames[1:],
    duration=300,   # milliseconds per frame
    loop=0          # 0 = infinite loop
)