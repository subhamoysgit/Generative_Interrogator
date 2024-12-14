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
from dataset import SHARPdataset
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics.pairwise import cosine_similarity

def calculate_srij_R(s, th):
    """produces image of High-Gradient Polarity Transition Region (R)

    Args:
        s (np.ndarray): input magnetic patch
        th (float): threshold

    Returns:
        image (np.ndarray): R image
    """  
    pos = (s>th).astype(float)
    neg = (s<-th).astype(float)
    kernel = np.ones((7, 7))
    pos_d = binary_dilation(pos, kernel)
    neg_d = binary_dilation(neg, kernel)
    return s*pos_d*neg_d


def calculate_params(s, th):
    """calculates physical parameters from magnetic patches

    Args:
        s (np.ndarray): input magnetic patch
        th (float): threshold

    Returns:
        physical parameters
    """
    pos = (s>th).astype(float)
    neg = (s<-th).astype(float)
    if pos.sum()>0 and neg.sum()>0:
        ind_p_y = np.array(np.where(pos==1))[0]
        ind_p_x = np.array(np.where(pos==1))[1]
        ind_n_y = np.array(np.where(neg==1))[0]
        ind_n_x = np.array(np.where(neg==1))[1]
        cen_p_y = np.sum(ind_p_y*s[pos==1])/np.sum(s[pos==1])
        cen_p_x = np.sum(ind_p_x*s[pos==1])/np.sum(s[pos==1])
        cen_n_y = np.sum(ind_n_y*s[neg==1])/np.sum(s[neg==1])
        cen_n_x = np.sum(ind_n_x*s[neg==1])/np.sum(s[neg==1])
        
        dis = (cen_n_x - cen_p_x)**2 + (cen_n_y - cen_p_y)**2
        tilt = np.arcsin((cen_n_y - cen_p_y)/(dis**0.5))
        dis = (dis**0.5)/(128*np.sqrt(2))
        uflux = np.abs(s).sum()/(128*128)
        pflux = np.abs(s[s>th]).sum()/np.sum(s>th)
        nflux = np.abs(s[s<-th]).sum()/np.sum(s<-th)
        eflux = s[s>th].sum() + s[s<-th].sum()
        area = (pos.sum() + neg.sum())/(128*128)
        lp = cen_p_x>cen_n_x
        R = np.sum(np.abs(calculate_srij_R(s, th)))
        return dis, tilt, uflux, pflux, nflux, eflux, area, lp, R
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
   
   
   
def calculate_params_(s, th):
    """calculates physical parameters from magnetic patches

    Args:
        s (np.ndarray): input magnetic patch
        th (float): threshold

    Returns:
        physical parameters
    """
    pos = (s>th).astype(float)
    neg = (s<-th).astype(float)
    ind_p_y = np.array(np.where(pos==1))[0]
    ind_p_x = np.array(np.where(pos==1))[1]
    ind_n_y = np.array(np.where(neg==1))[0]
    ind_n_x = np.array(np.where(neg==1))[1]
    cen_p_y = np.sum(ind_p_y*s[s>th])/np.sum(s[s>th])
    cen_p_x = np.sum(ind_p_x*s[s>th])/np.sum(s[s>th])
    cen_n_y = np.sum(ind_n_y*s[s<-th])/np.sum(s[s<-th])
    cen_n_x = np.sum(ind_n_x*s[s<-th])/np.sum(s[s<-th])
    dis = (cen_n_x - cen_p_x)**2 + (cen_n_y - cen_p_y)**2
    tilt = np.arcsin((cen_n_y - cen_p_y)/(dis**0.5))
    dis = (dis**0.5)/(128*np.sqrt(2))
    uflux = np.abs(s).sum()/(128*128)
    pflux = np.abs(s[s>0]).sum()/np.sum(s>0)
    nflux = np.abs(s[s<0]).sum()/np.sum(s<0)
    eflux = s[s>th].sum() + s[s<-th].sum()
    area = (pos.sum() + neg.sum())/(128*128)
    lp = cen_p_x-cen_n_x
    R = np.sum(np.abs(calculate_srij_R(s, th)))
    return dis, tilt, uflux, pflux, nflux, eflux, area, lp, R
    
    
def hyperplane(latent_space, clf):
    """projects latent vector on decision boundaries (hyperplane) defined by clf
    """ 
    latent_plane = latent_space.copy()
    for i in range(latent_plane.shape[0]):
        dot = np.dot(clf.coef_[0][1:], latent_space[i,1:])
        latent_plane[i,0]  = (-clf.intercept_[0] - dot)/clf.coef_[0][0]
    return latent_plane


def project_latent_2d(latent_space,
                      latent_plane,
                      clf,
                      type='custom'):
    
    """Performs different 2D projections of the latent space.
    Custom projection works such that the decision boundary becomes a straight line in 2D.
    """    
    type = 'custom'
    latent_master = np.vstack((latent_space, latent_plane))

    if type=='custom':
        embeddings_2d = latent_master[:,:2].copy()
        for i in range(latent_master.shape[0]):
            dot = np.dot(clf.coef_[0][1:], latent_master[i,1:])
            embeddings_2d[i,1]  = dot/clf.coef_[0][1]
    elif type=='pca':
        pca_2d = PCA(n_components=2)
        embeddings_2d = pca_2d.fit_transform(latent_master)
    else:
        tsne_2d = TSNE(n_components=2, random_state=0)
        embeddings_2d = tsne_2d.fit_transform(latent_master)

    M = np.max(embeddings_2d[:latent_master.shape[0]//2,:], axis=0)
    m = np.min(embeddings_2d[:latent_master.shape[0]//2,:], axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)
    return embeddings_2d


def image_on_latent(gen_imgs, embeddings_2d,
                                     plane_2d, type  = 'custom',
                                     title = None, 
                                     mask = False,
                                     pol_connect=False,
                                     fig = None,
                                     ax = None):
    """Creates a scatter plot with image overlays.

    Args:
        gen_imgs (np.ndarray): generated images
        embeddings_2d (np.ndarray): 2-dimensional embeddings
        plane_2d (_type_): _description_
        type (str): ype of projection. Defaults to 'custom'.
        title : Name of the physical parameter. Defaults to None.
        mask (bool): Apply masking. Defaults to False.
        pol_connect (bool): Connect polarities. Defaults to False.
        fig : Defaults to None.
        ax : Defaults to None.

    Returns:
        2D embedding (latent) space with Image and decision boundary overlaid
    """    
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 5e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)
        
    th = 0.15
    ep = 0.0
    cmap = plt.cm.coolwarm
    norm = matplotlib.colors.Normalize(vmin=-1000, vmax=1000)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    coefficients = np.polyfit(plane_2d[:,0], plane_2d[:,1], 1)  # 1 indicates linear fit
    slope, intercept = coefficients
    
    dd = []
    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 3.0)
        img = gen_imgs[idx, 0, :, :]
        excess = img[img>th].sum() + img[img<-th].sum()#np.sum(np.abs(img))/(128*128)#
        if mask: img = calculate_srij_R(img, 0.15) 
        img = Image.fromarray(img)
        img = functional.resize(img, thumbnail_size)
        img = np.array(img)
        
        if pol_connect:
            s = img
            pos = (s>th).astype(float)
            neg = (s<-th).astype(float)
            ind_p_y = np.array(np.where(pos==1))[0]
            ind_p_x = np.array(np.where(pos==1))[1]
            ind_n_y = np.array(np.where(neg==1))[0]
            ind_n_x = np.array(np.where(neg==1))[1]
            
            if np.sum(s[s>th])>0 and np.sum(s[s<-th])<0:
                cen_p_y = np.sum(ind_p_y*s[s>th])/np.sum(s[s>th])
                cen_p_x = np.sum(ind_p_x*s[s>th])/np.sum(s[s>th])
                cen_n_y = np.sum(ind_n_y*s[s<-th])/np.sum(s[s<-th])
                cen_n_x = np.sum(ind_n_x*s[s<-th])/np.sum(s[s<-th])
                img = 0.5*(1+np.repeat(img[:, :, np.newaxis], 3, axis=2))
                img = cv2.line(img, [int(cen_p_x), int(cen_n_x)], 
                            [int(cen_p_y), int(cen_n_y)], (0, 0, 1) , 1) 
        img = 0.5*(1+np.repeat(img[:, :, np.newaxis], 3, axis=2))
        img_over = img.copy()
        img_over[:,:,0] = 0.5*(1+excess/1000)
        img_over[:,:,1] = 0.0
        img_over[:,:,-1] = 0.5*(1-excess/1000)
        img = (1-ep)*img + ep*img_over
        im = osb.OffsetImage(img)#, cmap=plt.cm.gray_r)
        im_ = im.get_children()[0]
        im_.set_clim(vmin=-1, vmax=1)
        img_box = osb.AnnotationBbox(
            im,
            [1-embeddings_2d[idx][1],embeddings_2d[idx][0]],
            pad=0.1,
        )
        ax.add_artist(img_box)
        dist = (embeddings_2d[idx][0]*slope - embeddings_2d[idx][1] + intercept)/np.sqrt(1+slope**2)
        dd.append(dist)
    idx_min = shown_images_idx[np.argmin(np.array(dd))]
    idx_max = shown_images_idx[np.argmax(np.array(dd))]
    print(rcp["figure.figsize"][0], thumbnail_size)
    if type=='custom':
        line_idx = np.argsort(plane_2d[:,0])
        ax.plot(1-plane_2d[line_idx,1],plane_2d[line_idx,0],'-r',zorder=100)
    else:
        ax.plot(1-plane_2d[:,1],plane_2d[:,0],'*r',zorder=100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.set_title(title, loc='left')
    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")
    coord_min = [1-embeddings_2d[idx_min][1], embeddings_2d[idx_min][0]]
    coord_max = [1-embeddings_2d[idx_max][1], embeddings_2d[idx_max][0]]
    return fig, coord_min, coord_max, gen_imgs[idx_min, 0, :, :], gen_imgs[idx_max, 0, :, :]


def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.
  
  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].
  
  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.
  
  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.
  
  Raises:
    LinAlgError: If there are more than two condition boundaries and the method fails 
                 to find a projected boundary orthogonal to all condition boundaries.
  Source: https://github.com/genforce/interfacegan/blob/master/utils/manipulator.py
  """
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  elif len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)
  else:
    for cond_boundary in args:
      assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
              cond_boundary.shape[1] == primal.shape[1])
    cond_boundaries = np.squeeze(np.asarray(args))
    A = np.matmul(cond_boundaries, cond_boundaries.T)
    B = np.matmul(cond_boundaries, primal.T)
    x = np.linalg.solve(A, B)
    new = primal - (np.matmul(x.T, cond_boundaries))
    return new / np.linalg.norm(new)




def fetch_n_neighbor_filenames(query_embedding, embeddings_dict, dist_type,
                               start_date=None, end_date=None, num_images=9):
    """Function to fetch filenames of nearest neighbors

    Args:
        query_embedding (np.ndarray): Embedding for query image.
        embeddings_dict (dict[filenames, embeddings]): Dictionary mapping filenames to embeddings.
        distance (str): Distance metric.
        num_images (int): Number of similar images to return. Defaults to 9.

    Returns:
        filenames: Filenames of images similar to the given embedding.

    """
    embeddings = embeddings_dict['embeddings']
    filenames = embeddings_dict['filenames']

    if dist_type.upper() == "EUCLIDEAN":
        distances = embeddings - query_embedding
        distances = np.power(distances, 2).sum(-1).squeeze()
    elif dist_type.upper() == "COSINE":
        distances = -1*cosine_similarity(embeddings,
                                         np.array([query_embedding]))
        distances = distances[:, 0]

    # Filter by date
    if start_date is not None and end_date is not None:
        # start_date = datetime.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')    #https://github.com/hits-sdo/hits-sdo-downloader/blob/main/search_download/downloader.py
        # end_date = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')
        dates = np.array([datetime.datetime.strptime(filename.split('_')[0], '%Y%m%d').date() for filename in filenames])
        
        # my_datetime = datetime.datetime.combine(my_date, datetime.time(23, 59, 59))

        mask = (dates >= start_date) & (dates <= end_date) 
        distances = distances[mask]
        filenames = np.array(filenames)[mask]

    nn_indices = np.argsort(distances)[:num_images]
    nearest_neighbors = [filenames[idx] for idx in nn_indices]
    return nearest_neighbors




def main():
    DATA_DIR = '/d0/kvandersande/sharps_hdf5/'
    data_sharp=SHARPdataset(DATA_DIR,use_first=True, image_size=128)
    df = {'File': [], 'USFLUX': [],
          'PFLUX':[], 'NFLUX':[],
          'PSEP': [], 'AREA': [],
          'R': [], 'LP': [],
          'TILT': [],'EFLUX':[]}
    for i in range(len(data_sharp)):
        real, f = data_sharp[i]
        dis, tilt, uflux, pflux, nflux, eflux, area, lp, R = calculate_params_(real[0,:,:].numpy(), 0.15)
        df['File'].append(f)
        df['USFLUX'].append(uflux)
        df['AREA'].append(area)
        df['R'].append(R)
        df['PSEP'].append(dis)
        df['LP'].append(lp)
        df['TILT'].append(tilt)
        df['PFLUX'].append(pflux)
        df['NFLUX'].append(nflux)
        df['EFLUX'].append(eflux)
        print(i, f, '--uflux--', uflux, '--lead pol--', lp, '--R--', R)
        
    df = pd.DataFrame(df)
    df.to_csv('SHARP_calc_params_new.csv')
        
if __name__ == '__main__':
    main()