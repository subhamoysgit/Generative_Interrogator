import glob
from torch.utils.data import Dataset
import numpy as np
import h5py
import torch, torchvision
from PIL import Image
import pandas as pd

class SHARPdataset(Dataset):

    def __init__(self, data_path: str, include_sharp: bin=False,
                 data_stride:int = 1, image_size = 256,
                 datatype=np.float32,
                 crop:bool = False,
                 use_first:bool = False):
        '''
            Initializes image files in the dataset
            
            Args:
                data_path (str): path to the folder containing the SHARP h5
                data_stride (int): stride to use when loading the images to work 
                                    with a reduced version of the data
                datatype (numpy.dtype): datatype to use for the images
                param (bin): add SHARP parameters
                use_first (bin): use only first channel i.e. radial field image
        '''
        self.use_first = use_first
        self.data_path = data_path
        self.image_size = image_size
        self.crop = crop
        df = pd.read_csv('./data/index_sharps.csv')
        if self.crop: 
            df = df[df['naxis1']>self.image_size][df['naxis2']>self.image_size]
            self.sharp_images = list(df['file'])
        else:
            self.sharp_images = list(df['file'])
            #self.sharp_images = glob.glob(data_path + "/*.h5")
        if data_stride>1:
            self.sharp_images = self.sharp_images[::data_stride]
        
        self.include_sharp = include_sharp
        self.datatype = datatype
        
        if self.include_sharp:
            self.sharp_params = ['usflux', 'meangam',
                    'meangbt', 'meangbz', 
                    'meangbh', 'meanjzd', 
                    'meanjzh', 'totusjz',
                    'totusjh', 'meanalp', 
                    'absnjzh', 'savncpp',
                    'meanpot', 'totpot', 
                    'meanshr', 'shrgt45',
                    'r_value', 'size',
                    'area', 'nacr',
                    'size_acr', 'area_acr',
                    'mtot', 'mnet',
                    'mpos_tot', 'mneg_tot',
                    'mmean', 'mstdev', 'mskew']
            self.sharp_data = df[self.sharp_params]
            self.sharp_data.fillna(0, inplace=True)
            self.p_max = list(self.sharp_data.max())
            self.sharp_data = (self.sharp_data - self.sharp_data.min())/(self.sharp_data.max() - self.sharp_data.min())

    def __len__(self):
        '''
            Calculates the number of images in the dataset
                
            Returns:
                int: number of images in the dataset
        '''
        return len(self.sharp_images)

    def __getitem__(self, idx):
        '''
            Retrieves an image from the dataset and creates a copy of it,
            applying a series of random augmentations to the copy.

            Args:
                idx (int): index of the image to retrieve
                
            Returns:
            
        '''
        file = h5py.File(self.sharp_images[idx])
        key = list(file.keys())[0]
        data = np.array(file[key])
        #data = (np.clip(data, -1000, 1000)/1000 + 1)/2
        data = np.clip(data, -1000, 1000)/1000
        # data = np.nan_to_num(data, copy=False, nan=0.5, posinf=1, neginf=0)
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=1, neginf=-1)
        data = data[None, :, :, :]
        data = torch.from_numpy(data.astype(self.datatype))
        if self.crop:
            cen_row = data.shape[2]//2
            cen_col = data.shape[3]//2
            half_size = self.image_size//2
            data = data[:, :,
                        (cen_row - half_size):(cen_row + half_size),
                        (cen_col - half_size):(cen_col + half_size)]
        else:
            resize = torchvision.transforms.Resize((self.image_size, self.image_size),
                                                   antialias=True)
            data = resize(data)
        
        if self.include_sharp:
            params = np.array(list(self.sharp_data.iloc[idx]))
            # params = np.array([params[i]/self.p_max[i] for i in range(len(self.sharp_params))])
            params = torch.from_numpy(params.astype(self.datatype))
            if self.use_first:
                return data[0,:1,:,:], params, self.sharp_images[idx]
            else:
                return data[0,:,:,:], params, self.sharp_images[idx]
        else:
            if self.use_first:
                return data[0,:1,:,:], self.sharp_images[idx]
            else:
                return data[0,:,:,:], self.sharp_images[idx]