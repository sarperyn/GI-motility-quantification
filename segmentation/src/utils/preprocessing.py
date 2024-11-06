import os
import numpy as np
import cv2
import glob
import yaml
import umap

from src.utils.variable_utils import MADISON_STOMACH
from src.utils.model_utils import *

from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA


    
def load_images_from_folder(dataset=MADISON_STOMACH, 
                            mode='train', 
                            n_samples=1000, 
                            target_resize=None, 
                            scaler=None,
                            dim_reduction_method=None,
                            n_components=50):

    image_paths = sorted(glob.glob(os.path.join(dataset, mode, '*image.png')))[:n_samples]
    mask_paths  = sorted(glob.glob(os.path.join(dataset, mode, '*mask.png')))[:n_samples]

    assert len(image_paths) == len(mask_paths)
    
    images, masks = load_and_resize_images(image_paths=image_paths, 
                                           mask_paths=mask_paths, 
                                           target_size=(256, 256) if target_resize==None else target_resize)
    
    images = np.array(images)
    masks = np.array(masks)

    images = scale(images, scaler) if scaler != None else images
    images = dim_reduction(images, n_components, dim_reduction_method) if dim_reduction_method != None else images

    return images, masks

def load_and_resize_images(image_paths, mask_paths, target_size):
    images = [cv2.resize(cv2.imread(img, 0), target_size) for img in image_paths]  
    masks = [cv2.resize(cv2.imread(mask, 0), target_size) for mask in mask_paths] 
    return images, masks

def scale(data, scaler_name='standard'):

    n_samples, width, height = data.shape
    data_flattened = data.reshape(n_samples, -1)

    if scaler_name == 'minmax':
        scaler = MinMaxScaler()

    elif scaler_name == 'standard':
        scaler = StandardScaler() 

    elif scaler_name == 'robust':
        scaler = RobustScaler() 
    
    elif scaler_name == 'normalizer':
        scaler = Normalizer()

    else:
        raise NotImplementedError
    
    data_scaled = scaler.fit_transform(data_flattened)
    data_scaled = data_scaled.reshape(n_samples, width, height)
    print("Scaler is applied.")
    return data_scaled

def dim_reduction(data, n_components=30, method='pca'):

    n_samples, width, height = data.shape
    data_low = data.reshape(n_samples, -1)

    n_components = min(n_components, n_samples)

    if method == 'pca':
        pca = PCA(n_components=n_components)
        data_low = pca.fit_transform(data_low)
    
    elif method == 'umap':
        umap_red = umap.UMAP(n_components=n_components, random_state=42)
        data_low = umap_red.fit_transform(data_low)

    else:
        raise NotImplementedError
    
    data_low = data_low.reshape(n_samples, n_components)

    print("Dimension reduction is applied.")

    return data_low

