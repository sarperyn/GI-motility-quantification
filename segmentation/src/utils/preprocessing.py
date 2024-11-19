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
    """
    Loads images and masks from a dataset, optionally applies resizing, scaling, 
    and dimensionality reduction.

    Args:
        dataset (str): Path to the dataset folder.
        mode (str): Dataset mode ('train', 'val', or 'test'). Default is 'train'.
        n_samples (int): Number of samples to load. Default is 1000.
        target_resize (tuple, optional): Target size for resizing (width, height). Default is None.
        scaler (str, optional): Name of the scaler to apply ('minmax', 'standard', 'robust', 'normalizer'). Default is None.
        dim_reduction_method (str, optional): Method for dimensionality reduction ('pca', 'umap'). Default is None.
        n_components (int): Number of components for dimensionality reduction. Default is 50.

    Returns:
        tuple: A tuple of images (processed) and masks (original size).
    """
    image_paths = sorted(glob.glob(os.path.join(dataset, mode, '*image.png')))[:n_samples]
    mask_paths  = sorted(glob.glob(os.path.join(dataset, mode, '*mask.png')))[:n_samples]

    assert len(image_paths) == len(mask_paths)
    
    images, masks = load_and_resize_images(image_paths=image_paths, 
                                           mask_paths=mask_paths, 
                                           target_size=(256, 256) if target_resize is None else target_resize)
    
    images = np.array(images)
    masks = np.array(masks)

    images = scale(images, scaler) if scaler is not None else images
    images = dim_reduction(images, n_components, dim_reduction_method) if dim_reduction_method is not None else images

    return images, masks

def load_and_resize_images(image_paths, mask_paths, target_size):
    """
    Loads and resizes images and masks.

    Args:
        image_paths (list): List of paths to the image files.
        mask_paths (list): List of paths to the mask files.
        target_size (tuple): Target size for resizing (width, height).

    Returns:
        tuple: Resized images and masks as lists.
    """
    images = [cv2.resize(cv2.imread(img, 0), target_size) for img in image_paths]  
    masks = [cv2.resize(cv2.imread(mask, 0), target_size) for mask in mask_paths] 
    return images, masks

def scale(data, scaler_name='standard'):
    """
    Scales the data using a specified scaler.

    Args:
        data (np.ndarray): Input data of shape (n_samples, width, height).
        scaler_name (str): Scaler to use ('minmax', 'standard', 'robust', 'normalizer'). Default is 'standard'.

    Returns:
        np.ndarray: Scaled data with the same shape as input.
    """
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
        raise NotImplementedError("Scaler not implemented.")

    data_scaled = scaler.fit_transform(data_flattened)
    data_scaled = data_scaled.reshape(n_samples, width, height)
    print("Scaler is applied.")
    return data_scaled

def dim_reduction(data, n_components=30, method='pca'):
    """
    Applies dimensionality reduction to the data.

    Args:
        data (np.ndarray): Input data of shape (n_samples, width, height).
        n_components (int): Number of components for reduction. Default is 30.
        method (str): Reduction method ('pca', 'umap'). Default is 'pca'.

    Returns:
        np.ndarray: Data with reduced dimensions of shape (n_samples, n_components).
    """
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
        raise NotImplementedError("Dimensionality reduction method not implemented.")

    print("Dimension reduction is applied.")
    return data_low
