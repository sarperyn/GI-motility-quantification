from sklearn import svm

import os
import numpy as np
import cv2
import glob
import yaml

from src.utils.variable_utils import MADISON_STOMACH
from src.utils.data_utils import min_max_normalize
from src.utils.model_utils import *

    
def load_images_from_folder(mode='train'):

    image_paths = sorted(glob.glob(os.path.join(MADISON_STOMACH, mode, '*image.png')))[:500]
    mask_paths  = sorted(glob.glob(os.path.join(MADISON_STOMACH, mode, '*mask.png')))[:500]

    assert len(image_paths) == len(mask_paths)

    image_features = []
    image_labels = []
    
    for idx in range(len(image_paths)):
        img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_paths[idx], cv2.IMREAD_UNCHANGED)

        img = cv2.resize(img, (256, 256), 
               interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), 
               interpolation = cv2.INTER_LINEAR)
        

        normalized_img = min_max_normalize(img, 
                          min_val=0, 
                          max_val=1)
        
        img_feature = normalized_img.flatten()
        mask_feature = mask.flatten()
        
        image_features.append(img_feature)
        image_labels.append(mask_feature)

    image_features = np.stack(image_features).flatten()
    image_labels = np.stack(image_labels).flatten()   
    assert image_features.shape == image_labels.shape   
    return image_features, image_labels

def get_configs(config_path):

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    svm_model = get_model_class(config)
    kernel    = config['model_config']['kernel']
    scaler = get_class(config['preprocessing']['scaler_class'])
    dim_reduction = get_class(config['preprocessing']['dr_method'])
    save_path  = config['save_path']

    return svm_model, kernel, scaler, dim_reduction, save_path


