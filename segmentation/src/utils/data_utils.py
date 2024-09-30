from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import cv2


class Augmentation():
    def __init__(self):
        print("Augmentation class init")
    
    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def rotate90(self, img, times=1):
        return np.rot90(img, k=times)
    
    def flip(self, img, horizontal=True, vertical=True):
        return np.flipud(np.fliplr(img)) if horizontal and vertical else (np.flipud(img) if vertical else (np.fliplr(img) if horizontal else img))

    def shift(self, img, right_shift, down_shift):
        return np.roll(np.roll(img, right_shift, axis=1), down_shift, axis=0)

def load_csv(filepath):
    return pd.read_csv(filepath)

def save_csv(dataframe, filepath):
    dataframe.to_csv(filepath, index=False)

def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return normalized_tensor

