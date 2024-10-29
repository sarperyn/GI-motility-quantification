from torchvision import transforms
import numpy as np
import pandas as pd
import torch
import cv2


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

