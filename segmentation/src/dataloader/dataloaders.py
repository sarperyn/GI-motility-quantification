import torch
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from PIL import Image
# import sys
# sys.path.append(os.path.dirname(os.getcwd()))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2

from ..utils.variable_utils import MADISON_DATA
        
class MadisonStomach(Dataset):

    def __init__(self, data_path, args, mode='train', augment=False) -> None:

        self.seed = args.seed
        self.image_paths = sorted(glob.glob(os.path.join(data_path, mode, '*image*.png')))
        self.mask_paths  = sorted(glob.glob(os.path.join(data_path, mode, '*mask*.png')))

        assert len(self.image_paths) == len(self.mask_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

        if mode == 'train':
            self.augment = augment
        else:
            self.augment = False

        if self.augment:
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)
        
        img = self.transform(img)
        mask = self.mask_transform(mask)

        if self.augment:
            torch.manual_seed(self.seed)
            img = self.augmentation_transforms(img)
            torch.manual_seed(self.seed)
            mask = self.augmentation_transforms(mask)

        return img, mask

if __name__ == '__main__':

    dataset    = MadisonStomach(segmentation_path='/home/syurtseven/gsoc-2024/data/stomach_masks',augment=False)
    dataloader = DataLoader(dataset=dataset, batch_size=5)
    data = next(iter(dataloader))

    img, mask = data
    print(len(dataloader))
    print(img.shape)
    print(mask.shape)