import torch
import cv2
import os
import glob
from typing import Any
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from torchvision import transforms

from src.dataloader.datasets import MainDatasetClass


class MadisonStomach(MainDatasetClass):

    def __init__(self, data_path, args, mode='train') -> None:

        self.seed = args.seed
        self.image_paths = sorted(glob.glob(os.path.join(data_path, mode, '*image*.png')))
        self.mask_paths  = sorted(glob.glob(os.path.join(data_path, mode, '*mask*.png')))

        assert len(self.image_paths) == len(self.mask_paths)

        self.transform, self.mask_transform = self.transformation()
        self.augmentations = self.augmentation() if mode=='train' else None
            
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)
        
        img = self.transform(img)
        mask = self.mask_transform(mask)

        if self.augmentations is not None:
            torch.manual_seed(self.seed)
            img = self.augmentations(img)
            mask = self.augmentations(mask)

        return img, mask
    
    def transformation(self):

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])

        return img_transform, mask_transform
    
    def augmentation(self):

        augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        
        return augmentations 
