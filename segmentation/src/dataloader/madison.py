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
    """
    A custom dataset class for processing stomach image and mask pairs.
    Extends the MainDatasetClass to provide dataset handling for training and evaluation.
    """

    def __init__(self, data_path, args, mode='train') -> None:
        """
        Initializes the MadisonStomach dataset.

        Args:
            data_path (str): Path to the dataset directory.
            args (Namespace): Arguments containing configuration (e.g., seed).
            mode (str): Mode of the dataset, either 'train' or 'val/test'.
        """
        self.seed = args.seed
        self.image_paths = sorted(glob.glob(os.path.join(data_path, mode, '*image*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(data_path, mode, '*mask*.png')))

        # Ensure the number of images matches the number of masks
        assert len(self.image_paths) == len(self.mask_paths)

        # Set up transformations and augmentations
        self.transform, self.mask_transform = self.transformation()
        self.augmentations = self.augmentation() if mode == 'train' else None

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of image-mask pairs in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        """
        Retrieves an image and its corresponding mask by index.

        Args:
            index (int): Index of the image-mask pair to retrieve.

        Returns:
            tuple: A tuple containing the transformed (and augmented, if applicable)
                   image and mask as tensors.
        """
        # Load image and mask
        img = cv2.imread(self.image_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_UNCHANGED)

        # Apply transformations
        img = self.transform(img)
        mask = self.mask_transform(mask)

        # Apply augmentations if in training mode
        if self.augmentations is not None:
            torch.manual_seed(self.seed)  # Set seed for deterministic augmentations
            img = self.augmentations(img)
            mask = self.augmentations(mask)

        return img, mask

    def transformation(self):
        """
        Defines the transformation pipeline for the images and masks.

        Returns:
            tuple: Two transformation pipelines, one for images and one for masks.
        """
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),  # Resize images to 256x256
        ])

        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),  # Resize masks to 256x256
        ])

        return img_transform, mask_transform

    def augmentation(self):
        """
        Defines the augmentation pipeline for the images and masks.

        Returns:
            transforms.Compose: Augmentation pipeline applied during training.
        """
        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

        return augmentations
