from typing import Any
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class MainDatasetClass(Dataset, ABC):
    """
    Abstract base class for creating custom PyTorch datasets.
    Extends PyTorch's Dataset class and enforces the implementation
    of transformation and augmentation methods in derived classes.
    """

    def __init__(self) -> None:
        """
        Initializes the dataset object.
        This base implementation does not define any attributes but
        can be extended in derived classes as needed.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        This method must be implemented in derived classes to provide
        the correct dataset size.
        """
        pass

    def __getitem__(self, index) -> Any:
        """
        Retrieves a single data sample and its associated label (if applicable)
        based on the given index.

        Args:
            index: Index of the data sample to retrieve.

        Returns:
            A single data sample. This method should be overridden by subclasses.
        """
        return super().__getitem__(index)

    @abstractmethod
    def transformation(self):
        """
        Applies transformations to the dataset samples.
        This method must be implemented in derived classes to define
        specific transformations (e.g., resizing, normalization).

        Returns:
            Transformed dataset samples.
        """
        pass

    @abstractmethod
    def augmentation(self):
        """
        Applies augmentations to the dataset samples.
        This method must be implemented in derived classes to define
        specific augmentations (e.g., flipping, rotations, random crops).

        Returns:
            Augmented dataset samples.
        """
        pass
