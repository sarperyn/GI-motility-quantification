from typing import Any
from abc import ABC, abstractmethod
from torch.utils.data import Dataset



class MainDatasetClass(Dataset, ABC):

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    @abstractmethod
    def transformation(self):
        pass

    @abstractmethod
    def augmentation(self):
        pass