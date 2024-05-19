from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import List, Tuple


class Inpainter(ABC):
    @abstractmethod
    def __call__(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        pass


class Segmenter(ABC):
    @abstractmethod
    def __call__(
        self, image: Image.Image, label: str
    ) -> Tuple[List[float], List[np.ndarray]]:
        pass
