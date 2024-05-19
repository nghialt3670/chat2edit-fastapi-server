from typing import List, Tuple
from PIL.Image import Image
from numpy import ndarray
from chat2edit.tools.base import Inpainter, Segmenter


class Toolkit:
    def __init__(self, segmenter: Segmenter, inpainter: Inpainter) -> None:
        self._segmenter = segmenter
        self._inpainter = inpainter

    def segment(self, image: Image, label: str) -> Tuple[List[float], List[ndarray]]:
        return self._segmenter(image, label)

    def inpaint(self, image: Image, mask: ndarray) -> Image:
        return self._inpainter(image, mask)
