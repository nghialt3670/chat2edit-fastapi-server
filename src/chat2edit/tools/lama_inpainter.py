import cv2
import torch
import numpy as np
from PIL import Image
from iopaint.model import LaMa
from iopaint.schema import InpaintRequest

from chat2edit.tools.base import Inpainter
from chat2edit.utils.image import expand_mask


MASK_EXPANDING_ITERATIONS = 10


class LaMaInpainter(LaMa, Inpainter):
    def __init__(self, checkpoint: str, device: str) -> None:
        self.model = torch.jit.load(checkpoint, "cpu").eval().to(device)
        self.device = device

    def __call__(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        expanded_mask = expand_mask(mask, self.MASK_EXPANDING_ITERATIONS)
        config = InpaintRequest(hd_strategy="Resize")
        inpainted_image = super().__call__(image, expanded_mask, config)
        return Image.fromarray(inpainted_image.astype(np.uint8))
