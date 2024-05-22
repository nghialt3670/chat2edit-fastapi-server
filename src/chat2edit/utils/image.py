from base64 import b64encode, b64decode
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

from io import BytesIO
from PIL import Image
from typing import Tuple, Literal
from cv2 import (
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
)


KERNEL = getStructuringElement(MORPH_ELLIPSE, (3, 3))


def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphologyEx(mask, MORPH_OPEN, KERNEL)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # type: ignore
    return mask


def expand_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask


def expand_box(
    box: Tuple[int, int, int, int], image_size: Tuple[int, int], factor: float
) -> Tuple[int, int, int, int]:
    width, height = image_size
    xmin, ymin, xmax, ymax = box
    x_offset = (xmax - xmin) * factor / 2
    y_offset = (ymax - ymin) * factor / 2
    xmin = max(0, xmin - x_offset)
    ymin = max(0, ymin - y_offset)
    xmax = min(width, xmax + y_offset)
    ymax = min(height, ymax + y_offset)
    return xmin, ymin, xmax, ymax


def cut_image_from_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    mask = Image.fromarray(mask)
    box = mask.getbbox()
    xmin, ymin, xmax, ymax = box
    cut_image = Image.new("RGBA", (xmax - xmin, ymax - ymin))
    cut_image.paste(image.crop(box), (0, 0), mask.crop(box))
    return cut_image


def get_mask(
    mask: np.ndarray, size: Tuple[int, int], offsets: Tuple[int, int]
) -> np.ndarray:
    full_mask = np.zeros(size[::-1], dtype=np.uint8)
    x_offset, y_offset = offsets
    x_end = min(x_offset + mask.shape[1], size[0])
    y_end = min(y_offset + mask.shape[0], size[1])
    full_mask[y_offset:y_end, x_offset:x_end] = mask[
        : y_end - y_offset, : x_end - x_offset
    ]
    mask = mask.astype(np.uint8)
    full_mask = full_mask.astype(np.uint8)
    return full_mask


def image_to_mask(image: Image.Image) -> np.ndarray:
    alpha_channel = image.split()[-1]
    mask = np.asarray(alpha_channel)
    mask = np.where(mask == 0, 0, 255)
    return mask


def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def pil_image_to_data_url(image: Image.Image) -> str:
    image_bytes = BytesIO()
    image.save(image_bytes, image.format)
    mimetype = f"image/{image.format.lower()}"
    base64 = b64encode(image_bytes.getvalue()).decode("utf-8")
    return f"data:{mimetype};base64,{base64}"


def data_url_to_pil_image(data_url: str) -> Image.Image:
    base64 = data_url[data_url.index(",") + 1 :]
    image_bytes = BytesIO(b64decode(base64))
    return Image.open(image_bytes)
