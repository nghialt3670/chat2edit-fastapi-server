from ast import List
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from typing import List, Tuple
from torchvision.ops import box_convert
from segment_anything import sam_model_registry, SamPredictor

from chat2edit.tools.base import Segmenter
from chat2edit.utils.image import expand_box


BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
BOX_EXPAND_FACTOR = 0.1
TRANSFORM = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class GroundedSAM(Segmenter):
    def __init__(
        self,
        gdino_checkpoint: str,
        gdino_config: str,
        gdino_device: str,
        sam_checkpoint: str,
        sam_model_type: str,
        sam_device: str,
    ) -> None:
        self.gdino_checkpoint = gdino_checkpoint
        self.gdino_config = gdino_config
        self.gdino_device = gdino_device
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.sam_device = sam_device
        self.gdino_predictor = load_model(gdino_config, gdino_checkpoint, gdino_device)
        sam = sam_model_registry[sam_model_type](sam_checkpoint)
        sam.to(sam_device)
        self.sam_predictor = SamPredictor(sam)

    def __call__(
        self, image: Image.Image, label: str
    ) -> Tuple[List[float], List[np.ndarray]]:
        h, w, _ = np.asarray(image).shape
        caption = label + " ."
        boxes, logits, _ = predict(
            model=self.gdino_predictor,
            image=TRANSFORM(image.convert("RGB"), None)[0],
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=self.gdino_device,
        )
        boxes = box_convert(
            boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy"
        )
        scores = list(map(float, logits))
        boxes = list(map(tuple, boxes.numpy().astype(int)))
        masks = []
        self.sam_predictor.set_image(np.array(image.convert("RGB")))
        for box in boxes:
            expanded_box = expand_box(box, image.size, BOX_EXPAND_FACTOR)
            curr_masks, _, _ = self.sam_predictor.predict(
                box=np.array(expanded_box), multimask_output=False
            )
            mask = curr_masks[0].astype(np.uint8) * 255
            masks.append(mask)

        return scores, masks
