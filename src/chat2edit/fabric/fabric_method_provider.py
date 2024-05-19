from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, TypeVar, Union
from PIL import Image as ImageModule
import numpy as np

from chat2edit.core.exec_signal import ExecSignal
from chat2edit.core.message import SysMessage
from chat2edit.core.method_provider import MethodProvider
from chat2edit.fabric.fabric_models import (
    FabricCanvas,
    FabricCollection,
    FabricGroup,
    FabricImage,
    FabricImageObject,
    FabricObject,
    FabricTextbox,
)
from chat2edit.tools.toolkit import Toolkit
from chat2edit.utils.image import data_url_to_pil_image, pil_image_to_data_url


Image = TypeVar("Image", FabricCollection, None)
Object = TypeVar("Object", FabricImageObject, None)
Text = TypeVar("Text", FabricTextbox, None)


class FabricMethodProvider(MethodProvider):
    def __init__(self, toolkit: Toolkit) -> None:
        super().__init__()
        self._toolkit = toolkit

    @MethodProvider.provide
    def response(self, text: str, images: Optional[List[Image]] = None) -> None:
        if images is None:
            images = []
        self._set_signal(
            status="info",
            text="",
            sys_message=SysMessage(status="success", text=text, attachments=images),
        )

    @MethodProvider.provide
    def move(
        self,
        image: Image,
        target: Union[Image, Object, Text],
        position: Tuple[int, int],
    ) -> None:
        if not isinstance(image, FabricCanvas):
            self._signal(
                status="error",
                text="The 'image' argument in the 'move' method must be an instance of 'Image'",
            )
            return

        if not isinstance(target, (FabricImageObject, FabricGroup, FabricTextbox)):
            self._signal(
                status="error",
                text="The 'target' argument in the 'move' method must be an instance of 'Image', 'Object', or 'Text'",
            )
            return

        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not all(isinstance(coord, int) for coord in position)
        ):
            self._signal(
                status="error",
                text="The 'position' argument in the 'insert' method must be a tuple with two integer elements.",
            )
            return

        if not image.contains(target):
            self._signal(status="error", text="The target is not within the image.")
            return

        if (
            position[0] > image.backgroundImage.width
            or position[1] > image.backgroundImage.width
        ):
            self._signal(
                status="warning",
                text="The specified position exceeds the size of the image.",
            )

        target.left = position[0]
        target.top = position[1]

        if isinstance(target, FabricImageObject) and not target.inpainted:
            self._inpaint(image, target)

    @MethodProvider.provide
    def rotate(
        self,
        image: Image,
        target: Union[Image, Object, Text],
        angle: float,
        direction: Literal["cw", "ccw"],
    ) -> None:
        if not isinstance(image, FabricCanvas):
            self._signal(
                status="error",
                text="The 'image' argument in the 'rotate' method must be an instance of 'Image'",
            )
            return

        if not isinstance(target, FabricImageObject, FabricGroup, FabricTextbox):
            self._signal(
                status="error",
                text="The 'target' argument in the 'rotate' method must be an instance of 'Image', 'Object', or 'Text'",
            )
            return

        if not isinstance(target, float, int):
            self._signal(
                status="error",
                text="The 'angle' argument in the 'rotate' method must be a number",
            )
            return

        if direction not in ["cw", "ccw"]:
            self._set_signal(
                status="error",
                text="The 'direction' argument in the 'rotate' method must be either 'cw' or 'ccw'",
            )
            return

        if not image.contains(target):
            self._set_signal(status="error", text="The target is not within the image.")
            return

        if direction == "cw":
            target.angle += angle
        else:
            target.angle -= angle

        if isinstance(target, FabricImageObject) and not target.inpainted:
            self._inpaint(image, target)

    @MethodProvider.provide
    def flip(
        self, image: Image, target: Union[Image, Object, Text], axis: Literal["x", "y"]
    ) -> None:
        if not isinstance(image, FabricCanvas):
            self._set_signal(
                status="error",
                text="The 'image' argument in the 'flip' method must be an instance of 'Image'",
            )
            return

        if not isinstance(target, FabricImageObject, FabricGroup, FabricTextbox):
            self._set_signal(
                status="error",
                text="The 'target' argument in the 'flip' method must be an instance of 'Image', 'Object', or 'Text'",
            )
            return

        if axis not in ["x", "y"]:
            self._set_signal(
                status="error",
                text="The 'axis' argument in the 'flip' method must be either 'x' or 'y'",
            )
            return

        if not image.contains(target):
            self._set_signal(status="error", text="The target is not within the image.")
            return

        if isinstance(target, FabricImageObject) and not target.inpainted:
            self._inpaint(image, target)

        if axis == "x":
            target.flipX = not target.flipX
        else:
            target.flipY = not target.flipY

        if isinstance(target, FabricImageObject) and not target.inpainted:
            self._inpaint(image, target)

    @MethodProvider.provide
    def scale(
        self,
        image: Image,
        target: Union[Image, Object, Text],
        factor: float,
        axis: Optional[Literal["x", "y"]] = None,
    ) -> None:
        if not isinstance(image, FabricCanvas):
            self._set_signal(
                status="error",
                text="The 'image' argument in the 'scale' method must be an instance of 'Image'",
            )
            return

        if not isinstance(target, FabricImageObject, FabricGroup, FabricTextbox):
            self._set_signal(
                status="error",
                text="The 'target' argument in the 'scale' method must be an instance of 'Image', 'Object', or 'Text'",
            )
            return

        if not isinstance(factor, (float, int)):
            self._set_signal(
                status="error",
                text="The 'factor' argument in the 'scale' method must be a number",
            )
            return

        if axis not in (None, "x", "y"):
            self._set_signal(
                status="error",
                text="The 'axis' argument in the 'scale' method must be either 'x', 'y', or None",
            )
            return

        if not image.contains(target):
            self._set_signal(status="error", text="The target is not within the image.")
            return

        if axis is None:
            target.scaleX *= factor
            target.scaleY *= factor
        elif axis == "x":
            target.scaleX *= factor
        elif axis == "y":
            target.scaleY *= factor

        if isinstance(target, FabricImageObject) and not target.inpainted:
            self._inpaint(image, target)

    @MethodProvider.provide
    def insert(
        self,
        target: Union[Image, Object, Text],
        image: Image,
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not isinstance(image, FabricCanvas):
            self._set_signal(
                status="error",
                text="The 'image' argument in the 'insert' method must be an instance of 'Image'",
            )
            return

        if not isinstance(target, FabricImageObject, FabricGroup, FabricTextbox):
            self._set_signal(
                status="error",
                text="The 'target' argument in the 'insert' method must be an instance of 'Image', 'Object', or 'Text'",
            )
            return

        if position is not None:
            if (
                not isinstance(position, tuple)
                or len(position) != 2
                or not all(isinstance(coord, int) for coord in position)
            ):
                self._set_signal(
                    status="error",
                    text="The 'position' argument in the 'insert' method must be a tuple with two integer elements.",
                )
                return

            if (
                position[0] > image.backgroundImage.width
                or position[1] > image.backgroundImage.width
            ):
                self._set_signal(
                    status="warning",
                    text="The specified position exceeds the size of the image.",
                )

        image.add(target)
        if position is not None:
            target.left = position[0]
            target.top = position[1]

    @MethodProvider.provide
    def remove(self, target: Union[Image, Object, Text], image: Image) -> None:
        if not isinstance(image, FabricCanvas):
            self._set_signal(
                status="error",
                text="The 'image' argument in the 'remove' method must be an instance of 'Image'",
            )
            return

        if not isinstance(target, FabricImageObject, FabricGroup, FabricTextbox):
            self._set_signal(
                status="error",
                text="The 'target' argument in the 'remove' method must be an instance of 'Image', 'Object', or 'Text'",
            )
            return

        if not image.contains(target):
            self._set_signal(status="error", text="The target is not within the image.")
            return

        image.remove(target)
        if isinstance(target, FabricImageObject) and not target.inpainted:
            self._inpaint(image, target)

    def replace(self, target: Union[Image, Object, Text], prompt: str) -> None:
        pass

    @MethodProvider.provide
    def apply_filter(
        self,
        target: Union[Image, Object],
        filter_name: str,
        filter_value: float = 1.0,
    ) -> None:
        if not isinstance(target, (FabricImageObject, FabricCollection)):
            self._set_signal(
                status="error",
                text="Can only apply filter on 'Image' or 'Object' in 'apply_filter' method",
            )
            return

        filt = None
        if self._compare_filter_names(filter_name, ["grayscale"]):
            filt = {"type": "grayscale"}
        elif self._compare_filter_names(filter_name, ["invert", "negative"]):
            filt = {"type": "invert"}
        elif self._compare_filter_names(filter_name, ["brightness"]):
            filt = {"type": "brightness", "brightness": filter_value}
        elif self._compare_filter_names(filter_name, ["blur"]):
            filt = {"type": "blur", "blur": filter_value}
        elif self._compare_filter_names(filter_name, ["contrast"]):
            filt = {"type": "contrast", "contrast": filter_value}
        elif self._compare_filter_names(filter_name, ["noise"]):
            filt = {"type": "noise", "noise": filter_value}
        elif self._compare_filter_names(filter_name, ["pixelate"]):
            filt = {"type": "pixelate", "blocksize": filter_value * 10}
        # elif self._compare_filter_names(filter_name, ["temperature", "warmth"]):
        #     filt = Filter(name="temperature", value=filter_value)
        # elif self._compare_filter_names(filter_name, ["saturation"]):
        #     filt = Filter(name="saturation", value=filter_value)
        # elif self._compare_filter_names(filter_name, ["opacity", "transparent"]):
        #     filt = Filter(name="opacity", value=filter_value)
        else:
            self._set_signal(
                status="error", text=f"Filter '{filter_name}' is not supported"
            )
            return

        if isinstance(target, FabricCollection):
            for obj in target.objects:
                obj.filters.append(filt)
        elif isinstance(target, FabricImageObject):
            target.filters.append(filt)

    @MethodProvider.provide
    def detect(self, image: Image, prompt: str) -> List[Object]:
        if not isinstance(image, FabricCanvas):
            self._set_signal(
                status="error",
                text="The 'image' argument in the 'detect' method must be an instance of 'Image'",
            )
            return

        if not isinstance(prompt, str):
            self._set_signal(
                status="error",
                text="The 'prompt' argument in the 'detect' method must be a string",
            )
            return

        detected_objects = [
            obj
            for obj in image.objects
            if self._compare_object_labels(prompt, list(obj.labelToScore.keys()))
        ]
        if len(detected_objects) != 0:
            return detected_objects

        parent_pil_image = image.backgroundImage.get_pil_image()
        scores, masks = self._toolkit.segment(parent_pil_image, prompt)
        for score, mask in zip(scores, masks):
            xmin, ymin, xmax, ymax = obj_box = ImageModule.fromarray(mask).getbbox()
            obj_width, obj_height = obj_size = xmax - xmin, ymax - ymin
            obj_pil_image = ImageModule.new("RGBA", obj_size)
            obj_pil_image.format = "PNG"
            obj_pil_image.paste(
                parent_pil_image.crop(obj_box),
                mask=ImageModule.fromarray(mask).crop(obj_box),
            )
            obj_data_url = pil_image_to_data_url(obj_pil_image)
            obj = FabricImageObject(
                parentId=image.id,
                type="image",
                left=xmin,
                top=ymin,
                width=obj_width,
                height=obj_height,
                labelToScore={prompt: score},
                src=obj_data_url,
            )
            image.objects.append(obj)
            detected_objects.append(obj)

        self._set_signal(
            status="warning",
            text=f"Detected {len(detected_objects)} '{prompt}' in the image",
        )
        return detected_objects

    def generate(
        self, prompt: str, category: Literal["object", "image"]
    ) -> Union[Object, Image]:
        pass

    @MethodProvider.provide
    def create_text(
        self,
        content: str,
        color: Optional[str] = None,
        font: Optional[str] = None,
        size: Optional[int] = None,
        style: Optional[str] = None,
        weight: Optional[str] = None,
    ) -> Text:
        return Text(
            content=content,
            color=color,
            font_family=font,
            font_size=size,
            font_style=style,
            font_weight=weight,
        )

    def _inpaint(
        self, parent: Union[FabricCanvas, FabricGroup], obj: FabricImageObject
    ) -> None:
        base_image = None
        if isinstance(parent, FabricCanvas):
            base_image = parent.backgroundImage.get_pil_image()
        elif isinstance(parent, FabricGroup):
            base_image = parent.objects[0].get_pil_image()

        width, height = base_image.size
        obj_image = obj.get_pil_image()
        obj_fit_mask = np.array(obj_image.convert("L"))
        obj_mask = np.zeros((height, width), dtype=np.uint8)
        xmin, ymin, xmax, ymax = obj.get_box()
        obj_mask[ymin:ymax, xmin:xmax] = np.where(obj_fit_mask != 0, 255, 0)
        inpainted_image = self._toolkit.inpaint(base_image, obj_mask)

        if isinstance(parent, FabricCanvas):
            parent.backgroundImage.src = pil_image_to_data_url(inpainted_image)
        elif isinstance(parent, FabricGroup):
            parent.objects[0].src = pil_image_to_data_url(inpainted_image)

    def _compare_filter_names(
        self, filter_name: str, check_filter_names: Iterable[str]
    ) -> bool:
        for check_name in check_filter_names:
            name1 = filter_name.lower()
            name2 = check_name.lower()
            if name1 in name2 or name2 in name1:
                return True

        return False

    def _compare_object_labels(self, label: str, check_labels: List[str]) -> bool:
        for check_label in check_labels:
            label1 = label.lower()
            label2 = check_label.lower()
            if label1 in label2 or label2 in label1:
                return True

        return False
