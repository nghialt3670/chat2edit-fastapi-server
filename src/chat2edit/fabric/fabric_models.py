from typing import Dict, List, Tuple, Union
from uuid import uuid4
from altair import Optional
from pydantic import BaseModel, Field
from PIL import Image

from chat2edit.core.message import Attachment
from chat2edit.utils.image import data_url_to_pil_image


def create_id() -> str:
    return str(uuid4())


class FabricObject(BaseModel):
    id: str = Field(default_factory=create_id)
    type: str
    angle: float = 0.0
    flipX: bool = False
    flipY: bool = False
    scaleX: float = 1.0
    scaleY: float = 1.0
    left: int = 0
    top: int = 0
    opacity: float = 1.0
    fill: Optional[str] = None
    shadow: Optional[str] = None
    stroke: Optional[str] = None
    strokeWidth: int = 0
    width: float
    height: float

    def __hash__(self) -> int:
        return hash(self.id)


class FabricCollection(BaseModel):
    objects: List[Union["FabricGroup", "FabricUploadedImage", "FabricImageObject"]]

    def contains(self, obj: FabricObject) -> bool:
        return obj in self.objects

    def add(self, obj: FabricObject) -> None:
        self.objects.append(obj)

    def remove(self, obj: FabricObject) -> None:
        self.objects.remove(obj)


class FabricImage(FabricObject):
    cropX: Optional[int] = None
    cropY: Optional[int] = None
    filters: List[Dict] = Field(default_factory=list)
    src: str

    def get_pil_image(self) -> Image.Image:
        return data_url_to_pil_image(self.src)


class FabricUploadedImage(FabricImage):
    filename: str


class FabricImageObject(FabricImage):
    labelToScore: Dict[str, float]
    inpainted: bool = False

    def get_box(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height


class FabricTextbox(FabricObject):
    fontFamily: str
    fontSize: str
    fontStyle: str
    fontWeight: str
    text: str


class FabricGroup(FabricObject, FabricCollection):
    pass


class FabricCanvas(FabricCollection, Attachment):
    id: str
    backgroundImage: FabricUploadedImage

    def get_type(self) -> str:
        return "image"
