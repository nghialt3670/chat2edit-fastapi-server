from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


class Attachment(ABC):
    @abstractmethod
    def get_type(self) -> str:
        pass


@dataclass
class UserMessage:
    chat_id: str
    text: str = ""
    attachments: List[Attachment] = field(default_factory=list)


@dataclass
class SysMessage:
    status: Literal["success", "fail"]
    text: str = ""
    attachments: List[Attachment] = field(default_factory=list)


@dataclass
class ExecMessage:
    status: Literal["success", "warning", "error"]
    text: str
    command: str
    context: Dict[str, Any]
    sys_message: Optional[SysMessage] = None
