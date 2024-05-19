from dataclasses import dataclass
from typing import Literal, Optional

from chat2edit.core.message import SysMessage


@dataclass
class ExecSignal:
    status: Literal["info", "warning", "error"]
    text: str = ""
    sys_message: Optional[SysMessage] = None
