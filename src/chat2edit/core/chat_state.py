from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ChatState:
    context: Dict[str, Any] = field(default_factory=dict)
    variable_count: Dict[str, int] = field(default_factory=dict)
    curr_prompt: str = ""
    curr_response: str = ""
