from abc import ABC
from functools import wraps
import inspect
from typing import Callable, Dict, Literal, Optional

from chat2edit.core.exec_signal import ExecSignal
from chat2edit.core.message import SysMessage


class MethodProvider(ABC):
    def __init__(self) -> None:
        self._exec_signal: ExecSignal = ExecSignal(status="info")

    def provide(method):
        method._provide = True
        return method

    def get_signal(self) -> ExecSignal:
        return self._exec_signal

    def clear_signal(self) -> None:
        self._exec_signal = ExecSignal(status="info")

    def _set_signal(
        self,
        status: Literal["info", "warning", "error"],
        text: str,
        sys_message: Optional[SysMessage] = None,
    ) -> None:
        self._exec_signal.status = status
        self._exec_signal.text = text
        self._exec_signal.sys_message = sys_message

    def get_bound_methods_dict(self) -> Dict[str, Callable]:
        methods_dict = inspect.getmembers(self, predicate=inspect.ismethod)
        public_methods = {
            name: method
            for name, method in methods_dict
            if getattr(method, "_provide", False)
        }
        bound_methods_dict = {
            name: method.__get__(self, MethodProvider)
            for name, method in public_methods.items()
        }
        return bound_methods_dict
