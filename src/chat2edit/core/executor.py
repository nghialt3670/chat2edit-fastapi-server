import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional

from altair import value

from chat2edit.core.exec_signal import ExecSignal
from chat2edit.core.message import Attachment, ExecMessage, SysMessage
from chat2edit.core.method_provider import MethodProvider


class Executor:
    def __init__(self, method_provider: MethodProvider) -> None:
        self._method_provider = method_provider
        self._exec_context = method_provider.get_bound_methods_dict()

    def get_methods(self) -> List[Callable]:
        return [obj for _, obj in self._exec_context.items() if inspect.ismethod(obj)]

    def __call__(self, commands: Iterable[str], context: Dict[str, Any]) -> ExecMessage:
        curr_context = self._exec_context.copy()
        curr_context.update(context)
        curr_signal = curr_command = None
        for command in commands:
            curr_command = command
            try:
                exec(curr_command, locals(), curr_context)
                curr_signal = self._method_provider.get_signal()
                if curr_signal.status != "info":
                    break
            except Exception as e:
                curr_signal = ExecSignal(status="error", text=str(e))
                break

        curr_context = {
            name: value
            for name, value in curr_context.items()
            if name not in self._exec_context
        }
        exec_message = ExecMessage(
            status=curr_signal.status,
            text=curr_signal.text,
            command=curr_command,
            context=curr_context,
            sys_message=curr_signal.sys_message,
        )
        self._method_provider.clear_signal()
        return exec_message
