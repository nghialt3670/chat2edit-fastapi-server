from abc import ABC, abstractmethod
from typing import List

from chat2edit.core.chat_state import ChatState
from chat2edit.core.executor import Executor
from chat2edit.core.message import ExecMessage, SysMessage, UserMessage
from chat2edit.core.method_provider import MethodProvider
from chat2edit.core.open_ai_llm import OpenAILLM


SYS_FAIL_MESSAGE = SysMessage(status="fail")


class SelfPrompter(ABC):
    def __init__(
        self,
        method_provider: MethodProvider,
        api_key: str,
        model: str,
        prompt_limit: int,
    ) -> None:
        self._executor = Executor(method_provider)
        self._llm = OpenAILLM(api_key, model)
        self._prompt_limit = prompt_limit

    @abstractmethod
    def _extract_commands(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def _update_chat_state_from_user_message(
        self, chat_state: ChatState, message: UserMessage
    ) -> ChatState:
        pass

    @abstractmethod
    def _update_chat_state_from_exec_message(
        self, chat_state: ChatState, message: ExecMessage
    ) -> ChatState:
        pass

    def __call__(self, chat_state: ChatState, message: UserMessage) -> SysMessage:
        chat_state = self._update_chat_state_from_user_message(chat_state, message)
        prompt_count = 0
        response = None
        while prompt_count < self._prompt_limit:
            try:
                response = self._llm([chat_state.curr_prompt])
                print(
                    "-----------------------------------------------------------------------------------------------------------------"
                )
                print("### Prompt:")
                print(chat_state.curr_prompt)
                print()
                print("### Response:")
                print(response)
                print(
                    "-----------------------------------------------------------------------------------------------------------------"
                )

                prompt_count += 1
            except Exception as e:
                chat_state.curr_prompt = ""
                return SYS_FAIL_MESSAGE

            commands = self._extract_commands(response)
            if not commands:
                chat_state.curr_prompt = ""
                return SYS_FAIL_MESSAGE

            exec_message = self._executor(commands, chat_state.context)
            chat_state.curr_response = response
            chat_state = self._update_chat_state_from_exec_message(
                chat_state, exec_message
            )

            if exec_message.sys_message:
                return exec_message.sys_message

        return SYS_FAIL_MESSAGE
