import inspect
import re
from typing import Any, Dict, List


from chat2edit.core.chat_state import ChatState
from chat2edit.core.message import ExecMessage, UserMessage
from chat2edit.core.method_provider import MethodProvider
from chat2edit.core.self_prompter import SelfPrompter
from chat2edit.fabric.fabric_prompt import VI_PROMPT_TEMPLATE


ACTION_EXTRACT_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)
OBSERVATION_PATTERN = "<observation>\n    {observation}\n</observation>\n"


class Chat2Edit(SelfPrompter):
    def __init__(
        self,
        method_provider: MethodProvider,
        api_key: str,
        model: str,
        prompt_limit: int,
    ) -> None:
        super().__init__(method_provider, api_key, model, prompt_limit)
        self._base_prompt = self._create_base_prompt(VI_PROMPT_TEMPLATE)

    def _create_base_prompt(self, prompt_template: str) -> List[str]:
        declarations = [
            f"def {method.__name__}{inspect.signature(method)}".replace("~", "")
            for method in self._executor.get_methods()
        ]
        return prompt_template.format(methods="\n".join(declarations))

    def _update_chat_state_from_user_message(
        self, chat_state: ChatState, message: UserMessage
    ) -> ChatState:
        message_context = {}
        new_prompt = None
        for attachment in message.attachments:
            var_count = chat_state.variable_count.get(attachment.get_type(), 0)
            var_name = attachment.get_type() + str(var_count)
            message_context[var_name] = attachment

        if chat_state.curr_prompt == "":
            new_prompt = self._base_prompt
        else:
            new_prompt = chat_state.curr_prompt[:-3]

        observation = f"user_message(text='{message.text}', images=[{', '.join(message_context.keys())}])"
        new_prompt += OBSERVATION_PATTERN.format(observation=observation)
        chat_state.context.update(message_context)
        chat_state.curr_prompt = new_prompt + "..."
        return chat_state

    def _update_chat_state_from_exec_message(
        self, chat_state: ChatState, message: ExecMessage
    ) -> ChatState:
        new_prompt = chat_state.curr_prompt[:-3]
        stop_position = chat_state.curr_response.find(message.command) + len(
            message.command
        )
        new_prompt += chat_state.curr_response[:stop_position] + "\n</action>\n"

        if not message.sys_message:
            observation = f"sys_{message.status}('{message.text}')"
            new_prompt += OBSERVATION_PATTERN.format(observation=observation)

        chat_state.context.update(message.context)
        chat_state.curr_prompt = new_prompt + "..."

        return chat_state

    def _extract_commands(self, llm_response: str) -> List[str]:
        matches = re.findall(ACTION_EXTRACT_PATTERN, llm_response)
        commands = []
        for match in matches:
            commands.extend(command.strip() for command in match.split("\n"))

        commands = [command for command in commands if command != ""]
        return commands
