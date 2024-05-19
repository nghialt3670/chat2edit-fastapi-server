from typing import Optional, Sequence

from openai import OpenAI


class OpenAILLM:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(
        self,
        messages: Sequence[str],
        system_message: Optional[str] = None,
        stop_word: Optional[str] = None,
    ) -> str:
        if len(messages) % 2 == 0:
            raise ValueError("Messages length should be odd")

        formated_messages = []
        if system_message is not None:
            formated_messages.append({"role": "system", "content": system_message})
        for user_message, llm_message in zip(messages[::2], messages[1::2]):
            formated_messages.append({"role": "user", "content": user_message})
            formated_messages.append({"role": "assistant", "content": llm_message})
        formated_messages.append({"role": "user", "content": messages[-1]})
        response = self.client.chat.completions.create(
            model=self.model, messages=formated_messages, stop=stop_word
        )
        return response.choices[0].message.content
