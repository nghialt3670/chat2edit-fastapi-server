from typing import List, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import redis
import pickle
from chat2edit.chat2edit import Chat2Edit
from chat2edit.core.chat_state import ChatState
from chat2edit.core.message import UserMessage
from chat2edit.fabric.fabric_method_provider import FabricMethodProvider
from chat2edit.fabric.fabric_models import FabricCanvas
from chat2edit.core.open_ai_llm import OpenAILLM
from chat2edit.tools.grounded_sam import GroundedSAM
from chat2edit.tools.lama_inpainter import LaMaInpainter
from chat2edit.tools.toolkit import Toolkit

with open("src/config/my_config.yaml", "r") as f:
    config = yaml.safe_load(f)


app = FastAPI()
rd = redis.Redis()


frontend_origin = config["frontend"]["origin"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

grounded_sam = GroundedSAM(
    gdino_checkpoint=config["tools"]["groundingdino"]["checkpoint"],
    gdino_config=config["tools"]["groundingdino"]["config"],
    gdino_device=config["tools"]["groundingdino"]["device"],
    sam_checkpoint=config["tools"]["sam"]["checkpoint"],
    sam_model_type=config["tools"]["sam"]["model_type"],
    sam_device=config["tools"]["sam"]["device"],
)

lama_inpainter = LaMaInpainter(
    checkpoint=config["tools"]["lama"]["checkpoint"],
    device=config["tools"]["lama"]["device"],
)

toolkit = Toolkit(segmenter=grounded_sam, inpainter=lama_inpainter)
method_provider = FabricMethodProvider(toolkit=toolkit)
llm = OpenAILLM(api_key=config["openai"]["api_key"], model=config["openai"]["model"])
chat2edit = Chat2Edit(
    method_provider=method_provider,
    api_key=config["openai"]["api_key"],
    model=config["openai"]["model"],
    prompt_limit=3,
)


class EditingRequest(BaseModel):
    chat_id: str
    instruction: str
    canvases: List[FabricCanvas]


class EditingResponse(BaseModel):
    response: str
    status: Literal["success", "fail"]
    canvases: List[FabricCanvas]


@app.post("/edit")
def edit(request: EditingRequest) -> EditingResponse:
    pickled_chat_state = rd.get(request.chat_id)
    chat_state = None
    if not pickled_chat_state:
        chat_state = ChatState()
    else:
        chat_state = pickle.loads(pickled_chat_state)
    user_message = UserMessage(
        chat_id=request.chat_id, text=request.instruction, attachments=request.canvases
    )
    sys_message = chat2edit(chat_state, user_message)

    pickled_chat_state = pickle.dumps(chat_state)
    rd.set(request.chat_id, pickled_chat_state)

    return EditingResponse(
        response=sys_message.text,
        status=sys_message.status,
        canvases=sys_message.attachments,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
