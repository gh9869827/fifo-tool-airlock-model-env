from pydantic import BaseModel
from enum import Enum


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"

class Model(str, Enum):
    Phi4MiniInstruct = "Phi4MiniInstruct"
    Phi4MultimodalInstruct = "Phi4MultimodalInstruct"

class GenerationParameters(BaseModel):
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool | None = None
    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

class Message(BaseModel):
    role: Role
    content: str

class InferenceRequest(BaseModel):
    model: Model
    adapter: str | None = None
    messages: list[Message]
    parameters: GenerationParameters | None = GenerationParameters()
    container_name: str

class InferenceRequestContainerized(BaseModel):
    model: Model
    adapter: str | None = None
    messages: list[Message]
    parameters: GenerationParameters | None = GenerationParameters()
