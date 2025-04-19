from pydantic import BaseModel
from typing import List, Optional
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
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None

class Message(BaseModel):
    role: Role
    content: str

class InferenceRequest(BaseModel):
    model: Model
    messages: List[Message]
    parameters: Optional[GenerationParameters] = GenerationParameters()
    container_name: str

class InferenceRequestContainerized(BaseModel):
    model: Model
    messages: List[Message]
    parameters: Optional[GenerationParameters] = GenerationParameters()
