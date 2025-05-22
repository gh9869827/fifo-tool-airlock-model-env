from __future__ import annotations
import requests
from fifo_tool_airlock_model_env.common.models import (
    InferenceRequest,
    GenerationParameters,
    Model,
    Message
)

def call_airlock_model_server(container_name: str,
                              model: Model,
                              messages: list[Message],
                              adapter: str | None = None,
                              parameters: GenerationParameters | None = None,
                              host: str = "http://127.0.0.1:8000") -> str:
    """
    Sends messages to the FastAPI inference server and returns the generated response.
    """
    req = InferenceRequest(
        model=model,
        adapter=adapter,
        messages=messages,
        container_name=container_name,
        parameters=parameters or GenerationParameters()
    )
    url = f"{host}/generate"

    response = requests.post(url, json=req.model_dump(exclude_none=True), timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"FastAPI server error: {response.status_code} - {response.text}")

    return response.text.strip()
