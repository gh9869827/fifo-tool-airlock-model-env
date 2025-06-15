from __future__ import annotations
import requests
from fifo_tool_airlock_model_env.common.models import (
    InferenceRequest,
    GenerationParameters,
    Model,
    Message
)

def call_airlock_model_server(
    container_name: str,
    model: Model,
    messages: list[Message],
    images: list[str] | None = None,
    adapter: str | None = None,
    parameters: GenerationParameters | None = None,
    host: str = "http://127.0.0.1:8000"
) -> str:
    """
    Sends an inference request to the FastAPI airlock server and returns the generated response.

    Args:
        container_name (str):
            Name of the airlock container serving the model.

        model (Model):
            Target model for inference.

        messages (list[Message]):
            Sequence of input messages.

        images (list[str] | None):
            Optional list of base64-encoded images for multimodal models.
            Ignored for text-only models.

        adapter (str | None):
            Optional adapter or fine-tune identifier.

        parameters (GenerationParameters | None):
            Optional generation configuration.

        host (str):
            Server URL.

    Returns:
        str:
            The generated response from the model server.

    Raises:
        RuntimeError: If the server returns a non-200 status code.
    """
    req = InferenceRequest(
        model=model,
        adapter=adapter,
        messages=messages,
        images=images,
        container_name=container_name,
        parameters=parameters or GenerationParameters()
    )
    url = f"{host}/generate"

    response = requests.post(url, json=req.model_dump(exclude_none=True), timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"FastAPI server error: {response.status_code} - {response.text}")

    return response.text.strip()
