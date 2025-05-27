import logging
import subprocess
from fastapi import FastAPI, Response
import uvicorn
from fifo_tool_airlock_model_env.common.models import (
    InferenceRequest,
    InferenceRequestContainerized
)

app = FastAPI()

def call_model(container_name: str, request: InferenceRequestContainerized) -> str:
    """
    Calls an isolated, containerized model with the given messages, images, and generation
    parameters.

    Args:
        container_name (str):
            The name of the Docker container running the model.

        request (InferenceRequestContainerized):
            model (Model):
                Which model to use.
            adapter (str | None):
                Optional adapter/fine-tune identifier.
            messages (List[Message]):
                Input message sequence.
            images (List[str] | None):
                Optional list of base64-encoded images for multimodal models.
                Each item must be a base64-encoded image string (no URLs or paths).
                Ignored by text-only models.
            parameters (GenerationParameters | None):
                Optional generation config.

    Returns:
        str: The generated model response.

    Raises:
        RuntimeError: If the docker command fails.
    """
    try:

        result = subprocess.run(
            ["docker", "exec", "-i", container_name,
             "python3", "-m", "fifo_tool_airlock_model_env.client.run"],
            input=request.model_dump_json(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            check=True
        )

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stdout) from e

@app.post("/generate")
async def generate(request: InferenceRequest):
    try:

        output = call_model(
            container_name=request.container_name,
            request=InferenceRequestContainerized(
                model=request.model,
                adapter=request.adapter,
                messages=request.messages,
                parameters=request.parameters,
                images=request.images
            )
        )

        return Response(content=output, media_type="text/plain")

    except Exception:
        logging.exception("Unhandled error in /generate endpoint")
        return Response(
            content="Internal server error.",
            status_code=500,
            media_type="text/plain"
        )

if __name__ == "__main__":
    uvicorn.run("fifo_tool_airlock_model_env.bridge.fastapi_server:app",
                host="127.0.0.1",
                port=8000,
                log_level="info")
