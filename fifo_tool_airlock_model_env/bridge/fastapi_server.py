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
    Calls an isolated, containerized model with the given messages and generation parameters.

    Args:
        request (InferenceRequestContainerized):
            messages (List[Message]): 
                A list of chat messages in the format [{'role': 'user', 'content': 'Hello!'}, ...].
            parameters (GenerationParameters):
                Configuration options for text generation such as temperature, top_k, top_p, 
                and others.

        container_name (str):
            The name of the Docker container running the model.

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
                parameters=request.parameters
            )
        )

        return Response(content=output, media_type="text/plain")

    except Exception as e:

        return Response(
            content=str(e),
            status_code=500,
            media_type="text/plain"
        )

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="127.0.0.1", port=8000, log_level="info")

# airlock_model_env>uvicorn bridge.fastapi_server:app --host 127.0.0.1 --port 8000
# pip install uvicorn fastapi
