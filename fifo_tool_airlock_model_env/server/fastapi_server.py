from .logging_config import configure_logging

configure_logging()

import os
from fastapi import FastAPI, Response
import uvicorn
# import torch
import logging
from contextlib import asynccontextmanager
from fifo_tool_airlock_model_env.server.llm_model_loader import (
    LoadedModels
)
from fifo_tool_airlock_model_env.common.models import (
    InferenceRequestContainerized
)

# uncomment to enable repeatability
# torch.random.manual_seed(0)

models = LoadedModels()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.info("🚀 Starting up: loading models...")

    try:
        models.load_from_config("model_config.json")
        logging.info("✅ All models loaded successfully")
    except Exception as e:
        logging.error("❌ Failed to load models: %s", e)
        raise

    yield

    logging.info("🛑 Shutting down gracefully")

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: InferenceRequestContainerized):
    try:
        model = models.get_model(request.model)
        if model is None:
            return Response(
                status_code=400,
                content=f"Unrecognized model '{request.model}'"
            )

        try:
            output = model.generate(request)
        except Exception as e:
            print(e)
            raise e

        return Response(content=output, media_type="text/plain")

    except Exception:
        logging.exception("❌ Unhandled error in /generate endpoint")
        return Response(
            content="Internal server error.",
            status_code=500,
            media_type="text/plain"
        )

if __name__ == "__main__":

    # Ensure all Hugging Face operations run in offline mode
    # since we are in the airlock environment
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    uvicorn.run("fifo_tool_airlock_model_env.server.fastapi_server:app",
                host="127.0.0.1",
                port=8000,
                log_level="info")
