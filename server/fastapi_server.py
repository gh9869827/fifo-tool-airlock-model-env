from .logging_config import configure_logging

configure_logging()

from fastapi import FastAPI, Response
import uvicorn
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Union
from contextlib import asynccontextmanager
from .llm_model_phi_4_mini_instruct import LLMModelPhi4MiniInstruct
from .llm_model_phi_4_multimodal_instruct import LLMModelPhi4MultimodalInstruct
from common.models import InferenceRequestContainerized

# uncomment to enable repeatability
torch.random.manual_seed(0)


@dataclass
class LoadedModels:

    phi4MiniInstruct: Optional[LLMModelPhi4MiniInstruct] = None
    phi4MultimodalInstruct: Optional[LLMModelPhi4MultimodalInstruct] = None

    def get_model(self, name: str) -> Optional[Union[LLMModelPhi4MiniInstruct,
                                                     LLMModelPhi4MultimodalInstruct]]:
        if name == "Phi4MiniInstruct":
            return self.phi4MiniInstruct

        if name == "Phi4MultimodalInstruct":
            return self.phi4MultimodalInstruct

        return None


models = LoadedModels()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("🚀 Starting up: loading models...")

    logging.info("📦 Loading 'phi-4-mini-instruct' model")
    models.phi4MiniInstruct = LLMModelPhi4MiniInstruct(
        model_path="microsoft/Phi-4-mini-instruct",
        adapter_map={
            "mini-date-dsl": "/home/fifodev/tmp/checkpoint_dir"
        },
        max_concurrent_per_adapter=1
    )
    models.phi4MiniInstruct.load_model()

    logging.info("📦 Loading 'phi-4-multimodal-instruct' model")
    models.phi4MultimodalInstruct = LLMModelPhi4MultimodalInstruct(
        model_path="microsoft/Phi-4-multimodal-instruct",
        adapter_map={},
        max_concurrent_per_adapter=1
    )
    models.phi4MultimodalInstruct.load_model()

    logging.info("✅ All models loaded successfully")
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

    except Exception as e:
        return Response(
            content=str(e),
            status_code=500,
            media_type="text/plain"
        )

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="127.0.0.1", port=8000, log_level="info")
