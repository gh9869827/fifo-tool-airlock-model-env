
import logging
from fastapi import FastAPI, Response
import uvicorn
import torch
from dataclasses import dataclass
from typing import Optional, Union
from contextlib import asynccontextmanager
from .llm_model_phi_4_mini_instruct import LLMModelPhi4MiniInstruct
from .llm_model_phi_4_multimodal_instruct import LLMModelPhi4MultimodalInstruct
from common.models import InferenceRequestContainerized

# uncomment to enable repeatability
torch.random.manual_seed(0)

logger = logging.getLogger("uvicorn")

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
    logger.info("🚀 Starting up: loading models...")

    logger.info("📦 Loading 'phi-4-mini-instruct' model")
    models.phi4MiniInstruct = LLMModelPhi4MiniInstruct()
    models.phi4MiniInstruct.load_model()

    logger.info("📦 Loading 'phi-4-multimodal-instruct' model")
    models.phi4MultimodalInstruct = LLMModelPhi4MultimodalInstruct()
    models.phi4MultimodalInstruct.load_model()

    logger.info("✅ All models loaded successfully")
    yield
    logger.info("🛑 Shutting down gracefully")

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

        output = model.generate(request)

        return Response(content=output, media_type="text/plain")

    except Exception as e:
        return Response(
            content=str(e),
            status_code=500,
            media_type="text/plain"
        )

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="127.0.0.1", port=8000, log_level="info")
