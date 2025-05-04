import threading
import logging
import time
from typing import Callable
from common.models import InferenceRequestContainerized
from peft import LoraConfig, get_peft_model
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM
)
from .llm_model_phi_4_base import LLMModelPhi4Base

logger = logging.getLogger(__name__)

class LLMModelPhi4WithAdapters(LLMModelPhi4Base):
    """
    Base class that supports adapter-based Phi-4 models (Mini or Multimodal).

    Attributes:
        _model_path (str):
            Path to the base model.

        _adapter_map (dict[str, str]):
            Dictionary mapping adapter names to their local paths or HF repo locations.

        _semaphores (dict[str, threading.Semaphore]):
            Semaphore used to control concurrent access per adapter.
    """

    _model_path: str
    _adapter_map: dict[str, str]
    _semaphores: dict[str, threading.Semaphore]

    def __init__(self,
                 model_path: str,
                 adapter_map: dict[str, str],
                 max_concurrent_per_adapter: int = 2):
        """
        Initialize the Phi-4 model with optional LoRA adapters.

        Args:
            model_path (str):
                Path to the base model (e.g., Hugging Face model ID).

            adapter_map (dict[str, str]):
                Mapping of adapter names to adapter paths.

            max_concurrent_per_adapter (int):
                Max number of concurrent accesses per adapter.

        Raises:
            RuntimeError: If 'base' is used as an adapter name.
        """
        if "base" in adapter_map:
            raise RuntimeError("'base' adapter name is reserved.")

        super().__init__()
        self._model_path = model_path
        self._adapter_map = adapter_map
        self._semaphores = {
            name: threading.Semaphore(max_concurrent_per_adapter)
            for name in ["base"] + list(adapter_map.keys())
        }

    def _load_common_model(self, loader_fn: Callable, use_cuda: bool = False) -> None:
        """
        Load the model and adapters into memory.

        Args:
            loader_fn (Callable):
                Loader like AutoTokenizer.from_pretrained or AutoProcessor.from_pretrained.

            use_cuda (bool):
                Whether to move the model to CUDA.
        """
        self._tokenizer_or_processor = loader_fn(self._model_path, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="auto" if not use_cuda else "cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2" if use_cuda else None
        )
        if use_cuda:
            base_model = base_model.cuda()

        noop_config = LoraConfig(
            r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            target_modules=[
                "self_attn.qkv_proj",
                "self_attn.o_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )

        self._model = get_peft_model(base_model, noop_config, adapter_name="base")

        for name, param in self._model.named_parameters():
            if "base" in name and 'lora' in name:
                param.data.zero_()

        logger.info("✅ Created base adapter into model %s: %s", type(self._model), self._model_path)

        for name, path in self._adapter_map.items():
            logger.info("🔧 Loading adapter '%s' into model %s: %s", name, type(self._model), self._model_path)
            self._model.load_adapter(path, adapter_name=name)
            logger.info("✅ Adapter '%s' loaded into model %s: %s", name, type(self._model), self._model_path)

        self._generation_config = GenerationConfig.from_pretrained(self._model_path)
        self._model.set_adapter("base")
        self._active_adapter = "base"

    def generate(self, request: InferenceRequestContainerized) -> str:
        """
        Generate a model response from a prompt request.

        Args:
            request (InferenceRequestContainerized):
                Input request payload.

        Returns:
            str: The model-generated response.
        """
        if self._model is None or self._tokenizer_or_processor is None:
            raise RuntimeError("Model must be loaded before calling generate().")

        prompt = self._build_prompt(request.messages)
        inputs = self._tokenize_input(prompt)
        input_token_count = inputs["input_ids"].shape[1]
        config = self._prepare_generation_config(request.parameters)

        with self._with_semaphore(request.adapter):
            self._set_adapter(request.adapter)
            start = time.perf_counter()
            output_ids = self._model.generate(**inputs, generation_config=config)
            duration = time.perf_counter() - start

        output_ids = output_ids[:, input_token_count:]
        output_token_count = output_ids.shape[1]
        response = self._decode_output(output_ids)
        self._print_token_stats(self._model_path, request.adapter, input_token_count, output_token_count, duration)
        return response
