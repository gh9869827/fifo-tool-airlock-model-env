import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from .llm_model_phi_4 import LLMModelPhi4
from common.models import InferenceRequestContainerized
from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
from typing import Dict

class LLMModelPhi4MiniInstruct(LLMModelPhi4):
    def __init__(self):
        self._model_path: str = "microsoft/Phi-4-mini-instruct"  # base model
        self._adapter_path: str = "/home/fifodev/tmp/checkpoint_dir"       # your LoRA weights
        self._model = None
        self._tokenizer = None
        self._generation_args: Dict = {}

    def load_model(self) -> None:
        # 1. Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )

        # 2. Load your fine-tuned LoRA weights
        self._model = PeftModel.from_pretrained(base_model, self._adapter_path)

        # 3. Load tokenizer — from the adapter dir in case it's been updated
        self._tokenizer = AutoTokenizer.from_pretrained(self._adapter_path, trust_remote_code=True)

        # 4. Generation defaults
        self._generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }

    def generate(self, request: InferenceRequestContainerized) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model must be loaded before calling generate()")

        prompt = self._build_prompt(request.messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_token_count = inputs["input_ids"].shape[1]

        allowed_keys = set(self._model.generation_config.to_dict().keys())
        filtered_params = self._filter_generation_args(request.parameters, allowed_keys)
        generation_args = {**self._generation_args, **filtered_params}

        start = time.perf_counter()
        outputs = self._model.generate(**inputs, **generation_args)
        duration = time.perf_counter() - start

        generated_ids = outputs[0][input_token_count:]
        output_token_count = generated_ids.shape[0]

        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        self._print_token_stats(input_token_count, output_token_count, duration)
        return response
