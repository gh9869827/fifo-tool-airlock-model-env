import time

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig
)
from .llm_model_phi_4 import LLMModelPhi4
from common.models import InferenceRequestContainerized


class LLMModelPhi4MultimodalInstruct(LLMModelPhi4):
    def __init__(self):
        self._model_path: str = "microsoft/Phi-4-multimodal-instruct"
        self._processor = None
        self._model = None
        self._generation_config = None

    def load_model(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).cuda()

        self._generation_config = GenerationConfig.from_pretrained(self._model_path)

    def generate(self, request: InferenceRequestContainerized) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model must be loaded before calling generate()")

        prompt = self._build_prompt(request.messages)
        inputs = self._processor(text=prompt, return_tensors='pt').to('cuda:0')
        input_token_count = inputs['input_ids'].shape[1]

        gen_config = GenerationConfig.from_dict(self._generation_config.to_dict())
        filtered_params = self._filter_generation_args(request.parameters, gen_config.to_dict().keys())
        for k, v in filtered_params.items():
            setattr(gen_config, k, v)

        start = time.perf_counter()
        generate_ids = self._model.generate(
            **inputs,
            num_logits_to_keep=0,
            generation_config=gen_config,
        )
        duration = time.perf_counter() - start

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        output_token_count = generate_ids.shape[1]

        response = self._processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        self._print_token_stats(input_token_count, output_token_count, duration)
        return response
