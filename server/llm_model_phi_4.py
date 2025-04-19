from typing import List, Dict
from .llm_model import LLMModel
from common.models import GenerationParameters, Message


class LLMModelPhi4(LLMModel):

    def _build_prompt(self, messages: List[Message]) -> str:
        prompt_prefix = {
            "user": '<|user|>',
            "assistant": '<|assistant|>',
            "system": '<|system|>',
            "tool": '<|user|>',
        }
        prompt_suffix = '<|end|>'

        prompt = ""
        for message in messages:
            prompt += f"{prompt_prefix[message.role]}{message.content}{prompt_suffix}"
        prompt += prompt_prefix["assistant"]
        return prompt

    def _filter_generation_args(self, parameters: GenerationParameters, allowed_keys: set) -> Dict:
        return {
            k: v for k, v in parameters.model_dump(exclude_none=True).items()
            if k in allowed_keys
        }
