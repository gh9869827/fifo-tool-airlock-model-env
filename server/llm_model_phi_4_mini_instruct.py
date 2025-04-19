from transformers import (
    AutoTokenizer
)
from .llm_model_phi_4_base_with_adapters import LLMModelPhi4WithAdapters


class LLMModelPhi4MiniInstruct(LLMModelPhi4WithAdapters):
    """
    Wrapper for Phi-4-mini-instruct models using text-only input.
    """

    def load_model(self) -> None:
        """
        Load the tokenizer and model using AutoTokenizer.
        """
        self._load_common_model(AutoTokenizer.from_pretrained, use_cuda=True)

    def _tokenize_input(self, prompt: str):
        """
        Tokenize the given prompt string into input tensor format for the model.

        Args:
            prompt (str):
                The prompt string to tokenize.

        Returns:
            dict: A dictionary of tensors suitable for input into the model.
        """
        return self._tokenizer_or_processor(prompt, return_tensors="pt").to(self._model.device)

    def _decode_output(self, tokens) -> str:
        """
        Decode the generated tokens back into a human-readable string.

        Args:
            tokens:
                The token sequence output by the model.

        Returns:
            str: A string representing the decoded model response.
        """
        return self._tokenizer_or_processor.decode(tokens[0], skip_special_tokens=True)
