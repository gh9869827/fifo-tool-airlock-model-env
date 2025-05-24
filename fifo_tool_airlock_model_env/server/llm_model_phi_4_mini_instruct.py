from transformers import (
    AutoTokenizer
)
from torch import Tensor
from PIL import Image
from fifo_tool_airlock_model_env.server.llm_model_phi_4_base_with_adapters import (
    LLMModelPhi4WithAdapters
)


class LLMModelPhi4MiniInstruct(LLMModelPhi4WithAdapters):
    """
    Wrapper for Phi-4-mini-instruct models using text-only input.
    """

    def load_model(self) -> None:
        """
        Load the tokenizer and model using AutoTokenizer.
        """
        self._load_common_model(AutoTokenizer.from_pretrained, use_cuda=True)

    def _tokenize_input(self, prompt: str, images: list[Image.Image] | None = None):
        """
        Tokenize the given prompt string and optional images into input tensor format for the model.

        Args:
            prompt (str):
                The prompt string to tokenize.

            images (list[Image.Image] | None):
                Optional list of PIL Image objects for multimodal models.
                If None, only text will be tokenized. Ignored by text-only models.

        Returns:
            dict:
                A dictionary of tensors suitable for input into the model.
        """
        return self._tokenizer_or_processor(prompt, return_tensors="pt").to(self._model.device)

    def _decode_output(self, tokens: Tensor) -> str:
        """
        Decode the generated tokens back into a human-readable string.

        Args:
            tokens (Tensor):
                The token sequence output by the model (torch.LongTensor of shape 
                [batch_size, sequence_length]).

        Returns:
            str: The decoded model response as a string.
        """
        return self._tokenizer_or_processor.decode(tokens[0], skip_special_tokens=True)
