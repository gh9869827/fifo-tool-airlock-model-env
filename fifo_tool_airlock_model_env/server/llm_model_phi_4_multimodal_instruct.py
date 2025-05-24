from transformers import (
    AutoProcessor
)
from torch import Tensor
from PIL import Image
from fifo_tool_airlock_model_env.server.llm_model_phi_4_base_with_adapters import (
    LLMModelPhi4WithAdapters
)


class LLMModelPhi4MultimodalInstruct(LLMModelPhi4WithAdapters):
    """
    Wrapper for Phi-4-multimodal-instruct models using AutoProcessor for multimodal inputs.
    """

    def load_model(self) -> None:
        """
        Load the processor and model using AutoProcessor.
        """
        self._load_common_model(AutoProcessor.from_pretrained, use_cuda=True)

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
        return self._tokenizer_or_processor(
            text=prompt, images=images, return_tensors="pt"
        ).to("cuda")

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
        return self._tokenizer_or_processor.batch_decode(tokens, skip_special_tokens=True)[0]
