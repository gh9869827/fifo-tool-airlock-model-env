from typing import Tuple, Any
# Pylance: suppress missing type stub warning for transformers
from transformers import (  # type: ignore
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase
)
from torch import Tensor
from PIL import Image
from fifo_tool_airlock_model_env.server.llm_model_phi_4_base_with_adapters import (
    LLMModelPhi4WithAdapters
)


class LLMModelPhi4MiniInstruct(LLMModelPhi4WithAdapters[PreTrainedTokenizerBase]):
    """
    Wrapper for Phi-4-mini-instruct models using text-only input.
    """

    def load_model(self) -> None:
        """
        Load the tokenizer and model using AutoTokenizer.
        """
        # Pylance: Type of from_pretrained() is partially unknown
        self._load_common_model(
            AutoTokenizer.from_pretrained, # type: ignore[reportUnknownMemberType]
            use_cuda=True
        )

    def _tokenize_input(self,
                        prompt: str,
                        images: list[Image.Image] | None = None) -> BatchEncoding:
        """
        Tokenize the given prompt string and optional images into input tensor format for the model.

        Args:
            prompt (str):
                The prompt string to tokenize.

            images (list[Image.Image] | None):
                Optional list of PIL Image objects for multimodal models.
                If None, only text will be tokenized. Ignored by text-only models.

        Returns:
            BatchEncoding:
                A dictionary of tensors suitable for input into the model.
        """
        # Guaranteed by `generate()`; safe to assume non-None.
        assert self._model is not None
        assert self._tokenizer_or_processor is not None

        encoding = self._tokenizer_or_processor(
            prompt,
            return_tensors="pt"
        )

        return encoding.to(
            # Pylance: .device may not be defined on all model types
            # But we only use model classes (PreTrainedModel or PEFT-wrapped variants)
            # that always define .device, so this is safe in practice.
            self._model.device  # type: ignore[attr-defined]
        )

    def _decode_output(self, tokens: Tensor) -> str:
        """
        Decode the generated tokens back into a human-readable string.

        Args:
            tokens (Tensor):
                The token sequence output by the model (torch.LongTensor of shape 
                [batch_size, sequence_length]).

        Returns:
            str:
                The decoded model response as a string.
        """
        # Guaranteed by `generate()`; safe to assume non-None.
        assert self._tokenizer_or_processor is not None

        # Pylance: Type of decode() is partially unknown
        return self._tokenizer_or_processor.decode( # type: ignore[reportUnknownMemberType]
            tokens[0],
            skip_special_tokens=True
        )

    def _get_generate_args(self) -> Tuple[list[str], dict[str, Any]]:
        """
        Specify which arguments to pass to the model's `generate()` method.

        This implementation returns:
        - A list of dynamic input keys extracted from `_tokenize_input()` output
        - An empty dict of static kwargs, as no fixed-generation settings are required

        Returns:
            Tuple[list[str], dict[str, Any]]: 
                - Dynamic input keys. For Phi-4-mini-instruct (text-only), only
                  `input_ids` and `attention_mask` are used.
                - Empty dict of static kwargs, since the model behavior is controlled
                  entirely via input and generation config.
        """
        return [
            "input_ids",
            "attention_mask"
        ], {}
