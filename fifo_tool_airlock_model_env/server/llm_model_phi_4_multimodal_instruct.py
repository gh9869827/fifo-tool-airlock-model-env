from typing import Tuple, Any
# Pylance: suppress missing type stub warning for transformers
from transformers import (  # type: ignore
    AutoProcessor,
    BatchEncoding,
    ProcessorMixin
)
from torch import Tensor
from PIL import Image
from fifo_tool_airlock_model_env.server.llm_model_phi_4_base_with_adapters import (
    LLMModelPhi4WithAdapters
)


class LLMModelPhi4MultimodalInstruct(LLMModelPhi4WithAdapters[ProcessorMixin]):
    """
    Wrapper for Phi-4-multimodal-instruct models using AutoProcessor for multimodal inputs.
    """

    def load_model(self) -> None:
        """
        Load the processor and model using AutoProcessor.
        """
        self._load_common_model(
        # Pylance: Type of from_pretrained() is partially unknown
            AutoProcessor.from_pretrained, # type: ignore[reportUnknownMemberType]
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

        # Pylance: ProcessorMixin doesn't declare __call__, but it's callable at runtime
        encoding = self._tokenizer_or_processor( # type: ignore[reportOptionalCall]
            text=prompt,
            images=images,
            return_tensors="pt"
        )

        # Pylance: Type of to() is unknown
        return encoding.to(  # type: ignore[reportUnknownMemberType]
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
            str: The decoded model response as a string.
        """
        # Guaranteed by `generate()`; safe to assume non-None.
        assert self._tokenizer_or_processor is not None

        # Pylance: Type of batch_decode() is unknown
        return self._tokenizer_or_processor.batch_decode(  # type: ignore[reportUnknownMemberType]
            tokens,
            skip_special_tokens=True
        )[0]

    def _get_generate_args(self) -> Tuple[list[str], dict[str, Any]]:
        """
        Specify which arguments to pass to the model's `generate()` method.

        This implementation returns:
        - A list of dynamic input keys extracted from `_tokenize_input()` output,
          covering both image and audio modalities.
        - A static kwargs dict with `num_logits_to_keep=0`. This is a necessary workaround:
          `prepare_inputs_for_generation()` in Phi-4 MM may default this value to `None`,
          but `forward()` assumes it is always an `int` (default 0). This mismatch can cause
          invalid operations like `-None` and crash the generation process.

        Returns:
            Tuple[list[str], dict[str, Any]]:
                - Dynamic input keys: includes `input_ids`, `attention_mask`, image and audio 
                  embeddings, attention masks, and `input_mode`.
                - Static kwargs: `{"num_logits_to_keep": 0}` ensures compatibility with 
                  downstream expectations in the model's `forward()` method.
        """
        return [
            "input_ids",
            "attention_mask",
            "input_image_embeds",
            "image_attention_mask",
            "image_sizes",
            "input_audio_embeds",
            "audio_attention_mask",
            "audio_embed_sizes",
            "input_mode"
        ], {
            "num_logits_to_keep": 0
        }
