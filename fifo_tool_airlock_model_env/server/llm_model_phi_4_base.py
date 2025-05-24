from abc import abstractmethod
import logging
import threading
from torch import Tensor
from PIL import Image
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer
from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.server.llm_model import LLMModel

logger = logging.getLogger(__name__)


class LLMModelPhi4Base(LLMModel):
    """
    Base class for Phi-4 model wrappers with common logic for prompt handling,
    tokenization, decoding, adapter switching, and concurrency controls.

    Attributes:
        _model_path (str | None):
            Path to the model or HF hub ID

        _adapter_map (dict[str, str]):
            Dictionary mapping adapter names to their local paths or HF repo locations

        _semaphores (dict[str, threading.Semaphore]):
            Dictionary of semaphores (one per adapter) to limit concurrent access

        _lock (threading.Lock):
            Ensures thread-safe switching of adapters

        _active_adapter (str):
            Name of the currently selected adapter

        _model (PreTrainedModel | None):
            The actual loaded model instance (base or PEFT-wrapped)

        _tokenizer_or_processor (PreTrainedTokenizer | None):
            Tokenizer or processor object, used to encode/decode messages

        _generation_config (GenerationConfig | None):
            Default generation configuration to use unless overridden by a request
    """

    _model_path: str | None
    _adapter_map: dict[str, str]
    _semaphores: dict[str, threading.Semaphore]
    _lock: threading.Lock
    _active_adapter: str
    _model: PreTrainedModel | None
    _tokenizer_or_processor: PreTrainedTokenizer | None
    _generation_config: GenerationConfig | None

    def __init__(self):
        """
        Constructor.
        """
        self._model_path = None
        self._adapter_map = {}
        self._semaphores = {}
        self._lock = threading.Lock()
        self._active_adapter = "base"
        self._model = None
        self._tokenizer_or_processor = None
        self._generation_config = None

    def _build_prompt(self, messages: list[Message]) -> str:
        """
        Construct a prompt string from a list of structured messages.

        Args:
            messages (list[Message]):
                A list of Message objects containing role and content fields.

        Returns:
            str: A formatted string representing the full conversation prompt.
        """
        prompt = ""
        for message in messages:
            prefix = {
                "user": "<|user|>",
                "assistant": "<|assistant|>",
                "system": "<|system|>",
                "tool": "<|user|>"
            }[message.role]
            prompt += f"{prefix}{message.content}<|end|>"
        return prompt + "<|assistant|>"

    def _prepare_generation_config(self, parameters: GenerationParameters) -> GenerationConfig:
        """
        Merge the default generation configuration with user-specified overrides.

        Args:
            parameters (GenerationParameters):
                Generation parameters provided in the request.

        Returns:
            GenerationConfig: A GenerationConfig instance with user-overrides applied.
        """
        config = GenerationConfig.from_dict(self._generation_config.to_dict())
        for k, v in parameters.model_dump(exclude_none=True).items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement this method.")

    def _set_adapter(self, adapter_name: str | None) -> None:
        """
        Set the active adapter for inference, switching context as needed.

        Args:
            adapter_name (str | None):
                The name of the adapter to activate, or None to disable all adapters 
                and use the base model.

        Raises:
            ValueError: If the specified adapter is unknown.
        """
        with self._lock:
            if adapter_name is None:
                name = "base"
            else:
                name = adapter_name
                if name not in self._adapter_map:
                    raise ValueError(f"Unknown adapter: {name}")
            if self._active_adapter != name:
                self._model.set_adapter(name)
                self._active_adapter = name

    def _with_semaphore(self, adapter: str):
        """
        Get the semaphore associated with the adapter to enforce concurrency limits.

        Args:
            adapter (str):
                The name of the adapter (or "base")

        Returns:
            threading.Semaphore: A semaphore associated with the adapter.

        Raises:
            ValueError: If no semaphore is configured for the given adapter.
        """
        semaphore = self._semaphores.get(adapter or "base")
        if semaphore is None:
            raise ValueError(f"No semaphore defined for adapter '{adapter}'")
        return semaphore
