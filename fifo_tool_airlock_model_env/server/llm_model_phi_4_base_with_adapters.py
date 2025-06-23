from abc import abstractmethod
import threading
import logging
import time
import base64
import io
from typing import Any, Callable, Generic, Tuple, TypeVar, cast
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel, PeftMixedModel
import torch
# Pylance: suppress missing type stub warning for transformers
from transformers import (  # type: ignore
    GenerationConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    PreTrainedModel,
    BatchEncoding
)
from torch import Tensor
from fifo_tool_airlock_model_env.common.models import InferenceRequestContainerized
from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message
from fifo_tool_airlock_model_env.server.llm_model import LLMModel


# NOTE: Pillow's type stubs are incomplete (especially Image and ImageDraw).
#       Pylance strict mode requires `# type: ignore` on certain lines below.

logger = logging.getLogger(__name__)

def decode_base64_image(b64: str) -> Image.Image:
    """
    Decodes a base64-encoded image string into a PIL Image object in RGB mode.

    Args:
        b64 (str):
            The base64-encoded string representing the image.

    Returns:
        Image.Image:
            The decoded image as a PIL Image object in RGB mode.

    Raises:
        ValueError:
            If the input string is not a valid base64-encoded image.

        IOError:
            If the decoded bytes cannot be opened as an image.
    """
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB") # type: ignore

TTokenizerOrProcessor = TypeVar(
    "TTokenizerOrProcessor",
    bound=PreTrainedTokenizerBase | ProcessorMixin
)

class LLMModelPhi4WithAdapters(LLMModel, Generic[TTokenizerOrProcessor]):
    """
    Base class that supports adapter-based Phi-4 models (Mini or Multimodal).

    Attributes:

        _model_path (str):
            Path to the model or HF hub ID

        _model (PreTrainedModel | None):
            The actual loaded model instance (base or PEFT-wrapped)

        _tokenizer_or_processor (PreTrainedTokenizerBase | ProcessorMixin | None):
            Tokenizer or processor object, used to encode/decode messages

        _generation_config (GenerationConfig | None):
            Default generation configuration to use unless overridden by a request

        _adapter_map (dict[str, str]):
            Dictionary mapping adapter names to their local paths or HF repo locations.

        _active_adapter (str):
            Name of the currently selected adapter

        _semaphores (dict[str, threading.Semaphore]):
            Dictionary of semaphores (one per adapter) to limit concurrent access

        _lock (threading.Lock):
            Ensures thread-safe switching of adapters
    """

    _model_path: str
    _model: PreTrainedModel | PeftModel | PeftMixedModel | None
    _tokenizer_or_processor: TTokenizerOrProcessor | None
    _generation_config: GenerationConfig | None
    _adapter_map: dict[str, str]
    _active_adapter: str
    _semaphores: dict[str, threading.Semaphore]
    _lock: threading.Lock

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
        self._model = None
        self._tokenizer_or_processor = None
        self._generation_config = None
        self._adapter_map = dict(adapter_map)
        self._active_adapter = "base"
        self._semaphores = {
            name: threading.Semaphore(max_concurrent_per_adapter)
            for name in ["base"] + list(adapter_map.keys())
        }
        self._lock = threading.Lock()

    def _load_common_model(self,
                           loader_fn: Callable[..., TTokenizerOrProcessor],
                           use_cuda: bool = False) -> None:
        """
        Load the model and adapters into memory.

        Args:
            loader_fn (Callable[..., TTokenizerOrProcessor]):
                Loader like AutoTokenizer.from_pretrained or AutoProcessor.from_pretrained.

            use_cuda (bool):
                Whether to move the model to CUDA.
        """
        self._tokenizer_or_processor = loader_fn(
            self._model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # Cast is needed because return type of from_pretrained() is not declared statically
        base_model = cast(
            PreTrainedModel,
            # Pylance: Type of from_pretrained() is partially unknown
            AutoModelForCausalLM.from_pretrained( # type: ignore[reportUnknownMemberType]
                self._model_path,
                device_map="auto" if not use_cuda else "cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                local_files_only=True,
                _attn_implementation="flash_attention_2" if use_cuda else None
            )
        )

        if use_cuda:
            # Cast is needed because return type of `.cuda()` may be partially unknown due to
            # model union
            base_model = cast(
                PreTrainedModel,
                # Pylance: Type of cuda() is partially unknown
                base_model.cuda() # type: ignore[reportUnknownMemberType]
            )

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
            logger.info("🔧 Loading adapter '%s' into model %s: %s",
                        name, type(self._model), self._model_path)
            # Pylance: Type of load_adapter() is partially unknown
            self._model.load_adapter( # type: ignore[reportUnknownMemberType]
                path,
                adapter_name=name
            )
            logger.info("✅ Adapter '%s' loaded into model %s: %s",
                        name, type(self._model), self._model_path)

        # Pylance: Type of from_pretrained() is partially unknown
        self._generation_config = \
            GenerationConfig.from_pretrained( # type: ignore[reportUnknownMemberType]
                self._model_path
            )
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
        images = [decode_base64_image(img) for img in request.images] if request.images else None

        inputs = self._tokenize_input(prompt, images)
        input_token_count = cast(torch.Tensor, inputs["input_ids"]).shape[1]
        config = self._prepare_generation_config(request.parameters)

        with self._with_semaphore(request.adapter):
            self._set_adapter(request.adapter)
            start = time.perf_counter()

            dynamic_keys, static_kwargs = self._get_generate_args()

            # Extract only the allowed dynamic keys from inputs
            generate_kwargs : dict[str, Any] = {
                k: inputs[k] for k in dynamic_keys if k in inputs
            }

            # Merge static kwargs like num_logits_to_keep or others
            generate_kwargs.update(static_kwargs)

            # Always include the generation config
            generate_kwargs["generation_config"] = config

            # Cast is needed because return type of `.generate()` can't be inferred; we expect
            # torch.Tensor (dtype=torch.long)
            output_ids = cast(
                Tensor,
                # Pylance: Type of generate() is partially unknown
                self._model.generate(  # type: ignore[reportUnknownMemberType]
                    **generate_kwargs
                )
            )

            duration = time.perf_counter() - start

        output_ids = output_ids[:, input_token_count:]
        output_token_count = output_ids.shape[1]
        response = self._decode_output(output_ids)
        self._print_token_stats(
            self._model_path, request.adapter, input_token_count, output_token_count, duration
        )
        return response

    def _set_adapter(self, adapter_name: str | None) -> None:
        """
        Set the active adapter for inference, switching context as needed.

        Args:
            adapter_name (str | None):
                The name of the adapter to activate, or None to disable all adapters 
                and use the base model.

        Raises:
            ValueError: If the specified adapter is unknown.
            RuntimeError: Model must be loaded before calling generate().
        """
        if self._model is None:
            raise RuntimeError("Model must be loaded before calling generate().")

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

    def _with_semaphore(self, adapter: str | None):
        """
        Get the semaphore associated with the adapter to enforce concurrency limits.

        Args:
            adapter (str | None):
                The name of the adapter to use. If None, defaults to the base model's semaphore.

        Returns:
            threading.Semaphore:
                A semaphore associated with the adapter.

        Raises:
            ValueError: If no semaphore is configured for the given adapter.
        """
        semaphore = self._semaphores.get(adapter or "base")
        if semaphore is None:
            raise ValueError(f"No semaphore defined for adapter '{adapter}'")
        return semaphore

    def _build_prompt(self, messages: list[Message]) -> str:
        """
        Construct a prompt string from a list of structured messages.

        Args:
            messages (list[Message]):
                A list of Message objects containing role and content fields.

        Returns:
            str:
                A formatted string representing the full conversation prompt.
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
            GenerationConfig:
                A GenerationConfig instance with user-overrides applied.
        """
        if self._generation_config is None:
            raise RuntimeError("Model must be loaded before calling generate().")

        # Pylance: Type of from_dict() is partially unknown
        config = GenerationConfig.from_dict( # type: ignore[reportUnknownMemberType]
            self._generation_config.to_dict()
        )
        for k, v in parameters.model_dump(exclude_none=True).items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    @abstractmethod
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

    @abstractmethod
    def _get_generate_args(self) -> Tuple[list[str], dict[str, Any]]:
        """
        Specify which arguments to pass to the model's `generate()` method.

        Subclasses define:
        - A list of dynamic input keys to extract from the output of `_tokenize_input()`
        - A dictionary of static keyword arguments to pass with fixed values

        Returns:
            Tuple[list[str], dict[str, Any]]: 
                - List of dynamic keys (e.g., "input_ids", "attention_mask")
                - Dict of static kwargs (e.g., {"num_logits_to_keep": 0})
        """
        raise NotImplementedError("Subclasses must implement this method.")
