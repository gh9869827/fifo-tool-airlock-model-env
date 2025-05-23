from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
import json
import logging
from fifo_tool_airlock_model_env.common.models import Model
from pydantic import BaseModel, Field

from fifo_tool_airlock_model_env.server.llm_model import (
    LLMModel
)
from fifo_tool_airlock_model_env.server.llm_model_phi_4_mini_instruct import (
    LLMModelPhi4MiniInstruct
)
from fifo_tool_airlock_model_env.server.llm_model_phi_4_multimodal_instruct import (
    LLMModelPhi4MultimodalInstruct
)

class ModelConfig(BaseModel):
    """
    Configuration for a single model instance.

    Attributes:
        name (str):
            Unique name used to reference the model instance.

        class_name (str):
            Fully qualified model class name as a string.

        model_path (str):
            HuggingFace model ID or local path.

        adapter_map (Dict[str, str]):
            Dictionary mapping adapter names to file paths.

        max_concurrent_per_adapter (int):
            Maximum concurrent inference jobs per adapter.
    """
    name: str = Field(..., description="Model instance name (e.g. 'phi4MiniInstruct')")
    class_name: str = Field(..., description="Model class name (e.g. 'LLMModelPhi4MiniInstruct')")
    model_path: str
    adapter_map: Dict[str, str] = Field(default_factory=dict)
    max_concurrent_per_adapter: int = 1


class AppConfig(BaseModel):
    """
    Root configuration model containing a list of model configurations.

    Attributes:
        models (list[ModelConfig]):
            List of model configurations to load.
    """
    models: list[ModelConfig]


@dataclass
class LoadedModels:
    """
    Dynamic registry of loaded model instances.

    Attributes:
        _registry (Dict[Model, LLMModel]):
            Dictionary mapping model enums to loaded model instances.
    """
    _registry: Dict[str, object] = field(default_factory=dict)

    def get_model(self, model: Model) -> LLMModel | None:
        """
        Retrieve a loaded model instance using its enum identifier.

        Args:
            model (Model):
                Enum key corresponding to the model name defined in the config.

        Returns:
            LLMModel | None: The model instance, or None if not found.
        """
        return self._registry.get(model)

    def load_from_config(self, path: str = "model_config.json") -> None:
        """
        Load models defined in a JSON configuration file.

        Args:
            path (str):
                Path to the JSON config file.

        Raises:
            ValueError: If the model class or name is not valid.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        config = AppConfig.model_validate(raw)

        available_classes: Dict[str, Tuple[Type[LLMModel], Model]] = {
            "LLMModelPhi4MiniInstruct": (
                LLMModelPhi4MiniInstruct,
                Model.Phi4MiniInstruct
            ),
            "LLMModelPhi4MultimodalInstruct": (
                LLMModelPhi4MultimodalInstruct,
                Model.Phi4MultimodalInstruct
            )
        }

        for model_cfg in config.models:
            entry = available_classes.get(model_cfg.class_name)
            if entry is None:
                raise ValueError(f"Unknown class_name '{model_cfg.class_name}'")
            cls, model = entry

            logging.info("📦 Loading '%s' as %s", model_cfg.model_path, model_cfg.name)

            instance = cls(
                model_path=model_cfg.model_path,
                adapter_map=model_cfg.adapter_map,
                max_concurrent_per_adapter=model_cfg.max_concurrent_per_adapter
            )
            instance.load_model()
            self._registry[model] = instance
