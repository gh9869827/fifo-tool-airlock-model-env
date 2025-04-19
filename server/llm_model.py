
from abc import ABC, abstractmethod
from common.models import InferenceRequestContainerized

class LLMModel(ABC):
    """
    Abstract base class for model implementations.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Load and initialize the model and its dependencies.
        """

    @abstractmethod
    def generate(self, request: InferenceRequestContainerized) -> str:
        """
        Generate a response from a list of role-based messages.
        """

    def _print_token_stats(self, input_tokens: int, output_tokens: int, duration: float) -> None:
        print(f"📥 {input_tokens:>4} tokens in   ➜   📤 {output_tokens:>4} tokens out   ⏱️ {duration:.2f}s")

