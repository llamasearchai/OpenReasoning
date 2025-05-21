"""
Base model interface for OpenReasoning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    """Input to a model."""

    prompt: Union[str, List[Dict[str, str]]] = Field(
        ..., description="The prompt or message list"
    )
    temperature: float = Field(0.7, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(
        None, description="Sequences that stop generation"
    )
    model_params: Optional[Dict[str, Any]] = Field(
        None, description="Additional model parameters"
    )


class ModelOutput(BaseModel):
    """Output from a model."""

    text: str = Field(..., description="The generated text")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    model_info: Dict[str, Any] = Field(
        ..., description="Information about the model used"
    )
    raw_response: Optional[Any] = Field(
        None, description="Raw response from the provider"
    )


class BaseModelProvider(ABC):
    """Base class for model providers."""

    @abstractmethod
    def generate(self, input_data: ModelInput) -> ModelOutput:
        """Generate text given an input."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get a list of available models from this provider."""
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate that the API key is correct and working."""
        pass
