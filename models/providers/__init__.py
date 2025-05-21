"""
Model provider implementations.
"""

from typing import Dict, Type

from ..base import BaseModelProvider
from .anthropic import AnthropicProvider
from .huggingface import HuggingFaceProvider
from .openai import OpenAIProvider

# Registry of model providers
PROVIDERS: Dict[str, Type[BaseModelProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "huggingface": HuggingFaceProvider,
}
