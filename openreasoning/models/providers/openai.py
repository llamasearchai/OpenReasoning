"""
OpenAI model provider implementation.
"""

import os
from typing import Any, Dict, List, Optional, Union

import openai
from loguru import logger

from openreasoning.core.config import settings

from ..base import BaseModelProvider, ModelInput, ModelOutput


class OpenAIProvider(BaseModelProvider):
    """OpenAI API model provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Default model to use
        """
        self.api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.default_model = model

        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            logger.warning("No OpenAI API key provided")
            self.client = None

    def _prepare_messages(self, input_data: ModelInput) -> List[Dict[str, str]]:
        """Convert input to OpenAI message format."""
        if isinstance(input_data.prompt, str):
            # Single string prompt -> convert to messages format
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_data.prompt},
            ]

        # Already in messages format
        return input_data.prompt

    def generate(self, input_data: ModelInput) -> ModelOutput:
        """Generate text using OpenAI API."""
        if not self.client:
            raise ValueError(
                "OpenAI client not initialized. Please provide an API key."
            )

        messages = self._prepare_messages(input_data)
        model = (
            input_data.model_params.get("model", self.default_model)
            if input_data.model_params
            else self.default_model
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
                stop=input_data.stop_sequences,
                **({} if not input_data.model_params else input_data.model_params),
            )

            return ModelOutput(
                text=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                model_info={"model": model, "provider": "openai", "id": response.id},
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get available OpenAI models."""
        if not self.client:
            return []

        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            return []

    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key."""
        if not self.api_key:
            return False

        try:
            # Try to list models as a simple API test
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Invalid OpenAI API key: {e}")
            return False
