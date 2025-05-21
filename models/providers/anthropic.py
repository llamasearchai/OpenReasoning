"""
Anthropic model provider implementation.
"""

import os
from typing import Any, Dict, List, Optional, Union

import anthropic
from loguru import logger

from openreasoning.core.config import settings

from ..base import BaseModelProvider, ModelInput, ModelOutput


class AnthropicProvider(BaseModelProvider):
    """Anthropic API model provider."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to environment variable)
            model: Default model to use
        """
        self.api_key = (
            api_key or settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.default_model = model

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            logger.warning("No Anthropic API key provided")
            self.client = None

    def _prepare_messages(self, input_data: ModelInput) -> List[Dict[str, Any]]:
        """Convert input to Anthropic message format."""
        if isinstance(input_data.prompt, str):
            # Single string prompt -> convert to messages format
            return [{"role": "user", "content": input_data.prompt}]

        # Convert from potentially OpenAI format to Anthropic format
        anthropic_messages = []
        for msg in input_data.prompt:
            role = msg["role"]
            content = msg["content"]

            # Map roles from OpenAI to Anthropic
            if role == "system":
                # Add as system message
                anthropic_messages.append(
                    {"role": "user", "content": [{"type": "text", "text": content}]}
                )
            elif role == "user":
                anthropic_messages.append(
                    {"role": "user", "content": [{"type": "text", "text": content}]}
                )
            elif role == "assistant":
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": content}],
                    }
                )

        return anthropic_messages

    def generate(self, input_data: ModelInput) -> ModelOutput:
        """Generate text using Anthropic API."""
        if not self.client:
            raise ValueError(
                "Anthropic client not initialized. Please provide an API key."
            )

        messages = self._prepare_messages(input_data)
        model = (
            input_data.model_params.get("model", self.default_model)
            if input_data.model_params
            else self.default_model
        )

        try:
            response = self.client.messages.create(
                model=model,
                messages=messages,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens or 1024,
                stop_sequences=input_data.stop_sequences,
                system=(
                    "You are a helpful, harmless, and honest AI assistant."
                    if isinstance(input_data.prompt, str)
                    else None
                ),
                **({} if not input_data.model_params else input_data.model_params),
            )

            return ModelOutput(
                text=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                },
                model_info={"model": model, "provider": "anthropic", "id": response.id},
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get available Anthropic models."""
        # Anthropic doesn't provide a models list API, so we return the known models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

    def validate_api_key(self) -> bool:
        """Validate the Anthropic API key."""
        if not self.api_key:
            return False

        try:
            # Try a simple API call to validate the key
            self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            logger.error(f"Invalid Anthropic API key: {e}")
            return False
