"""
HuggingFace model provider implementation.
"""

import os
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger

from openreasoning.core.config import settings

from ..base import BaseModelProvider, ModelInput, ModelOutput


class HuggingFaceProvider(BaseModelProvider):
    """HuggingFace API model provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ):
        """
        Initialize the HuggingFace provider.

        Args:
            api_key: HuggingFace API key (defaults to environment variable)
            model: Default model to use
        """
        self.api_key = (
            api_key or settings.huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        )
        self.default_model = model
        self.api_url = "https://api-inference.huggingface.co/models/"

        if not self.api_key:
            logger.warning("No HuggingFace API key provided")

    def _prepare_prompt(self, input_data: ModelInput) -> str:
        """Convert input to text prompt for HuggingFace."""
        if isinstance(input_data.prompt, str):
            return input_data.prompt

        # Convert from message format to string
        prompt_parts = []

        for msg in input_data.prompt:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")

        # Add the final assistant prompt to generate a response
        if not prompt_parts[-1].startswith("<|assistant|>"):
            prompt_parts.append("<|assistant|>")

        return "\n".join(prompt_parts)

    def generate(self, input_data: ModelInput) -> ModelOutput:
        """Generate text using HuggingFace API."""
        if not self.api_key:
            raise ValueError("HuggingFace API key not provided.")

        prompt = self._prepare_prompt(input_data)
        model = (
            input_data.model_params.get("model", self.default_model)
            if input_data.model_params
            else self.default_model
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": input_data.temperature,
                "max_new_tokens": input_data.max_tokens or 512,
                "return_full_text": False,
            },
        }

        # Add stop sequences if provided
        if input_data.stop_sequences:
            payload["parameters"]["stop"] = input_data.stop_sequences

        # Add any additional parameters
        if input_data.model_params:
            payload["parameters"].update(input_data.model_params)

        try:
            response = requests.post(
                f"{self.api_url}{model}", headers=headers, json=payload
            )
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = result.get("generated_text", "")

            # HuggingFace doesn't return token counts, so we estimate
            prompt_tokens = len(prompt) // 4  # Very rough estimate
            completion_tokens = len(generated_text) // 4  # Very rough estimate

            return ModelOutput(
                text=generated_text,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                model_info={"model": model, "provider": "huggingface"},
                raw_response=result,
            )

        except Exception as e:
            logger.error(f"Error calling HuggingFace API: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get a list of recommended HuggingFace models."""
        # Return a curated list of popular models
        return [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "tiiuae/falcon-40b-instruct",
            "tiiuae/falcon-7b-instruct",
            "microsoft/phi-2",
            "HuggingFaceH4/zephyr-7b-beta",
        ]

    def validate_api_key(self) -> bool:
        """Validate the HuggingFace API key."""
        if not self.api_key:
            return False

        try:
            # Check API key by making a simple model info request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.api_url}{self.default_model}", headers=headers
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Invalid HuggingFace API key: {e}")
            return False
