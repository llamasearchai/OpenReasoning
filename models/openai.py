"""
OpenAI integration for OpenReasoning.
"""

import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from ..core.config import settings
from ..core.models import Image, Message, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class OpenAIModel:
    """OpenAI model integration."""

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        streaming: bool = False,
        api_base: Optional[str] = None,
    ):
        """Initialize OpenAI model."""
        self.model = model or settings.default_model
        self.api_key = (
            api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        )
        self.streaming = streaming
        self.api_base = api_base

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def _process_messages_with_images(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process messages that may contain images."""
        processed_messages = []

        for message in messages:
            if isinstance(message, dict):
                # Handle messages with images
                if "images" in message and message.get("role") == "user":
                    # Extract images
                    images = message.pop("images", [])
                    content = message.get("content", "")

                    # Build content array for multimodal message
                    content_array = [{"type": "text", "text": content}]

                    for img in images:
                        # Handle different image sources
                        image_part = {"type": "image_url"}

                        if isinstance(img, dict):
                            # Handle pre-formatted image object
                            if "url" in img:
                                image_part["image_url"] = {"url": img["url"]}
                            elif "file_path" in img:
                                # Load image from file
                                try:
                                    with open(img["file_path"], "rb") as f:
                                        base64_image = base64.b64encode(
                                            f.read()
                                        ).decode("utf-8")
                                        image_part["image_url"] = {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                except Exception as e:
                                    logger.error(
                                        f"Failed to load image from {img['file_path']}: {str(e)}"
                                    )
                                    continue
                            elif "base64_data" in img:
                                # Use provided base64 data
                                image_part["image_url"] = {
                                    "url": f"data:image/jpeg;base64,{img['base64_data']}"
                                }
                        elif isinstance(img, str):
                            # Assume it's a URL
                            image_part["image_url"] = {"url": img}

                        content_array.append(image_part)

                    # Create new message with content array
                    new_message = {**message, "content": content_array}
                    processed_messages.append(new_message)
                else:
                    # No image processing needed
                    processed_messages.append(message)
            else:
                # Pass through non-dict messages
                processed_messages.append(message)

        return processed_messages

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Get a completion from the model."""
        try:
            # Process messages that may contain images
            processed_messages = self._process_messages_with_images(messages)

            kwargs = {
                "model": self.model,
                "messages": processed_messages,
                "temperature": temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            if tools:
                kwargs["tools"] = tools

            start_time = time.time()
            response = self.client.chat.completions.create(**kwargs)
            end_time = time.time()

            tool_calls = []
            if (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                for tool_call in response.choices[0].message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tool_call.function.arguments}

                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=arguments,
                        )
                    )

            model_response = ModelResponse(
                id=response.id,
                model=self.model,
                provider="openai",
                content=response.choices[0].message.content or "",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "latency_seconds": end_time - start_time,
                },
                metadata={
                    "tool_calls": tool_calls if tool_calls else None,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing OpenAI request: {e}")
            raise

    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Get a completion from the model asynchronously."""
        try:
            # Process messages that may contain images
            processed_messages = self._process_messages_with_images(messages)

            kwargs = {
                "model": self.model,
                "messages": processed_messages,
                "temperature": temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            if tools:
                kwargs["tools"] = tools

            start_time = time.time()
            response = await self.async_client.chat.completions.create(**kwargs)
            end_time = time.time()

            tool_calls = []
            if (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                for tool_call in response.choices[0].message.tool_calls:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": tool_call.function.arguments}

                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=arguments,
                        )
                    )

            model_response = ModelResponse(
                id=response.id,
                model=self.model,
                provider="openai",
                content=response.choices[0].message.content or "",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "latency_seconds": end_time - start_time,
                },
                metadata={
                    "tool_calls": tool_calls if tool_calls else None,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing OpenAI request: {e}")
            raise

    def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for texts."""
        try:
            embedding_model = model or settings.default_embedding_model
            response = self.client.embeddings.create(model=embedding_model, input=texts)

            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    async def get_embeddings_async(
        self, texts: List[str], model: str = None
    ) -> List[List[float]]:
        """Get embeddings for texts asynchronously."""
        try:
            embedding_model = model or settings.default_embedding_model
            response = await self.async_client.embeddings.create(
                model=embedding_model, input=texts
            )

            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
