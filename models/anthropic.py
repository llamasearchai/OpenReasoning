"""
Anthropic integration for OpenReasoning.
"""

import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import anthropic
from anthropic import Anthropic, AsyncAnthropic

from ..core.config import settings
from ..core.models import Image, Message, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class AnthropicModel:
    """Anthropic model integration."""

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: str = None,
        streaming: bool = False,
    ):
        """Initialize Anthropic model."""
        self.model = model
        self.api_key = (
            api_key or settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.streaming = streaming

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert message format to Anthropic format with support for images."""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # Handle as a system prompt for Claude
                anthropic_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                # Check for images in the user message
                if "images" in msg:
                    # Process multimodal content
                    images = msg.pop("images", [])
                    content = []

                    # Add text content first
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})

                    # Add images
                    for img in images:
                        if isinstance(img, dict):
                            if "url" in img:
                                content.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "url", "url": img["url"]},
                                    }
                                )
                            elif "file_path" in img:
                                # Load image from file
                                try:
                                    with open(img["file_path"], "rb") as f:
                                        media_data = base64.b64encode(f.read()).decode(
                                            "utf-8"
                                        )
                                        content.append(
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": "image/jpeg",
                                                    "data": media_data,
                                                },
                                            }
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Failed to load image from {img['file_path']}: {str(e)}"
                                    )
                            elif "base64_data" in img:
                                content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": img["base64_data"],
                                        },
                                    }
                                )
                        elif isinstance(img, str):
                            # Assume it's a URL
                            content.append(
                                {"type": "image", "source": {"type": "url", "url": img}}
                            )

                    anthropic_messages.append({"role": "user", "content": content})
                else:
                    # Regular user message
                    anthropic_messages.append(
                        {"role": "user", "content": msg["content"]}
                    )
            elif msg["role"] == "assistant":
                anthropic_messages.append(
                    {"role": "assistant", "content": msg["content"]}
                )
            # Anthropic doesn't have a direct function output role, so we may need to handle this differently

        return anthropic_messages

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ModelResponse:
        """Get a completion from the model."""
        try:
            anthropic_messages = self._convert_messages(messages)

            kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            # Handle tools for Claude (format is different than OpenAI)
            if tools:
                # Convert OpenAI tool format to Anthropic tool format if needed
                anthropic_tools = []
                for tool in tools:
                    if "function" in tool:
                        anthropic_tool = {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "input_schema": tool["function"]["parameters"],
                        }
                        anthropic_tools.append(anthropic_tool)
                    else:
                        # Assume it's already in Anthropic format
                        anthropic_tools.append(tool)

                kwargs["tools"] = anthropic_tools

            start_time = time.time()
            response = self.client.messages.create(**kwargs)
            end_time = time.time()

            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        arguments = tool_call.input
                    except:
                        arguments = {}

                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id, name=tool_call.name, arguments=arguments
                        )
                    )

            # Extract text content from response
            content = ""
            if response.content and len(response.content) > 0:
                for block in response.content:
                    if block.type == "text":
                        content += block.text

            model_response = ModelResponse(
                id=response.id,
                model=self.model,
                provider="anthropic",
                content=content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_seconds": end_time - start_time,
                },
                metadata={
                    "tool_calls": tool_calls if tool_calls else None,
                    "stop_reason": response.stop_reason,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing Anthropic request: {e}")
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
            anthropic_messages = self._convert_messages(messages)

            kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            # Handle tools for Claude (format is different than OpenAI)
            if tools:
                # Convert OpenAI tool format to Anthropic tool format if needed
                anthropic_tools = []
                for tool in tools:
                    if "function" in tool:
                        anthropic_tool = {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "input_schema": tool["function"]["parameters"],
                        }
                        anthropic_tools.append(anthropic_tool)
                    else:
                        # Assume it's already in Anthropic format
                        anthropic_tools.append(tool)

                kwargs["tools"] = anthropic_tools

            start_time = time.time()
            response = await self.async_client.messages.create(**kwargs)
            end_time = time.time()

            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        arguments = tool_call.input
                    except:
                        arguments = {}

                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id, name=tool_call.name, arguments=arguments
                        )
                    )

            # Extract text content from response
            content = ""
            if response.content and len(response.content) > 0:
                for block in response.content:
                    if block.type == "text":
                        content += block.text

            model_response = ModelResponse(
                id=response.id,
                model=self.model,
                provider="anthropic",
                content=content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_seconds": end_time - start_time,
                },
                metadata={
                    "tool_calls": tool_calls if tool_calls else None,
                    "stop_reason": response.stop_reason,
                },
            )

            return model_response

        except Exception as e:
            logger.error(f"Error completing Anthropic request: {e}")
            raise
