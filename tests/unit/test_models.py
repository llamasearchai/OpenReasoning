"""
Unit tests for model modules.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from openreasoning.core.models import Message, ModelResponse
from openreasoning.models.openai import OpenAIModel

# Skip tests if API key is not available
if not os.environ.get("OPENAI_API_KEY"):
    pytestmark = pytest.mark.skip(reason="OPENAI_API_KEY not available")


class TestOpenAIModel:
    """Tests for OpenAI model integration."""

    def test_initialization(self):
        """Test model initialization."""
        model = OpenAIModel(model="gpt-4")
        assert model.model == "gpt-4"
        assert model.api_key is not None
        assert model.client is not None

    @patch("openreasoning.models.openai.OpenAI")
    def test_complete(self, mock_openai):
        """Test model completion."""
        # Set up mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.id = "test-id"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client.chat.completions.create.return_value = mock_response

        # Create model and get completion
        model = OpenAIModel(model="gpt-4", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        response = model.complete(messages=messages)

        # Verify response
        assert response.id == "test-id"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.content == "Test response"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15

        # Verify mock was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4"
        assert call_args["messages"] == messages
        assert call_args["temperature"] == 0.7
