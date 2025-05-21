"""
Core data models for OpenReasoning.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, field_validator, model_validator


class Message(BaseModel):
    """A message in a conversation."""

    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """A conversation with messages."""

    id: str
    messages: List[Message]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode='before')
    @classmethod
    def check_at_least_one_source(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not (data.get("url") or data.get("file_path") or data.get("base64_data")):
                raise ValueError(
                    "At least one of url, file_path, or base64_data must be provided"
                )
        return data


class Document(BaseModel):
    """A document for retrieval."""

    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def get_text_chunks(self, chunk_size: int = 1000) -> List["Document"]:
        """Split document into chunks of specified size."""
        if len(self.text) <= chunk_size:
            return [self]

        chunks = []
        for i in range(0, len(self.text), chunk_size):
            chunk_text = self.text[i : i + chunk_size]
            chunks.append(
                Document(
                    id=f"{self.id}-chunk-{i//chunk_size}",
                    text=chunk_text,
                    metadata={
                        **self.metadata,
                        "parent_id": self.id,
                        "chunk_index": i // chunk_size,
                        "is_chunk": True,
                    },
                )
            )
        return chunks


class RetrievalResult(BaseModel):
    """Result from a retrieval operation."""

    documents: List[Document]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Response from a model."""

    id: str
    model: str
    provider: str
    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class ToolCall(BaseModel):
    """A tool call made by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None


class CompletionRequest(BaseModel):
    """Request for a model completion."""

    messages: List[Message]
    model: Optional[str] = None
    provider: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Image(BaseModel):
    """An image for multimodal models."""

    url: Optional[str] = None
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def check_at_least_one_source(cls, data: Any) -> Any:
        """Ensure at least one image source is provided."""
        if isinstance(data, dict):
            if not any(data.get(field) for field in ["url", "file_path", "base64_data"]):
                raise ValueError(
                    "At least one of url, file_path, or base64_data must be provided"
                )
        return data
