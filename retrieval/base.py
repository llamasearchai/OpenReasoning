"""
Base classes for retrieval systems in OpenReasoning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document in the retrieval system."""

    content: str = Field(..., description="The document content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    id: Optional[str] = Field(None, description="Document ID")
    score: Optional[float] = Field(None, description="Retrieval score")

    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(id={self.id}, content={self.content[:50]}..., score={self.score})"


class SearchQuery(BaseModel):
    """A search query for the retrieval system."""

    text: str = Field(..., description="Query text")
    k: int = Field(5, description="Number of results to retrieve")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

    def __str__(self) -> str:
        """String representation of the query."""
        return f"SearchQuery(text={self.text}, k={self.k}, filter={self.filter})"


class BaseRetriever(ABC):
    """Base class for document retrievers."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[Document]:
        """
        Search for documents matching the query.

        Args:
            query: The search query

        Returns:
            A list of matching documents
        """
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the retrieval system.

        Args:
            documents: The documents to add

        Returns:
            A list of document IDs
        """
        pass

    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from the retrieval system.

        Args:
            ids: The document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_document(self, id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            id: The document ID

        Returns:
            The document, or None if not found
        """
        pass
