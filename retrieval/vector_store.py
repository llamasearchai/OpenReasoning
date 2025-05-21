"""
Vector store implementation for OpenReasoning.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.config import settings
from ..core.models import Document, RetrievalResult

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS is not available. Using numpy for vector similarity.")


class VectorStore:
    """In-memory vector store implementation."""

    def __init__(self, dimension: int = 1536):
        """Initialize the vector store."""
        self.dimension = dimension
        self.documents = {}  # id -> Document
        self.embeddings = {}  # id -> embedding

        # Initialize FAISS index if available
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = None

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        document_ids = []
        embeddings_to_add = []

        for document in documents:
            if not document.id:
                document.id = str(uuid.uuid4())

            if not document.embedding:
                raise ValueError(f"Document {document.id} missing embedding")

            if len(document.embedding) != self.dimension:
                raise ValueError(
                    f"Document {document.id} has embedding of dimension {len(document.embedding)}, expected {self.dimension}"
                )

            self.documents[document.id] = document
            self.embeddings[document.id] = document.embedding
            document_ids.append(document.id)
            embeddings_to_add.append(document.embedding)

        # Add to FAISS index if available
        if FAISS_AVAILABLE and embeddings_to_add:
            self.index.add(np.array(embeddings_to_add, dtype=np.float32))

        return document_ids

    def add_documents_with_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embedding_fn: Callable[[List[str]], List[List[float]]] = None,
    ) -> List[str]:
        """Add documents from texts, with optional metadata and embedding function."""
        if embedding_fn is None:
            raise ValueError("embedding_fn is required")

        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Prepare ids
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Get embeddings
        embeddings = embedding_fn(texts)

        # Create documents
        documents = []
        for i, (text, metadata, doc_id, embedding) in enumerate(
            zip(texts, metadatas, ids, embeddings)
        ):
            doc = Document(id=doc_id, text=text, metadata=metadata, embedding=embedding)
            documents.append(doc)

        # Add documents
        return self.add_documents(documents)

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filter_fn: Optional[Callable[[Document], bool]] = None,
        k_prefetch: int = None,
    ) -> RetrievalResult:
        """Search for documents similar to the query embedding."""
        if not self.documents:
            return RetrievalResult(documents=[], metadata={"total": 0})

        start_time = time.time()
        k_prefetch = k_prefetch or (limit * 2)

        if FAISS_AVAILABLE:
            # Use FAISS for search
            query_embedding_np = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_embedding_np, k_prefetch)

            # Get document IDs from indices
            doc_ids = list(self.documents.keys())
            result_docs = []

            for i, idx in enumerate(indices[0]):
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    doc = self.documents[doc_id]

                    if filter_fn and not filter_fn(doc):
                        continue

                    # Add distance to metadata
                    doc_with_score = Document(
                        id=doc.id,
                        text=doc.text,
                        metadata={
                            **doc.metadata,
                            "score": float(1.0 / (1.0 + distances[0][i])),
                        },
                        embedding=doc.embedding,
                    )
                    result_docs.append(doc_with_score)

                    if len(result_docs) >= limit:
                        break
        else:
            # Use numpy for search (slower)
            query_embedding_np = np.array(query_embedding)
            similarities = {}

            for doc_id, embedding in self.embeddings.items():
                doc = self.documents[doc_id]

                if filter_fn and not filter_fn(doc):
                    continue

                embedding_np = np.array(embedding)
                # Use cosine similarity
                similarity = np.dot(query_embedding_np, embedding_np) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding_np)
                )
                similarities[doc_id] = similarity

            # Sort by similarity (descending)
            sorted_ids = sorted(
                similarities.keys(), key=lambda x: similarities[x], reverse=True
            )

            # Get the top documents
            result_docs = []
            for doc_id in sorted_ids[:limit]:
                doc = self.documents[doc_id]
                # Add score to metadata
                doc_with_score = Document(
                    id=doc.id,
                    text=doc.text,
                    metadata={**doc.metadata, "score": float(similarities[doc_id])},
                    embedding=doc.embedding,
                )
                result_docs.append(doc_with_score)

        end_time = time.time()

        return RetrievalResult(
            documents=result_docs,
            metadata={
                "total": len(result_docs),
                "limit": limit,
                "latency_seconds": end_time - start_time,
            },
        )

    def search_by_text(
        self,
        query_text: str,
        limit: int = 5,
        filter_fn: Optional[Callable[[Document], bool]] = None,
        embedding_fn: Callable[[List[str]], List[List[float]]] = None,
    ) -> RetrievalResult:
        """Search for documents by query text."""
        if embedding_fn is None:
            raise ValueError("embedding_fn is required")

        # Get embedding for query
        query_embedding = embedding_fn([query_text])[0]

        # Search by embedding
        return self.search(query_embedding, limit, filter_fn)

    def delete(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        # FAISS doesn't support deletion directly, so we need to rebuild the index
        rebuild_index = False

        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                del self.embeddings[doc_id]
                rebuild_index = True

        if FAISS_AVAILABLE and rebuild_index:
            # Rebuild the index
            self.index = faiss.IndexFlatL2(self.dimension)
            if self.embeddings:
                embeddings_list = list(self.embeddings.values())
                self.index.add(np.array(embeddings_list, dtype=np.float32))

    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save documents
        documents_path = os.path.join(path, "documents.json")
        with open(documents_path, "w") as f:
            # Convert Document objects to dictionaries
            doc_dicts = {}
            for doc_id, doc in self.documents.items():
                doc_dict = doc.dict()
                doc_dicts[doc_id] = doc_dict

            json.dump(doc_dicts, f)

        # Save FAISS index if available
        if FAISS_AVAILABLE and self.index:
            index_path = os.path.join(path, "index.faiss")
            faiss.write_index(self.index, index_path)
        else:
            # Save embeddings separately
            embeddings_path = os.path.join(path, "embeddings.npz")
            np.savez(
                embeddings_path,
                **{
                    f"emb_{doc_id}": embedding
                    for doc_id, embedding in self.embeddings.items()
                },
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a vector store from disk."""
        # Load documents
        documents_path = os.path.join(path, "documents.json")
        with open(documents_path, "r") as f:
            doc_dicts = json.load(f)

        # Create vector store
        dimension = (
            next(iter(doc_dicts.values()))["embedding"][0] if doc_dicts else 1536
        )
        vector_store = cls(dimension=dimension)

        # Create Document objects
        documents = []
        for doc_id, doc_dict in doc_dicts.items():
            doc = Document(**doc_dict)
            documents.append(doc)

        # Add documents without embeddings
        vector_store.documents = {doc.id: doc for doc in documents}
        vector_store.embeddings = {doc.id: doc.embedding for doc in documents}

        # Load FAISS index if available
        index_path = os.path.join(path, "index.faiss")
        if FAISS_AVAILABLE and os.path.exists(index_path):
            vector_store.index = faiss.read_index(index_path)
        elif FAISS_AVAILABLE:
            # Rebuild index from embeddings
            vector_store.index = faiss.IndexFlatL2(vector_store.dimension)
            if vector_store.embeddings:
                embeddings_list = list(vector_store.embeddings.values())
                vector_store.index.add(np.array(embeddings_list, dtype=np.float32))

        return vector_store
