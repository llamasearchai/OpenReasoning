"""
Hybrid search implementation for OpenReasoning.
"""

import heapq
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.config import settings
from ..core.models import Document, RetrievalResult

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # Download NLTK resources if they don't exist
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    NLTK_AVAILABLE = True

except ImportError:
    NLTK_AVAILABLE = False
    logger.warning(
        "NLTK is not available. Using basic tokenization for keyword search."
    )


class HybridSearch:
    """Hybrid search combining semantic and keyword search."""

    def __init__(
        self, vector_store, semantic_weight: float = 0.7, cache_enabled: bool = True
    ):
        """Initialize hybrid search."""
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.cache_enabled = cache_enabled
        self.query_cache = {}  # Cache for recent queries
        self.cache_capacity = 100  # Maximum number of cached queries

        # Set up text processing for keyword search
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words("english"))

        # Build keyword index
        self.keyword_index = {}
        self.document_map = {}

        for doc_id, doc in self.vector_store.documents.items():
            self.document_map[doc_id] = doc
            self._index_document(doc)

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword search."""
        if NLTK_AVAILABLE:
            # Tokenize, remove stopwords, and stem
            tokens = word_tokenize(text.lower())
            tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token.isalnum() and token not in self.stop_words
            ]
        else:
            # Basic preprocessing
            tokens = text.lower().split()
            tokens = [token for token in tokens if token.isalnum()]

        return tokens

    def _index_document(self, document: Document) -> None:
        """Index a document for keyword search."""
        tokens = self._preprocess_text(document.text)

        # Count term frequency in document
        term_freq = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Add to index
        for token, freq in term_freq.items():
            if token not in self.keyword_index:
                self.keyword_index[token] = {}

            self.keyword_index[token][document.id] = freq

    def _keyword_search(self, query: str, limit: int = 5) -> Dict[str, float]:
        """Perform keyword search."""
        query_tokens = self._preprocess_text(query)

        # Calculate TF-IDF scores
        document_scores = {}
        num_documents = len(self.document_map)

        for token in query_tokens:
            if token in self.keyword_index:
                # Calculate IDF
                doc_freq = len(self.keyword_index[token])
                idf = np.log((num_documents + 1) / (doc_freq + 1)) + 1

                # Update scores for matching documents
                for doc_id, term_freq in self.keyword_index[token].items():
                    if doc_id not in document_scores:
                        document_scores[doc_id] = 0

                    # TF-IDF score
                    document_scores[doc_id] += term_freq * idf

        # Normalize scores
        if document_scores:
            max_score = max(document_scores.values())
            if max_score > 0:
                for doc_id in document_scores:
                    document_scores[doc_id] /= max_score

        return document_scores

    def add_document(self, document: Document) -> None:
        """Add a document to the search index."""
        # Add to document map
        self.document_map[document.id] = document

        # Index for keyword search
        self._index_document(document)

        # Clear cache to ensure fresh results
        self.query_cache = {}

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the search index."""
        for document in documents:
            # Add to document map
            self.document_map[document.id] = document

            # Index for keyword search
            self._index_document(document)

        # Clear cache to ensure fresh results
        self.query_cache = {}

    def remove_document(self, document_id: str) -> None:
        """Remove a document from the search index."""
        if document_id in self.document_map:
            # Remove from document map
            del self.document_map[document_id]

            # Remove from keyword index
            for token, docs in list(self.keyword_index.items()):
                if document_id in docs:
                    del self.keyword_index[token][document_id]

                    # Remove token if no documents left
                    if not self.keyword_index[token]:
                        del self.keyword_index[token]

            # Clear cache to ensure fresh results
            self.query_cache = {}

    def search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 5,
        filter_fn: Optional[Callable[[Document], bool]] = None,
        semantic_weight: Optional[float] = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        """Perform hybrid search."""
        start_time = time.time()

        # Try to use cache if enabled and requested
        cache_key = f"{query}_{limit}_{semantic_weight or self.semantic_weight}"
        if self.cache_enabled and use_cache and cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result.metadata["cache_hit"] = True
            cached_result.metadata["latency_seconds"] = time.time() - start_time
            return cached_result

        # Use provided semantic weight or default
        sem_weight = (
            semantic_weight if semantic_weight is not None else self.semantic_weight
        )
        key_weight = 1.0 - sem_weight

        # Perform semantic search
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit * 2,  # Get more results to refine
            filter_fn=filter_fn,
        )

        # Perform keyword search
        keyword_scores = self._keyword_search(query, limit * 2)

        # Combine results with weighted scores
        combined_scores = {}

        # Process semantic results
        for doc in semantic_results.documents:
            # Calculate normalized semantic score from metadata
            semantic_score = doc.metadata.get("score", 0.5)  # Default if not available

            # Get keyword score if available
            keyword_score = keyword_scores.get(doc.id, 0.0)

            # Calculate combined score
            combined_score = sem_weight * semantic_score + key_weight * keyword_score

            combined_scores[doc.id] = (combined_score, doc)

        # Add any keyword-only results not in semantic results
        for doc_id, keyword_score in keyword_scores.items():
            if doc_id not in combined_scores and doc_id in self.document_map:
                doc = self.document_map[doc_id]

                if filter_fn and not filter_fn(doc):
                    continue

                combined_score = key_weight * keyword_score
                combined_scores[doc.id] = (combined_score, doc)

        # Sort by combined score and take top results
        top_results = []
        for doc_id, (score, doc) in sorted(
            combined_scores.items(), key=lambda x: x[1][0], reverse=True
        )[:limit]:
            # Add combined score to metadata
            doc_with_score = Document(
                id=doc.id,
                text=doc.text,
                metadata={**doc.metadata, "combined_score": score},
                embedding=doc.embedding,
            )
            top_results.append(doc_with_score)

        end_time = time.time()

        result = RetrievalResult(
            documents=top_results,
            metadata={
                "total": len(top_results),
                "limit": limit,
                "semantic_weight": sem_weight,
                "keyword_weight": key_weight,
                "latency_seconds": end_time - start_time,
                "cache_hit": False,
            },
        )

        # Cache result if enabled
        if self.cache_enabled:
            # Maintain cache size
            if len(self.query_cache) >= self.cache_capacity:
                # Remove random entry to keep cache size in check
                # More sophisticated LRU cache could be implemented
                self.query_cache.pop(next(iter(self.query_cache)))

            self.query_cache[cache_key] = result

        return result

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        filter_fn: Optional[Callable[[Document], bool]] = None,
        semantic_weight: Optional[float] = None,
        embedding_fn: Callable[[List[str]], List[List[float]]] = None,
    ) -> RetrievalResult:
        """Search by text, computing embeddings on the fly."""
        if embedding_fn is None:
            raise ValueError("embedding_fn is required")

        # Get embedding for query
        query_embedding = embedding_fn([query])[0]

        # Perform hybrid search
        return self.search(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
            filter_fn=filter_fn,
            semantic_weight=semantic_weight,
        )
