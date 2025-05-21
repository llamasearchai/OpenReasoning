"""
Retrieval providers for OpenReasoning.
"""

from typing import Dict, Optional, Type

from ..base import BaseRetriever

# Import providers (will be populated as they're implemented)
try:
    from .chroma import ChromaRetriever

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from .faiss import FaissRetriever

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from .milvus import MilvusRetriever

    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False


# Registry of available providers
PROVIDERS: Dict[str, Type[BaseRetriever]] = {}

# Register providers if available
if HAS_CHROMA:
    PROVIDERS["chroma"] = ChromaRetriever

if HAS_FAISS:
    PROVIDERS["faiss"] = FaissRetriever

if HAS_MILVUS:
    PROVIDERS["milvus"] = MilvusRetriever


def get_retriever(provider_name: str, **kwargs) -> BaseRetriever:
    """
    Get a retriever instance by name.

    Args:
        provider_name: The name of the retriever provider
        **kwargs: Additional arguments to pass to the retriever constructor

    Returns:
        An instance of the specified retriever

    Raises:
        ValueError: If the provider is not found
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Retriever provider '{provider_name}' not found. Available providers: {available}"
        )

    return PROVIDERS[provider_name](**kwargs)
