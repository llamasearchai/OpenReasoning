"""
Agent providers for OpenReasoning.
"""

from typing import Dict, Optional, Type

from ..base import BaseAgent

# Import providers (will be populated as they're implemented)
try:
    from .reasoning import ReasoningAgent

    HAS_REASONING = True
except ImportError:
    HAS_REASONING = False

try:
    from .multiagent import MultiAgentSystem

    HAS_MULTIAGENT = True
except ImportError:
    HAS_MULTIAGENT = False

try:
    from .rag import RAGAgent

    HAS_RAG = True
except ImportError:
    HAS_RAG = False


# Registry of available providers
PROVIDERS: Dict[str, Type[BaseAgent]] = {}

# Register providers if available
if HAS_REASONING:
    PROVIDERS["reasoning"] = ReasoningAgent

if HAS_MULTIAGENT:
    PROVIDERS["multiagent"] = MultiAgentSystem

if HAS_RAG:
    PROVIDERS["rag"] = RAGAgent


def get_agent(agent_name: str, **kwargs) -> BaseAgent:
    """
    Get an agent instance by name.

    Args:
        agent_name: The name of the agent provider
        **kwargs: Additional arguments to pass to the agent constructor

    Returns:
        An instance of the specified agent

    Raises:
        ValueError: If the agent is not found
    """
    if agent_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Agent provider '{agent_name}' not found. Available providers: {available}"
        )

    return PROVIDERS[agent_name](**kwargs)
