"""
Base classes for agents in OpenReasoning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from openreasoning.models.base import ModelInput, ModelOutput


class Tool(BaseModel):
    """A tool that an agent can use."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )

    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"


class ToolCall(BaseModel):
    """A call to a tool by an agent."""

    tool_name: str = Field(..., description="Tool name")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool call")

    def __str__(self) -> str:
        """String representation of the tool call."""
        return f"{self.tool_name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})"


class ToolResult(BaseModel):
    """Result of a tool call."""

    tool_name: str = Field(..., description="Tool name")
    result: Any = Field(..., description="Result of the tool call")
    error: Optional[str] = Field(
        None, description="Error message if the tool call failed"
    )

    def __str__(self) -> str:
        """String representation of the tool result."""
        if self.error:
            return f"{self.tool_name} failed: {self.error}"
        return f"{self.tool_name} succeeded: {self.result}"


class AgentRequest(BaseModel):
    """Request to an agent."""

    prompt: str = Field(..., description="The prompt or task for the agent")
    tools: Optional[List[Tool]] = Field(None, description="Available tools")
    context: Optional[str] = Field(
        None, description="Additional context for the request"
    )
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model parameters")


class AgentResponse(BaseModel):
    """Response from an agent."""

    answer: str = Field(..., description="The agent's answer")
    reasoning: Optional[str] = Field(None, description="The agent's reasoning")
    tool_calls: Optional[List[ToolCall]] = Field(
        None, description="Tool calls made by the agent"
    )
    tool_results: Optional[List[ToolResult]] = Field(
        None, description="Results of tool calls"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class BaseAgent(ABC):
    """Base class for agents."""

    @abstractmethod
    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        Run the agent on a request.

        Args:
            request: The agent request

        Returns:
            The agent's response
        """
        pass

    @abstractmethod
    def get_available_tools(self) -> List[Tool]:
        """
        Get the tools available to this agent.

        Returns:
            A list of available tools
        """
        pass

    @abstractmethod
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            The result of the tool call
        """
        pass
