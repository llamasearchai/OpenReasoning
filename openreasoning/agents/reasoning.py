"""
Reasoning agent implementation for OpenReasoning.
"""

import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.models import Message, ModelResponse, ToolCall
from ..models.anthropic import AnthropicModel
from ..models.mlx_optimized import MLXModel
from ..models.openai import OpenAIModel
from ..utils.m3_optimizer import m3_optimizer
from .tools import registry

logger = logging.getLogger(__name__)


class ReasoningStep(BaseModel):
    """A step in the reasoning process."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    content: Any
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningAgent:
    """Reasoning agent that can use tools and perform multi-step reasoning."""

    def __init__(
        self,
        model_provider: str = None,
        model_name: str = None,
        max_iterations: int = 10,
        verbose: bool = False,
        temperature: float = 0.7,
        tool_categories: Optional[List[str]] = None,
    ):
        """Initialize the reasoning agent."""
        self.model_provider = model_provider or settings.default_model_provider
        self.model_name = model_name or settings.default_model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.temperature = temperature
        self.tool_categories = tool_categories

        # Enable M3 optimizations if applicable
        if settings.is_m3_chip:
            m3_optimizer.apply_optimizations()

        # Initialize model
        if self.model_provider == "openai":
            self.model = OpenAIModel(model=model_name)
        elif self.model_provider == "anthropic":
            self.model = AnthropicModel(model=model_name or "claude-3-opus-20240229")
        elif self.model_provider == "mlx":
            if not settings.is_apple_silicon:
                logger.warning(
                    "MLX models only work on Apple Silicon. Falling back to OpenAI."
                )
                self.model_provider = "openai"
                self.model = OpenAIModel(model=model_name)
            else:
                self.model = MLXModel(
                    model_path=model_name or "mlx-community/mistral-7b-instruct-v0.2-q4"
                )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        # Get available tools
        self.tools = self._get_tools_for_provider()

    def _get_tools_for_provider(self) -> List[Dict[str, Any]]:
        """Get tools in the format required by the model provider."""
        if self.model_provider == "openai":
            return registry.get_openai_tools(
                category=self.tool_categories[0] if self.tool_categories else None
            )
        elif self.model_provider == "anthropic":
            return registry.get_anthropic_tools(
                category=self.tool_categories[0] if self.tool_categories else None
            )
        else:
            # For now, use OpenAI format for other providers
            return registry.get_openai_tools(
                category=self.tool_categories[0] if self.tool_categories else None
            )

    async def reason_async(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: List[Dict[str, str]] = None,
        image_urls: List[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform reasoning to answer a query asynchronously."""
        # Create conversation history
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant with reasoning capabilities. "
                        "Solve problems step-by-step and use tools when needed."
                    ),
                }
            )

        # Add conversation history
        if history:
            messages.extend(history)

        # Create the query message
        user_content = query
        if context:
            user_content = f"Context:\n{context}\n\nQuery: {query}"

        user_message = {"role": "user", "content": user_content}

        # Add images if provided
        if image_urls:
            user_message["images"] = image_urls

        messages.append(user_message)

        # Initialize reasoning steps
        reasoning_steps = []

        # Track start time for timeout management
        start_time = time.time()

        # Perform reasoning iterations
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Check for timeout
            if timeout and (time.time() - start_time) > timeout:
                reasoning_steps.append(
                    ReasoningStep(
                        type="timeout",
                        content=f"Reasoning timed out after {timeout} seconds ({iteration} iterations completed)",
                        metadata={"iteration": iteration + 1},
                    )
                )
                break

            # Get response from model
            try:
                if hasattr(self.model, "complete_async"):
                    response = await self.model.complete_async(
                        messages=messages,
                        temperature=self.temperature,
                        tools=self.tools,
                    )
                else:
                    # Fallback to synchronous call in an executor
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.model.complete(
                            messages=messages,
                            temperature=self.temperature,
                            tools=self.tools,
                        ),
                    )
            except Exception as e:
                # Handle model error
                reasoning_steps.append(
                    ReasoningStep(
                        type="error",
                        content=f"Model error: {str(e)}",
                        metadata={"iteration": iteration + 1},
                    )
                )

                # Try to recover with a simpler message
                messages.append(
                    {
                        "role": "system",
                        "content": "There was an error. Please provide a simple response without using any tools.",
                    }
                )
                continue

            # Add assistant response to steps
            reasoning_steps.append(
                ReasoningStep(
                    type="thinking",
                    content=response.content,
                    metadata={
                        "iteration": iteration + 1,
                        "model": response.model,
                        "provider": response.provider,
                    },
                )
            )

            # Check if the model wants to use a tool
            tool_calls = response.metadata.get("tool_calls", None)

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )

                # Execute tools and add results
                for tool_call in tool_calls:
                    # Execute tool
                    try:
                        result = await registry.execute_tool_async(
                            tool_call.name, **tool_call.arguments
                        )

                        # Add tool result to steps
                        reasoning_steps.append(
                            ReasoningStep(
                                type="tool",
                                content={
                                    "name": tool_call.name,
                                    "arguments": tool_call.arguments,
                                    "result": result,
                                },
                                metadata={
                                    "iteration": iteration + 1,
                                    "tool_id": tool_call.id,
                                },
                            )
                        )

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result),
                            }
                        )
                    except Exception as e:
                        error_message = (
                            f"Error executing tool {tool_call.name}: {str(e)}"
                        )

                        # Add error to steps
                        reasoning_steps.append(
                            ReasoningStep(
                                type="error",
                                content=error_message,
                                metadata={
                                    "iteration": iteration + 1,
                                    "tool_id": tool_call.id,
                                    "tool_name": tool_call.name,
                                },
                            )
                        )

                        # Add error to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                            }
                        )
            else:
                # No tool calls, just add the response to messages
                messages.append({"role": "assistant", "content": response.content})

                # If no tool calls and we have a response, we're done
                break

        # Get final answer
        if self.verbose:
            logger.info("Getting final answer...")

        final_messages = messages.copy()
        final_messages.append(
            {
                "role": "user",
                "content": "Please provide your final answer based on the reasoning above.",
            }
        )

        try:
            if hasattr(self.model, "complete_async"):
                final_response = await self.model.complete_async(
                    messages=final_messages, temperature=self.temperature
                )
            else:
                # Fallback to synchronous call in an executor
                loop = asyncio.get_event_loop()
                final_response = await loop.run_in_executor(
                    None,
                    lambda: self.model.complete(
                        messages=final_messages, temperature=self.temperature
                    ),
                )
        except Exception as e:
            # Handle error in final response
            reasoning_steps.append(
                ReasoningStep(
                    type="error",
                    content=f"Error generating final answer: {str(e)}",
                    metadata={"final_answer_error": True},
                )
            )

            # Use the last assistant message as the final answer
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    final_content = msg["content"]
                    break
            else:
                final_content = "Unable to generate a final answer due to an error."

            # Create a minimal final response
            final_response = ModelResponse(
                id=f"error-{int(time.time())}",
                model=self.model_name,
                provider=self.model_provider,
                content=final_content,
                usage={},
                metadata={"error": str(e)},
            )

        reasoning_steps.append(
            ReasoningStep(
                type="final_answer",
                content=final_response.content,
                metadata={
                    "model": final_response.model,
                    "provider": final_response.provider,
                },
            )
        )

        # Calculate total duration
        total_duration = time.time() - start_time

        return {
            "query": query,
            "answer": final_response.content,
            "reasoning_steps": [step.dict() for step in reasoning_steps],
            "conversation": messages,
            "metadata": {
                "iterations": len(reasoning_steps),
                "model": final_response.model,
                "provider": final_response.provider,
                "timestamp": datetime.datetime.now().isoformat(),
                "duration_seconds": total_duration,
            },
        }

    def reason(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: List[Dict[str, str]] = None,
        image_urls: List[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform reasoning to answer a query (synchronous version)."""
        # Create an event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async version
        return loop.run_until_complete(
            self.reason_async(
                query=query,
                system_prompt=system_prompt,
                context=context,
                history=history,
                image_urls=image_urls,
                timeout=timeout,
            )
        )


class MultiAgentReasoning:
    """Reasoning system with multiple specialized agents."""

    def __init__(
        self,
        agents_config: List[Dict[str, Any]],
        coordinator_provider: str = "openai",
        coordinator_model: str = None,
        verbose: bool = False,
    ):
        """Initialize the multi-agent reasoning system."""
        self.agents = {}
        self.verbose = verbose

        # Initialize specialized agents
        for config in agents_config:
            agent_id = config.get("id", f"agent_{len(self.agents)}")
            agent = ReasoningAgent(
                model_provider=config.get("provider", "openai"),
                model_name=config.get("model"),
                tool_categories=config.get("tool_categories"),
                max_iterations=config.get("max_iterations", 10),
                verbose=verbose,
                temperature=config.get("temperature", 0.7),
            )
            self.agents[agent_id] = {
                "agent": agent,
                "description": config.get("description", f"Agent {agent_id}"),
                "expertise": config.get("expertise", []),
                "system_prompt": config.get("system_prompt"),
            }

        # Initialize coordinator
        self.coordinator = ReasoningAgent(
            model_provider=coordinator_provider,
            model_name=coordinator_model,
            verbose=verbose,
        )

    def _build_coordinator_prompt(self) -> str:
        """Build a system prompt for the coordinator."""
        agent_descriptions = []
        for agent_id, agent_info in self.agents.items():
            expertise = ", ".join(agent_info["expertise"])
            agent_descriptions.append(
                f"- {agent_id}: {agent_info['description']} (Expertise: {expertise})"
            )

        return (
            "You are a coordinator responsible for delegating tasks to specialized agents. "
            "Based on the user's query, you should decide which agent(s) to assign. "
            "Available agents:\n\n" + "\n".join(agent_descriptions)
        )

    async def reason_async(
        self,
        query: str,
        context: Optional[str] = None,
        history: List[Dict[str, str]] = None,
        image_urls: List[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform multi-agent reasoning to answer a query."""
        start_time = time.time()
        steps = []

        # First, have the coordinator decide which agent(s) to use
        coordinator_prompt = self._build_coordinator_prompt()

        coordinator_result = await self.coordinator.reason_async(
            query=f"Decide which specialized agent(s) should handle this query: {query}",
            system_prompt=coordinator_prompt,
            context=context,
            timeout=timeout,
        )

        steps.append(
            ReasoningStep(
                type="coordination",
                content=coordinator_result["answer"],
                metadata={"coordinator_steps": coordinator_result["reasoning_steps"]},
            )
        )

        # Extract agent IDs from coordinator's response
        selected_agents = []
        for agent_id in self.agents.keys():
            if agent_id.lower() in coordinator_result["answer"].lower():
                selected_agents.append(agent_id)

        if not selected_agents:
            # If no specific agents were selected, use all agents
            selected_agents = list(self.agents.keys())
            steps.append(
                ReasoningStep(
                    type="note",
                    content="No specific agents were identified. Using all available agents.",
                )
            )

        # Query each selected agent in parallel
        agent_tasks = []
        for agent_id in selected_agents:
            agent_info = self.agents[agent_id]
            agent_task = agent_info["agent"].reason_async(
                query=query,
                system_prompt=agent_info["system_prompt"],
                context=context,
                history=history,
                image_urls=image_urls,
                timeout=timeout,
            )
            agent_tasks.append((agent_id, agent_task))

        # Wait for all agents to complete
        agent_results = {}
        for agent_id, task in agent_tasks:
            try:
                result = await task
                agent_results[agent_id] = result
                steps.append(
                    ReasoningStep(
                        type="agent_result",
                        content=result["answer"],
                        metadata={
                            "agent_id": agent_id,
                            "agent_steps": result["reasoning_steps"],
                        },
                    )
                )
            except Exception as e:
                steps.append(
                    ReasoningStep(
                        type="agent_error",
                        content=f"Error from agent {agent_id}: {str(e)}",
                        metadata={"agent_id": agent_id},
                    )
                )

        # Have the coordinator synthesize the final answer
        synthesis_prompt = (
            "You are a coordinator responsible for synthesizing answers from multiple agents. "
            "Based on the agents' responses, create a comprehensive final answer."
        )

        synthesis_context = "Agent responses:\n\n"
        for agent_id, result in agent_results.items():
            synthesis_context += f"=== {agent_id} ===\n{result['answer']}\n\n"

        synthesis_result = await self.coordinator.reason_async(
            query="Synthesize a comprehensive final answer based on the agents' responses.",
            system_prompt=synthesis_prompt,
            context=synthesis_context,
            timeout=timeout,
        )

        steps.append(
            ReasoningStep(
                type="synthesis",
                content=synthesis_result["answer"],
                metadata={"synthesis_steps": synthesis_result["reasoning_steps"]},
            )
        )

        # Calculate total duration
        total_duration = time.time() - start_time

        return {
            "query": query,
            "answer": synthesis_result["answer"],
            "reasoning_steps": [step.dict() for step in steps],
            "agent_results": agent_results,
            "metadata": {
                "duration_seconds": total_duration,
                "selected_agents": selected_agents,
                "timestamp": datetime.datetime.now().isoformat(),
            },
        }

    def reason(
        self,
        query: str,
        context: Optional[str] = None,
        history: List[Dict[str, str]] = None,
        image_urls: List[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform multi-agent reasoning to answer a query (synchronous version)."""
        # Create an event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the async version
        return loop.run_until_complete(
            self.reason_async(
                query=query,
                context=context,
                history=history,
                image_urls=image_urls,
                timeout=timeout,
            )
        )
