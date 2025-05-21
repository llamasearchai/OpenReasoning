"""
Tool implementations for OpenReasoning agents.
"""

import asyncio
import base64
import datetime
import inspect
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tools that can be used by agents."""

    def __init__(self):
        """Initialize the tool registry."""
        self.tools = {}
        self.categories = {}  # Category -> [tool_names]

    def register(
        self, tool_func: Callable = None, *, category: str = "general"
    ) -> Callable:
        """Register a tool function."""

        def decorator(func):
            # Get function signature
            sig = inspect.signature(func)

            # Create Pydantic model for parameters
            fields = {}
            for name, param in sig.parameters.items():
                # Skip self and other special parameters
                if (
                    name == "self"
                    or param.kind == inspect.Parameter.VAR_POSITIONAL
                    or param.kind == inspect.Parameter.VAR_KEYWORD
                ):
                    continue

                # Get type annotation
                param_type = param.annotation
                if param_type is inspect.Parameter.empty:
                    param_type = Any

                # Get default value if any
                default = (
                    ... if param.default is inspect.Parameter.empty else param.default
                )

                # Add field
                fields[name] = (param_type, Field(default=default))

            # Create Pydantic model for parameters
            param_model = create_model(f"{func.__name__}Params", **fields)

            # Get return type annotation
            return_type = sig.return_annotation
            if return_type is inspect.Parameter.empty:
                return_type = Any

            # Get docstring for description
            description = func.__doc__ or f"Execute {func.__name__}"

            # Register tool
            self.tools[func.__name__] = {
                "func": func,
                "description": description,
                "parameters": param_model,
                "return_type": return_type,
                "category": category,
            }

            # Add to category
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(func.__name__)

            return func

        # Allow usage with or without arguments
        if tool_func is None:
            return decorator
        return decorator(tool_func)

    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get a tool by name."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")

        return self.tools[name]

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a category."""
        if category not in self.categories:
            return []

        return [self.tools[name] for name in self.categories[category]]

    def get_openai_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format."""
        openai_tools = []

        # Filter by category if specified
        tool_names = (
            self.categories.get(category, []) if category else self.tools.keys()
        )

        for name in tool_names:
            tool = self.tools[name]
            param_schema = tool["parameters"].model_json_schema()

            # Format parameters for OpenAI
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": param_schema,
                    },
                }
            )

        return openai_tools

    def get_anthropic_tools(
        self, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get tools in Anthropic format."""
        anthropic_tools = []

        # Filter by category if specified
        tool_names = (
            self.categories.get(category, []) if category else self.tools.keys()
        )

        for name in tool_names:
            tool = self.tools[name]
            param_schema = tool["parameters"].model_json_schema()

            # Format parameters for Anthropic
            anthropic_tools.append(
                {
                    "name": name,
                    "description": tool["description"],
                    "input_schema": param_schema,
                }
            )

        return anthropic_tools

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool with the given parameters."""
        tool = self.get_tool(name)
        return tool["func"](**kwargs)

    async def execute_tool_async(self, name: str, **kwargs) -> Any:
        """Execute a tool asynchronously."""
        tool = self.get_tool(name)

        # Check if the function is async
        if inspect.iscoroutinefunction(tool["func"]):
            # If async, await it directly
            return await tool["func"](**kwargs)
        else:
            # If not async, run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool["func"](**kwargs))


# Create tool registry instance
registry = ToolRegistry()


# Define some built-in tools
@registry.register(category="web")
def search_web(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Search the web for information."""
    # This is a stub implementation - in a real system, you'd use a search API
    try:
        # Simulate search latency
        time.sleep(0.5)

        return {
            "results": [
                {
                    "title": f"Result {i+1} for {query}",
                    "snippet": f"This is a simulated search result {i+1} for '{query}'.",
                    "url": f"https://example.com/search?q={quote(query)}&result={i+1}",
                }
                for i in range(num_results)
            ],
            "metadata": {
                "timestamp": str(datetime.datetime.now()),
                "query": query,
                "total_results": num_results,
            },
        }
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "timestamp": str(datetime.datetime.now()),
        }


@registry.register(category="weather")
def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get the current weather for a location."""
    # This is a stub implementation - in a real system, you'd use a weather API
    try:
        # Simulate API latency
        time.sleep(0.5)

        # Generate fake weather data based on location string
        import hashlib

        location_hash = int(hashlib.md5(location.encode()).hexdigest(), 16) % 100

        # Generate temperature based on hash (between -15 and 45 degrees Celsius)
        temp_celsius = (location_hash % 60) - 15

        # Convert to Fahrenheit if requested
        temp = temp_celsius if unit == "celsius" else (temp_celsius * 9 / 5) + 32

        # Generate other weather metrics
        humidity = (location_hash % 60) + 40  # 40-99%

        # Weather conditions based on hash
        conditions = [
            "sunny",
            "partly cloudy",
            "cloudy",
            "rainy",
            "stormy",
            "snowy",
            "foggy",
            "windy",
        ]
        condition = conditions[location_hash % len(conditions)]

        return {
            "location": location,
            "temperature": round(temp, 1),
            "unit": unit,
            "condition": condition,
            "humidity": humidity,
            "wind_speed": round((location_hash % 30) + 5, 1),
            "timestamp": str(datetime.datetime.now()),
        }
    except Exception as e:
        return {
            "error": str(e),
            "location": location,
            "timestamp": str(datetime.datetime.now()),
        }


@registry.register(category="math")
def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression safely."""
    try:
        # Use eval with a restricted namespace for safety
        allowed_names = {
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            "int": int,
            "float": float,
            "pow": pow,
            "divmod": divmod,
        }

        # Add math functions
        import math

        for name in dir(math):
            if not name.startswith("_"):
                allowed_names[name] = getattr(math, name)

        # Evaluate expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)

        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }
    except Exception as e:
        return {"expression": expression, "error": str(e), "type": "error"}


@registry.register(category="date_time")
def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """Get the current date and time in the specified timezone."""
    try:
        from datetime import datetime

        import pytz

        # Get timezone
        tz = pytz.timezone(timezone)

        # Get current time in timezone
        now = datetime.now(tz)

        return {
            "timezone": timezone,
            "datetime": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat(),
            "timestamp": now.timestamp(),
            "day_of_week": now.strftime("%A"),
            "month": now.strftime("%B"),
        }
    except ImportError:
        # Fallback if pytz is not available
        now = datetime.datetime.now()
        return {
            "timezone": "local",
            "datetime": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat(),
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"error": str(e), "timezone": timezone}


@registry.register(category="file_system")
def read_file(file_path: str, max_chars: int = 10000) -> Dict[str, Any]:
    """Read the contents of a file."""
    try:
        import os

        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "file_path": file_path}

        # Read file content
        with open(file_path, "r") as f:
            content = f.read(max_chars)

        # Check if truncated
        is_truncated = os.path.getsize(file_path) > max_chars

        return {
            "file_path": file_path,
            "content": content,
            "truncated": is_truncated,
            "size_bytes": os.path.getsize(file_path),
            "last_modified": datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "file_path": file_path}


@registry.register(category="web")
def fetch_url(url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Fetch the content of a URL."""
    try:
        response = requests.get(url, headers=headers, timeout=10)

        # Check if response is successful
        response.raise_for_status()

        # Get content type
        content_type = response.headers.get("Content-Type", "")

        # Handle different content types
        if "application/json" in content_type:
            # Parse JSON
            try:
                content = response.json()
                return {
                    "url": url,
                    "content": content,
                    "content_type": content_type,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                }
            except Exception as e:
                # Fall back to text if JSON parsing fails
                content = response.text
        else:
            # Return text content
            content = response.text

        return {
            "url": url,
            "content": content[:10000],  # Limit content size
            "content_type": content_type,
            "status_code": response.status_code,
            "truncated": len(response.text) > 10000,
            "response_time": response.elapsed.total_seconds(),
        }
    except Exception as e:
        return {"error": str(e), "url": url}
