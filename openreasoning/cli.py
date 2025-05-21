"""
Command-line interface for OpenReasoning.
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.layout import Layout
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from openreasoning import __version__
from openreasoning.models.providers import PROVIDERS

from .agents.reasoning import MultiAgentReasoning, ReasoningAgent
from .core.config import settings
from .core.logging import configure_logging
from .models.anthropic import AnthropicModel
from .models.mlx_optimized import MLXModel
from .models.openai import OpenAIModel
from .utils.m3_optimizer import m3_optimizer

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="openreasoning",
    help="OpenReasoning: Advanced Multimodal AI Reasoning Framework",
    add_completion=False,
)
console = Console()


@app.callback()
def callback():
    """OpenReasoning CLI"""
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)


@app.command()
def version():
    """Display version information."""
    console.print(f"OpenReasoning v{__version__}")


@app.command()
def optimize():
    """Detect and apply optimizations for the current hardware."""
    console.print(Panel("🚀 [bold]Hardware Optimization[/bold]", expand=False))

    if settings.is_apple_silicon:
        console.print("✅ Apple Silicon detected")

        if settings.is_m3_chip:
            console.print("✅ M3 chip detected")
            console.print("Applying M3-specific optimizations...")

            result = m3_optimizer.apply_optimizations()
            if result:
                console.print("[green]✅ M3 optimizations successfully applied[/green]")
            else:
                console.print("[yellow]⚠️ No optimizations were applied[/yellow]")

            # Print optimization status
            status = m3_optimizer.get_optimization_status()
            console.print("\n[bold]Optimization Status:[/bold]")
            for key, value in status.items():
                if isinstance(value, dict):
                    console.print(f"  [cyan]{key}[/cyan]:")
                    for subkey, subvalue in value.items():
                        console.print(f"    {subkey}: {subvalue}")
                else:
                    console.print(f"  [cyan]{key}[/cyan]: {value}")
        else:
            console.print(
                "[yellow]⚠️ M3 chip not detected. Using standard optimizations.[/yellow]"
            )
    else:
        console.print(
            "[yellow]⚠️ Apple Silicon not detected. No specific optimizations available.[/yellow]"
        )


@app.command()
def check():
    """Check environment and dependencies."""
    console.print(Panel("🔍 [bold]Environment Check[/bold]", expand=False))

    # Check Python version
    py_version = sys.version.split()[0]
    console.print(f"Python version: {py_version}")

    # Check API keys
    console.print("\n[bold]API Keys:[/bold]")
    providers = {
        "OpenAI": settings.openai_api_key,
        "Anthropic": settings.anthropic_api_key,
        "HuggingFace": settings.huggingface_api_key,
    }

    for provider, api_key in providers.items():
        if api_key:
            console.print(f"✅ {provider} API key: [green]Set[/green]")
        else:
            console.print(f"❌ {provider} API key: [red]Not set[/red]")

    # Check model providers
    console.print("\n[bold]Model Providers:[/bold]")
    for provider_name, provider_class in PROVIDERS.items():
        try:
            provider_instance = provider_class()
            console.print(f"✅ {provider_name}: [green]Available[/green]")
        except Exception as e:
            console.print(f"❌ {provider_name}: [red]Not available[/red] ({e})")

    # Hardware detection
    console.print("\n[bold]Hardware Detection:[/bold]")
    if settings.is_apple_silicon:
        console.print("✅ Apple Silicon: [green]Detected[/green]")
        if settings.is_m3_chip:
            console.print("✅ M3 chip: [green]Detected[/green]")
        else:
            console.print("❌ M3 chip: [yellow]Not detected[/yellow]")
    else:
        console.print("❌ Apple Silicon: [yellow]Not detected[/yellow]")

    # Check optional dependencies
    console.print("\n[bold]Optional Dependencies:[/bold]")
    try:
        import mlx

        console.print("✅ MLX: [green]Installed[/green]")
    except ImportError:
        console.print("❌ MLX: [yellow]Not installed[/yellow]")

    try:
        import torch

        console.print("✅ PyTorch: [green]Installed[/green]")
    except ImportError:
        console.print("❌ PyTorch: [yellow]Not installed[/yellow]")


@app.command()
def server(
    host: str = typer.Option("127.0.0.1", help="API server host"),
    port: int = typer.Option(8000, help="API server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the API server."""
    try:
        import uvicorn

        from openreasoning.api.server import app as api_app
    except ImportError:
        console.print(
            "[red]Error: FastAPI and uvicorn are required to run the server.[/red]"
        )
        console.print("Install them with: pip install fastapi uvicorn")
        return

    console.print(
        Panel(
            f"🚀 [bold]Starting OpenReasoning API Server on {host}:{port}[/bold]",
            expand=False,
        )
    )
    uvicorn.run("openreasoning.api.server:app", host=host, port=port, reload=reload)


@app.command()
def chat(
    model: str = typer.Option("gpt-4o", help="Model to use for chat"),
    provider: str = typer.Option(
        "openai", help="Provider to use for chat (openai, anthropic, huggingface)"
    ),
    temperature: float = typer.Option(0.7, help="Temperature for text generation"),
    multimodal: bool = typer.Option(
        False, help="Enable multimodal chat with image support"
    ),
):
    """Start an interactive chat session."""
    console.print(Panel("💬 [bold]OpenReasoning Chat[/bold]", expand=False))
    console.print(
        f"Using model: [cyan]{model}[/cyan] from provider: [cyan]{provider}[/cyan]"
    )

    if provider not in PROVIDERS:
        console.print(
            f"[red]Error: Provider '{provider}' not found. Available providers: {', '.join(PROVIDERS.keys())}[/red]"
        )
        return

    # Check for API key
    provider_class = PROVIDERS[provider]
    provider_instance = provider_class(model=model)

    if not provider_instance.api_key:
        console.print(f"[red]Error: No API key found for {provider}.[/red]")
        console.print(
            f"Please set the {provider.upper()}_API_KEY environment variable."
        )
        return

    # Validate API key
    console.print("Validating API key...")
    if not provider_instance.validate_api_key():
        console.print(f"[red]Error: Invalid API key for {provider}.[/red]")
        return

    console.print("[green]API key validated successfully.[/green]")

    # Start chat loop
    console.print(
        "\n[bold]Starting chat session. Type 'exit' or 'quit' to end the session.[/bold]"
    )
    console.print("[dim]Type your message and press Enter to send.[/dim]\n")

    history = []

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ")

        if user_input.lower() in ["exit", "quit", "q"]:
            console.print("[bold]Ending chat session.[/bold]")
            break

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Create model input
        from openreasoning.models.base import ModelInput

        input_data = ModelInput(
            prompt=history,
            temperature=temperature,
            max_tokens=1024,
        )

        try:
            with console.status("[bold green]Thinking...[/bold green]"):
                response = provider_instance.generate(input_data)

            # Display response
            console.print(f"[bold green]Assistant:[/bold green] {response.text}")

            # Add assistant response to history
            history.append({"role": "assistant", "content": response.text})

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def demo():
    """Run a quick demo of the OpenReasoning framework."""
    console.print(Panel("✨ [bold]OpenReasoning Demo[/bold]", expand=False))

    # Check for at least one API key
    if not settings.openai_api_key and not settings.anthropic_api_key:
        console.print(
            "[red]Error: No API keys found. At least one is required for the demo.[/red]"
        )
        console.print(
            "Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables."
        )
        return

    # Determine which provider to use
    if settings.openai_api_key:
        provider_name = "openai"
        provider_class = PROVIDERS["openai"]
        model = "gpt-4o"
    elif settings.anthropic_api_key:
        provider_name = "anthropic"
        provider_class = PROVIDERS["anthropic"]
        model = "claude-3-haiku-20240307"
    else:
        console.print("[red]No API keys available for demo.[/red]")
        return

    console.print(
        f"Using [cyan]{provider_name}[/cyan] with model [cyan]{model}[/cyan] for demo."
    )

    # Initialize provider
    provider = provider_class(model=model)

    # Show optimization status
    if settings.is_apple_silicon and settings.is_m3_chip:
        console.print("\n[bold]M3 Optimizations:[/bold]")
        m3_optimizer.apply_optimizations()
        status = m3_optimizer.get_optimization_status()
        console.print(f"Applied: {status['optimizations_applied']}")

    # Run a simple demo query
    console.print("\n[bold]Demo Query:[/bold]")

    demo_query = "Explain the concept of retrieval-augmented generation in 3 sentences."
    console.print(f"Query: [italic]{demo_query}[/italic]")

    from openreasoning.models.base import ModelInput

    input_data = ModelInput(
        prompt=demo_query,
        temperature=0.7,
        max_tokens=150,
    )

    try:
        with console.status("[bold green]Generating response...[/bold green]"):
            response = provider.generate(input_data)

        console.print("\n[bold]Response:[/bold]")
        console.print(f"[green]{response.text}[/green]")

        console.print("\n[bold]Usage:[/bold]")
        for key, value in response.usage.items():
            console.print(f"{key}: {value}")

    except Exception as e:
        console.print(f"[red]Error during demo: {e}[/red]")

    console.print("\n[bold]Demo completed.[/bold]")


if __name__ == "__main__":
    app()
