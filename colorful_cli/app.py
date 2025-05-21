"""
Colorful CLI application for OpenReasoning.
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

import typer
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Log, Markdown, Static

from openreasoning import __version__
from openreasoning.core.config import settings
from openreasoning.models.base import ModelInput
from openreasoning.models.providers import PROVIDERS
from openreasoning.utils.m3_optimizer import m3_optimizer


class ModelCompletionPanel(Static):
    """A panel that displays model completions."""

    def __init__(self, provider_name: str, model_name: str):
        super().__init__()
        self.provider_name = provider_name
        self.model_name = model_name
        self.history: List[Dict[str, str]] = []

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Container(
            Header(f"{self.provider_name.upper()} - {self.model_name}"),
            Markdown("*Ready for input*", classes="model-output"),
            classes="output-container",
        )

    async def submit_prompt(self, prompt: str):
        """Submit a prompt to the model and display the response."""
        self.history.append({"role": "user", "content": prompt})

        # Update UI to show the user prompt
        output = self.query_one(".model-output", Markdown)
        markdown_content = self._format_history()
        output.update(markdown_content)

        # Show loading state
        self.add_class("loading")
        output.update(markdown_content + "\n\n*Generating response...*")

        # Get the provider and generate response
        provider_class = PROVIDERS[self.provider_name]
        provider = provider_class(model=self.model_name)

        try:
            input_data = ModelInput(
                prompt=self.history,
                temperature=0.7,
                max_tokens=1024,
            )

            response = provider.generate(input_data)
            self.history.append({"role": "assistant", "content": response.text})

            # Update UI with response
            markdown_content = self._format_history()
            output.update(markdown_content)

            # Add token usage to log
            self.app.query_one(Log).write(
                f"[{self.provider_name}] Used {response.usage['prompt_tokens']} prompt tokens, "
                f"{response.usage['completion_tokens']} completion tokens"
            )

        except Exception as e:
            # Show error
            self.app.query_one(Log).write(f"[red]Error: {e}[/red]")
            output.update(markdown_content + f"\n\n*Error: {e}*")

        finally:
            # Remove loading state
            self.remove_class("loading")

    def _format_history(self) -> str:
        """Format the conversation history as markdown."""
        markdown = ""
        for msg in self.history:
            if msg["role"] == "user":
                markdown += f"\n\n**You:** {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                markdown += f"**Assistant:** {msg['content']}\n\n"
            elif msg["role"] == "system":
                markdown += f"*System: {msg['content']}*\n\n"
        return markdown


class SidebarMenu(Container):
    """Sidebar menu with settings and model selection."""

    def compose(self) -> ComposeResult:
        """Compose the sidebar layout."""
        yield Header("Settings")

        yield Container(
            Markdown("## Providers"),
            id="providers-container",
        )

        # Add provider buttons for each available provider
        providers_container = self.query_one("#providers-container", Container)

        for provider_name in PROVIDERS.keys():
            button = Button(
                f"{provider_name}",
                id=f"provider-{provider_name}",
                classes="provider-button",
            )
            providers_container.mount(button)

        # Add optimization status if on M3
        if settings.is_m3_chip:
            yield Container(
                Markdown("## M3 Optimizations"),
                Button("Apply Optimizations", id="apply-m3-optimizations"),
                Markdown(
                    f"Status: {'Active' if m3_optimizer.optimization_status['optimizations_applied'] else 'Inactive'}"
                ),
                id="m3-container",
            )

        yield Container(
            Markdown("## About"),
            Markdown(f"OpenReasoning v{__version__}"),
            Markdown("Advanced Multimodal AI Reasoning Framework"),
            id="about-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "apply-m3-optimizations":
            applied = m3_optimizer.apply_optimizations()
            status = "Active" if applied else "Failed to apply"

            # Update the status text
            m3_container = self.query_one("#m3-container", Container)
            m3_container.query(Markdown)[-1].update(f"Status: {status}")

            # Log the result
            self.app.query_one(Log).write(f"M3 optimizations: {status}")

        elif button_id and button_id.startswith("provider-"):
            provider_name = button_id.replace("provider-", "")

            # Set active provider by adding a class to the button
            for btn in self.query(".provider-button"):
                btn.remove_class("active")

            event.button.add_class("active")

            # Update the app's active provider
            self.app.set_active_provider(provider_name)


class OpenReasoningApp(App):
    """Main OpenReasoning colorful CLI app."""

    CSS_PATH = "style.css"
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+s", "toggle_sidebar", "Toggle Sidebar"),
    ]

    def __init__(self):
        super().__init__()
        self.active_provider = "openai"  # Default provider
        self.active_model = "gpt-4o"  # Default model

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()

        yield Container(
            SidebarMenu(id="sidebar"),
            Container(
                Container(
                    ModelCompletionPanel(self.active_provider, self.active_model),
                    id="main-panel",
                ),
                Container(
                    Input(placeholder="Enter your prompt here...", id="prompt-input"),
                    Button("Send", id="send-button"),
                    id="input-container",
                ),
                id="main-container",
            ),
            id="app-grid",
        )

        yield Container(
            Log(id="status-log", highlight=True, markup=True), id="log-container"
        )

        yield Footer()

    def on_mount(self) -> None:
        """Handle the mount event."""
        # Add welcome message to log
        self.query_one(Log).write("Welcome to OpenReasoning Colorful CLI!")

        # Check for API keys
        providers_with_keys = []
        for provider_name in PROVIDERS.keys():
            provider_class = PROVIDERS[provider_name]
            try:
                provider = provider_class()
                if provider.api_key:
                    providers_with_keys.append(provider_name)
                    self.query_one(Log).write(f"✅ {provider_name} API key detected")
            except:
                pass

        if not providers_with_keys:
            self.query_one(Log).write(
                "[red]⚠️ No API keys found! Please set at least one provider API key.[/red]"
            )
        else:
            # Use first available provider
            self.active_provider = providers_with_keys[0]

            # Update UI to reflect the active provider
            provider_btn = self.query_one(f"#provider-{self.active_provider}", Button)
            provider_btn.add_class("active")

            # Replace the model panel with the active provider
            main_panel = self.query_one("#main-panel")
            main_panel.remove_children()
            main_panel.mount(
                ModelCompletionPanel(self.active_provider, self.active_model)
            )

        # Apply M3 optimizations if available
        if settings.is_m3_chip:
            self.query_one(Log).write("🚀 M3 chip detected")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-button":
            self._handle_send()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submit events."""
        if event.input.id == "prompt-input":
            self._handle_send()

    def _handle_send(self) -> None:
        """Handle sending a prompt to the model."""
        prompt_input = self.query_one("#prompt-input", Input)
        prompt = prompt_input.value

        if prompt.strip():
            # Clear input
            prompt_input.value = ""

            # Get the active model panel and submit the prompt
            panel = self.query_one(ModelCompletionPanel)
            asyncio.create_task(panel.submit_prompt(prompt))

    def set_active_provider(self, provider_name: str) -> None:
        """Set the active model provider."""
        if provider_name not in PROVIDERS:
            self.query_one(Log).write(
                f"[red]Provider {provider_name} not available[/red]"
            )
            return

        self.active_provider = provider_name

        # Set default model for the provider
        provider_class = PROVIDERS[provider_name]
        provider = provider_class()

        models = provider.get_available_models()
        if models:
            self.active_model = models[0]

        # Replace the model panel with the new provider
        main_panel = self.query_one("#main-panel")
        main_panel.remove_children()
        main_panel.mount(ModelCompletionPanel(self.active_provider, self.active_model))

        self.query_one(Log).write(
            f"Switched to {provider_name} with model {self.active_model}"
        )

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        sidebar = self.query_one("#sidebar")
        sidebar.toggle_class("hidden")


def run_app():
    """Run the colorful CLI app."""
    app = OpenReasoningApp()
    app.run()


if __name__ == "__main__":
    run_app()
