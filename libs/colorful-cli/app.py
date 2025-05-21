#!/usr/bin/env python3
import os
import json
from typing import Optional, List
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

app = typer.Typer()
console = Console()

# Data storage
DATA_FILE = os.path.expanduser("~/.tasks.json")

def load_tasks():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"tasks": []}
    return {"tasks": []}

def save_tasks(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.command()
def add(description: str, priority: str = "medium"):
    """Add a new task with a description and priority."""
    data = load_tasks()
    
    # Validate priority
    if priority.lower() not in ["high", "medium", "low"]:
        console.print("[bold red]Error:[/] Priority must be 'high', 'medium', or 'low'")
        return
    
    task = {
        "id": len(data["tasks"]) + 1,
        "description": description,
        "priority": priority.lower(),
        "completed": False,
        "created_at": datetime.now().isoformat()
    }
    
    data["tasks"].append(task)
    save_tasks(data)
    
    console.print(
        Panel(
            f"[bold green]Task added successfully![/]\n\n"
            f"[bold]ID:[/] {task['id']}\n"
            f"[bold]Description:[/] {task['description']}\n"
            f"[bold]Priority:[/] {get_priority_text(task['priority'])}", 
            title="New Task",
            border_style="green"
        )
    )

def get_priority_text(priority):
    if priority == "high":
        return "[bold red]HIGH[/]"
    elif priority == "medium":
        return "[bold yellow]MEDIUM[/]"
    else:
        return "[bold green]LOW[/]"

@app.command()
def list(filter: Optional[str] = None):
    """List all tasks with colorful formatting."""
    data = load_tasks()
    tasks = data["tasks"]
    
    if filter:
        if filter.lower() in ["high", "medium", "low"]:
            tasks = [t for t in tasks if t["priority"] == filter.lower()]
        elif filter.lower() in ["completed", "done"]:
            tasks = [t for t in tasks if t["completed"]]
        elif filter.lower() in ["pending", "todo"]:
            tasks = [t for t in tasks if not t["completed"]]
    
    if not tasks:
        console.print("[yellow]No tasks found![/]")
        return
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=6)
    table.add_column("Description", min_width=20)
    table.add_column("Priority", width=10)
    table.add_column("Status", width=10)
    table.add_column("Created", width=20)
    
    for task in tasks:
        status = "[green]✓[/]" if task["completed"] else "[red]✗[/]"
        created = datetime.fromisoformat(task["created_at"]).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            str(task["id"]),
            task["description"],
            get_priority_text(task["priority"]),
            status,
            created
        )
    
    console.print(Panel(table, title="[bold blue]Task List[/]", border_style="blue"))

@app.command()
def complete(id: int):
    """Mark a task as completed."""
    data = load_tasks()
    
    for task in data["tasks"]:
        if task["id"] == id:
            task["completed"] = True
            save_tasks(data)
            console.print(f"[bold green]Task {id} marked as completed! ✓[/]")
            return
    
    console.print(f"[bold red]Task with ID {id} not found![/]")

@app.command()
def delete(id: int):
    """Delete a task by ID."""
    data = load_tasks()
    
    for i, task in enumerate(data["tasks"]):
        if task["id"] == id:
            del data["tasks"][i]
            save_tasks(data)
            console.print(f"[bold yellow]Task {id} deleted! 🗑️[/]")
            return
    
    console.print(f"[bold red]Task with ID {id} not found![/]")

@app.command()
def clear():
    """Clear all tasks."""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        console.print("[bold red]All tasks cleared! 🧹[/]")
    else:
        console.print("[yellow]No tasks to clear![/]")

@app.callback()
def main():
    """
    ✨ Colorful Task Manager CLI ✨
    
    A beautiful command-line task manager with colorful output.
    """
    console.print(
        Panel(
            "[bold cyan]Welcome to Colorful Task Manager![/]\n"
            "Type [bold]--help[/] after any command for more information.",
            border_style="cyan"
        )
    )

if __name__ == "__main__":
    app()