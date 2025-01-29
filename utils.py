from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from typing import Any, Dict, Optional

def get_user_question(console: Optional[Console] = None) -> str:
    console = console or Console()
    question = Prompt.ask("[bold green]Your Question[/bold green]")
    
    console.print(f"\n[bold magenta]You asked:[/bold magenta] [yellow]{question}[/yellow]")
    
    return question

def print_in_box(text: str, console: Optional[Console] = None, title: str = "", color: str = "white") -> None:
    console = console or Console()
    panel = Panel(text, title=title, border_style=color, expand=True)
    console.print(panel)

from typing import List, Dict

def format_similar_doc(docs: List[Dict]) -> str:
    formatted_docs = []
    
    for i, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "Untitled")
        content = metadata.get("content", "")
        
        formatted_docs.append(
            f"[green]{i + 1}.[/green] {title} \n[yellow]{content}[/yellow]"
        )
    
    return "\n\n".join(formatted_docs)

