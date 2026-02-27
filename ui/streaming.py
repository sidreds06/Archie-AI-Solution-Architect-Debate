from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

console = Console()


def stream_to_panel(
    client: ChatOpenAI,
    messages: list[BaseMessage],
    title: str,
    border_style: str,
) -> str:
    """Stream LLM tokens into a Rich Live panel. Returns the full response text."""
    chunks: list[str] = []
    with Live(
        Panel("", title=title, border_style=border_style),
        console=console,
        refresh_per_second=12,
    ) as live:
        for chunk in client.stream(messages):
            token = chunk.content or ""
            chunks.append(token)
            full_text = "".join(chunks)
            live.update(Panel(full_text, title=title, border_style=border_style))
    return "".join(chunks)
