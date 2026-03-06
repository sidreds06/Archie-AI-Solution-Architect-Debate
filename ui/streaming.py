import re

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

console = Console()

# Pattern to match raw tool call JSON that some models emit as text
_TOOL_JSON_RE = re.compile(r'\{"tool_uses"\s*:\s*\[.*?\]\}', re.DOTALL)


def _strip_tool_json(text: str) -> str:
    """Remove raw tool call JSON blocks from text content."""
    return _TOOL_JSON_RE.sub("", text).strip()


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
            full_text = _strip_tool_json("".join(chunks))
            live.update(Panel(full_text, title=title, border_style=border_style))

        # Final update with done indicator in title
        full_text = _strip_tool_json("".join(chunks))
        done_title = f"{title} [dim]\\[done][/dim]"
        live.update(Panel(full_text, title=done_title, border_style=border_style))

    return _strip_tool_json("".join(chunks))
