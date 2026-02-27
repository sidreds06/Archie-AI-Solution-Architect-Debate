from rich.console import Console

console = Console()

_STYLE_MAP = {
    "SCORE SHIFT!": "bold red",
    "LEAD CHANGE!": "bold red on white",
    "ADVERSARY CONCEDES POINT": "bold green",
    "ADVERSARY CONCEDES!": "bold green",
    "EXTRA ROUND GRANTED!": "bold yellow",
    "PROPOSER PIVOTS STRATEGY!": "bold blue",
    "DEEP DIVE REQUESTED!": "bold cyan",
    "NEAR TIE!": "bold yellow",
}


def render_dramatic_event(event: str) -> None:
    style = _STYLE_MAP.get(event, "bold white")
    console.print(f"\n  >>> {event} <<<\n", style=style)


def render_all_events(events: list[str]) -> None:
    for event in events:
        render_dramatic_event(event)
