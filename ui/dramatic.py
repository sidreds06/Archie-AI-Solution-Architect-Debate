"""Dramatic event rendering with contextual information."""

from rich.console import Console
from rich.rule import Rule

console = Console()

_STYLE_MAP = {
    "LEAD CHANGE": "bold red",
    "SCORE SHIFT": "bold yellow",
    "ADVERSARY CONCEDES": "bold green",
    "EXTRA ROUND GRANTED": "bold yellow",
    "PROPOSER PIVOTS STRATEGY": "bold blue",
    "DEEP DIVE REQUESTED": "bold cyan",
    "NEAR TIE": "bold yellow",
}


def render_dramatic_event(event: str, detail: str = "") -> None:
    """Render a dramatic event with optional context."""
    # Extract base event name for styling (strip trailing !)
    base = event.rstrip("!")
    style = _STYLE_MAP.get(base, "bold white")

    text = event
    if detail:
        text = f"{event} — {detail}"

    console.print(Rule(f"[{style}] {text} [/{style}]", style="dim"))


def render_events_with_context(
    events: list[str],
    proposer_score: float = 0.0,
    adversary_score: float = 0.0,
) -> None:
    """Render dramatic events with score context."""
    for event in events:
        base = event.rstrip("!")
        if base == "LEAD CHANGE":
            if adversary_score > proposer_score:
                detail = f"Adversary takes the lead ({adversary_score:.2f} vs {proposer_score:.2f})"
            else:
                detail = f"Proposer takes the lead ({proposer_score:.2f} vs {adversary_score:.2f})"
        elif base == "SCORE SHIFT":
            detail = f"Proposer {proposer_score:.2f} | Adversary {adversary_score:.2f}"
        elif base == "NEAR TIE":
            detail = f"Only {abs(proposer_score - adversary_score):.2f} apart"
        else:
            detail = ""

        render_dramatic_event(event, detail)


# Backward compat
def render_all_events(events: list[str]) -> None:
    """Render events without score context (legacy)."""
    for event in events:
        render_dramatic_event(event)
