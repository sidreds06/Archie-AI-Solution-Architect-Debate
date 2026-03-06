"""Rich CLI banners and visual helpers for phase transitions and scoring."""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()


def round_header(current_round: int, max_rounds: int) -> None:
    """Print a bold round header."""
    console.print()
    console.print(
        Rule(
            f"[bold white] ROUND {current_round} of {max_rounds} [/bold white]",
            style="bright_white",
        )
    )
    console.print()


def phase_banner(agent_name: str, action: str, color: str) -> None:
    """Print a subtle phase transition line before an agent speaks."""
    console.print(f"\n[{color}]{agent_name}[/{color}] [dim]{action}[/dim]\n")


def debate_start_banner() -> None:
    """Print a clear separator when the debate begins."""
    console.print()
    console.print(Rule("[bold white] DEBATE BEGINS [/bold white]", style="bright_white"))
    console.print()


def score_comparison(
    verdict: dict, current_round: int, model: str
) -> None:
    """Display moderator scoring as a structured, scannable output."""
    scores = verdict.get("scores", {})
    decision = verdict.get("decision", "continue")
    reasoning = verdict.get("reasoning", "")
    winner = verdict.get("winner", "TBD")

    p_total = scores.get("proposer", {}).get("weighted_total", 0.0)
    a_total = scores.get("adversary", {}).get("weighted_total", 0.0)

    # Score comparison table
    table = Table(
        title=f"[bold magenta]Moderator Scoring — Round {current_round}[/bold magenta]",
        border_style="magenta",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Agent", style="white", width=12)
    table.add_column("Score", justify="center", width=8)
    table.add_column("", width=20)

    # Visual bar
    p_bar = _score_bar(p_total, "blue")
    a_bar = _score_bar(a_total, "green")

    table.add_row("[blue]Proposer[/blue]", f"[bold blue]{p_total:.2f}[/bold blue]", p_bar)
    table.add_row("[green]Adversary[/green]", f"[bold green]{a_total:.2f}[/bold green]", a_bar)

    console.print()
    console.print(table)

    # Decision line
    decision_colors = {"continue": "cyan", "end": "red", "hitl": "yellow"}
    dc = decision_colors.get(decision, "white")
    winner_str = f"  Leader: [bold]{winner}[/bold]" if winner and winner != "TBD" else ""
    console.print(
        f"  Decision: [{dc}][bold]{decision.upper()}[/bold][/{dc}]{winner_str}"
    )

    # Reasoning
    if reasoning:
        console.print(f"  [dim italic]{reasoning}[/dim italic]")
    console.print()


def _score_bar(score: float, color: str, max_score: float = 5.0, width: int = 16) -> str:
    """Generate a simple text-based score bar."""
    filled = int((score / max_score) * width)
    filled = max(0, min(filled, width))
    empty = width - filled
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"



def verdict_banner(winner: str, p_score: float, a_score: float) -> None:
    """Print a dramatic winner announcement before the verdict brief."""
    console.print()
    console.print(Rule(style="bright_white"))

    winner_label = "PROPOSER" if winner == "proposer" else "ADVERSARY"
    winner_color = "blue" if winner == "proposer" else "green"

    console.print(
        Panel(
            f"[bold {winner_color}]{winner_label} WINS[/bold {winner_color}]\n\n"
            f"  [blue]Proposer:[/blue]  {p_score:.2f}    [green]Adversary:[/green] {a_score:.2f}",
            border_style=winner_color,
            title="[bold white]FINAL RESULT[/bold white]",
            padding=(1, 4),
        )
    )
    console.print()


def session_footer(rounds_played: int, memory_updated: bool) -> None:
    """Print a session-end footer."""
    mem_note = "Memory updated." if memory_updated else ""
    console.print()
    console.print(
        Rule(
            f"[dim] {rounds_played} round{'s' if rounds_played != 1 else ''} played  {mem_note} [/dim]",
            style="dim",
        )
    )
    console.print()
