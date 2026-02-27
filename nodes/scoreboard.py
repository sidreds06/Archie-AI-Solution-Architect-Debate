from rich.console import Console
from rich.table import Table

from state import DebateState
from ui.dramatic import render_all_events

console = Console()


def _detect_events(
    score_history: list[dict],
    proposer_score: float,
    adversary_score: float,
) -> list[str]:
    """Detect dramatic moments based on score changes."""
    events: list[str] = []
    delta = abs(proposer_score - adversary_score)

    if len(score_history) >= 1:
        prev = score_history[-1]
        prev_p = prev.get("proposer_score", 0)
        prev_a = prev.get("adversary_score", 0)

        # Lead change
        if (prev_p > prev_a and adversary_score > proposer_score) or (
            prev_a > prev_p and proposer_score > adversary_score
        ):
            events.append("LEAD CHANGE!")

        # Big score shift (> 0.5 swing for either agent)
        p_swing = abs(proposer_score - prev_p)
        a_swing = abs(adversary_score - prev_a)
        if p_swing > 0.5 or a_swing > 0.5:
            events.append("SCORE SHIFT!")

    # Near tie
    if delta < 0.15:
        events.append("NEAR TIE!")

    return events


def _compute_momentum(
    score_history: list[dict],
    proposer_score: float,
    adversary_score: float,
) -> dict:
    """Compute momentum based on score trends."""
    if not score_history:
        return {"proposer": proposer_score, "adversary": adversary_score}
    prev = score_history[-1]
    return {
        "proposer": proposer_score - prev.get("proposer_score", 0),
        "adversary": adversary_score - prev.get("adversary_score", 0),
    }


def update_scoreboard(state: DebateState) -> dict:
    """LangGraph node. Renders scoreboard table and detects dramatic events."""
    verdict_data = state.get("final_verdict") or {}
    scores = verdict_data.get("scores", {})
    current_round = state["round"]
    score_history = state.get("score_history", [])

    proposer_score = scores.get("proposer", {}).get("weighted_total", 0.0)
    adversary_score = scores.get("adversary", {}).get("weighted_total", 0.0)
    delta = proposer_score - adversary_score

    # Detect dramatic events
    events = _detect_events(score_history, proposer_score, adversary_score)

    # Build new score entry
    new_entry = {
        "round": current_round,
        "proposer_score": proposer_score,
        "adversary_score": adversary_score,
        "delta": delta,
    }
    updated_history = score_history + [new_entry]

    # Render the scoreboard table
    table = Table(
        title="[bold white]SCOREBOARD[/bold white]",
        border_style="bright_white",
        show_lines=True,
    )
    table.add_column("Round", style="white", justify="center", width=7)
    table.add_column("Proposer", style="blue", justify="center", width=12)
    table.add_column("Adversary", style="green", justify="center", width=12)
    table.add_column("Delta", style="yellow", justify="center", width=10)
    table.add_column("Trend", style="white", justify="center", width=7)

    for i, entry in enumerate(updated_history):
        # Determine trend arrow
        if i == 0:
            trend = "—"
        else:
            prev_delta = updated_history[i - 1]["delta"]
            curr_delta = entry["delta"]
            if curr_delta > prev_delta + 0.1:
                trend = "[blue]▲ P[/blue]"
            elif curr_delta < prev_delta - 0.1:
                trend = "[green]▲ A[/green]"
            else:
                trend = "[yellow]=[/yellow]"

        delta_str = f"{entry['delta']:+.2f}"
        if entry["delta"] > 0:
            delta_str = f"[blue]{delta_str}[/blue]"
        elif entry["delta"] < 0:
            delta_str = f"[green]{delta_str}[/green]"

        table.add_row(
            str(entry["round"]),
            f"{entry['proposer_score']:.2f}",
            f"{entry['adversary_score']:.2f}",
            delta_str,
            trend,
        )

    console.print()
    console.print(table)

    # Render dramatic events
    if events:
        render_all_events(events)

    # Compute momentum for next round's prompts
    momentum = _compute_momentum(score_history, proposer_score, adversary_score)

    return {
        "score_history": updated_history,
        "momentum": momentum,
        "dramatic_events": events,
    }
