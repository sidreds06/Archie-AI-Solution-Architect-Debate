from rich.console import Console
from rich.prompt import Prompt

from state import DebateState

console = Console()


def round_increment(state: DebateState) -> dict:
    """LangGraph node. Advances round counter and offers user a chance to inject direction."""
    new_round = state["round"] + 1
    max_rounds = state["max_rounds"]

    console.print(
        f"\n[dim]── Round {new_round} of {max_rounds} ──[/dim]"
    )
    console.print("[dim]Press Enter to continue, or type to add direction for the agents:[/dim]")
    user_input = Prompt.ask("[dim]>[/dim]", default="")

    updates: dict = {"round": new_round}
    if user_input.strip():
        updates["user_interjection"] = user_input.strip()

    return updates
