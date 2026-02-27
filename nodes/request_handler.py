from rich.console import Console

from config import MAX_DEEP_DIVES, MAX_EXTRA_ROUNDS
from state import DebateState
from ui.dramatic import render_all_events

console = Console()


def handle_agent_requests(state: DebateState) -> dict:
    """LangGraph node. Processes structured requests from agents."""
    requests = state.get("agent_requests", [])
    if not requests:
        return {"agent_requests": []}

    updates: dict = {"agent_requests": []}  # clear after processing
    events: list[str] = list(state.get("dramatic_events", []))
    deep_dives_used = state.get("deep_dives_used", 0)

    for req in requests:
        req_type = req.get("request_type", "")
        agent = req.get("agent", "unknown")
        detail = req.get("detail", "")

        if req_type == "agree" and agent == "adversary":
            # Adversary concedes — early termination
            events.append("ADVERSARY CONCEDES!")
            updates["debate_active"] = False
            updates["final_verdict"] = {
                **state.get("final_verdict", {}),
                "decision": "end",
                "winner": "proposer",
                "reasoning": f"Adversary conceded: {detail}",
            }
            console.print(
                f"\n[bold green]Adversary concedes: {detail}[/bold green]"
            )

        elif req_type == "deep_dive" and deep_dives_used < MAX_DEEP_DIVES:
            events.append("DEEP DIVE REQUESTED!")
            updates["pending_deep_dive"] = detail
            updates["deep_dives_used"] = deep_dives_used + 1
            console.print(
                f"\n[bold cyan]Deep dive requested by {agent}: {detail}[/bold cyan]"
            )

        elif req_type == "extra_round":
            current_max = state["max_rounds"]
            original_max = current_max  # we don't track original, just cap total extras
            if current_max < state.get("max_rounds", 5) + MAX_EXTRA_ROUNDS:
                events.append("EXTRA ROUND GRANTED!")
                updates["max_rounds"] = current_max + 1
                console.print(
                    f"\n[bold yellow]Extra round granted (requested by {agent}): {detail}[/bold yellow]"
                )

        elif req_type == "pivot" and agent == "proposer":
            events.append("PROPOSER PIVOTS STRATEGY!")
            console.print(
                f"\n[bold blue]Proposer pivoting strategy: {detail}[/bold blue]"
            )
            # No state change needed — proposer will naturally revise next round

    if events != list(state.get("dramatic_events", [])):
        new_events = [e for e in events if e not in state.get("dramatic_events", [])]
        if new_events:
            render_all_events(new_events)
        updates["dramatic_events"] = events

    return updates
