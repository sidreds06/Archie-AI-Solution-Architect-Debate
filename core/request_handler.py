"""Request handler logic — pure computation, no UI."""

from __future__ import annotations

from config import MAX_DEEP_DIVES, MAX_EXTRA_ROUNDS
from state import DebateState


def process_requests(state: DebateState) -> dict:
    """Process structured agent requests. Returns state updates."""
    requests = state.get("agent_requests", [])
    if not requests:
        return {"agent_requests": []}

    updates: dict = {"agent_requests": []}
    events: list[str] = list(state.get("dramatic_events", []))
    deep_dives_used = state.get("deep_dives_used", 0)

    for req in requests:
        req_type = req.get("request_type", "")
        agent = req.get("agent", "unknown")
        detail = req.get("detail", "")

        if req_type == "agree" and agent == "adversary":
            events.append("ADVERSARY CONCEDES!")
            updates["debate_active"] = False
            updates["final_verdict"] = {
                **state.get("final_verdict", {}),
                "decision": "end",
                "winner": "proposer",
                "reasoning": f"Adversary conceded: {detail}",
            }

        elif req_type == "deep_dive" and deep_dives_used < MAX_DEEP_DIVES:
            events.append("DEEP DIVE REQUESTED!")
            updates["pending_deep_dive"] = detail
            updates["deep_dives_used"] = deep_dives_used + 1

        elif req_type == "extra_round":
            current_max = state["max_rounds"]
            if current_max < state.get("max_rounds", 5) + MAX_EXTRA_ROUNDS:
                events.append("EXTRA ROUND GRANTED!")
                updates["max_rounds"] = current_max + 1

        elif req_type == "pivot" and agent == "proposer":
            events.append("PROPOSER PIVOTS STRATEGY!")

    if events != list(state.get("dramatic_events", [])):
        updates["dramatic_events"] = events

    return updates


def route_after_requests(state: DebateState) -> str:
    """Determine next step after request processing."""
    if state.get("pending_deep_dive"):
        return "deep_dive"

    if not state.get("debate_active", True):
        return "verdict"

    verdict_data = state.get("final_verdict") or {}
    decision = verdict_data.get("decision", "continue")

    if decision == "end":
        return "verdict"
    else:
        return "continue"
