"""Moderator hub routing logic for Gradio engine — mirrors nodes/moderator_hub.py."""

from __future__ import annotations

from config import HITL_MIN_ROUND_FOR_SMART_PAUSE, HITL_SCORE_CLOSENESS_THRESHOLD
from state import DebateState


def should_force_hitl(state: DebateState) -> bool:
    """Check if programmatic HITL should be triggered."""
    current_round = state["round"]
    if current_round < HITL_MIN_ROUND_FOR_SMART_PAUSE:
        return False

    hitl_this_round = sum(
        1 for h in state.get("hitl_history", []) if h.get("round") == current_round
    )
    if hitl_this_round >= 2:
        return False

    verdict = state.get("final_verdict") or {}
    decision = verdict.get("decision", "continue")

    if decision == "hitl":
        return True

    if decision != "continue":
        return False

    scores = verdict.get("scores", {})
    p_total = scores.get("proposer", {}).get("weighted_total", 0)
    a_total = scores.get("adversary", {}).get("weighted_total", 0)
    return abs(p_total - a_total) < HITL_SCORE_CLOSENESS_THRESHOLD


def route_after_scoring(state: DebateState) -> str:
    """Determine the next phase after scoring + scoreboard + request handling.
    Returns: 'hitl', 'continue', 'verdict'."""
    if not state.get("debate_active", True):
        return "verdict"
    verdict_data = state.get("final_verdict") or {}
    decision = verdict_data.get("decision", "continue")
    if decision == "end":
        return "verdict"
    if should_force_hitl(state):
        return "hitl"
    return "continue"
