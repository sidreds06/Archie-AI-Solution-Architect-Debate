"""HITL node logic for Gradio — no console I/O, yields events for the UI layer."""

from __future__ import annotations

from typing import Any

from state import DebateState

Event = dict[str, Any]


def generate_hitl_question(state: DebateState) -> str:
    """Generate a HITL question based on score analysis."""
    verdict = state.get("final_verdict") or {}
    scores = verdict.get("scores", {})
    p_scores = scores.get("proposer", {})
    a_scores = scores.get("adversary", {})
    p_total = p_scores.get("weighted_total", 0)
    a_total = a_scores.get("weighted_total", 0)

    max_diff_dim = ""
    max_diff = 0
    for dim in ["constraint_adherence", "technical_feasibility", "operational_complexity",
                 "scalability_fit", "evidence_quality", "cost_efficiency"]:
        diff = abs(p_scores.get(dim, 0) - a_scores.get(dim, 0))
        if diff > max_diff:
            max_diff = diff
            max_diff_dim = dim

    return (
        f"Scores are very close (Proposer: {p_total:.2f}, Adversary: {a_total:.2f}). "
        f"The biggest gap is in '{max_diff_dim.replace('_', ' ')}'. "
        f"Which matters more to you for this project -- and is there anything the agents are missing?"
    )
