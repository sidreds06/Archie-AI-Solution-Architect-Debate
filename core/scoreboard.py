"""Scoreboard logic — pure computation, no UI."""

from __future__ import annotations

from state import DebateState


def _detect_events(
    score_history: list[dict],
    proposer_score: float,
    adversary_score: float,
) -> list[str]:
    events: list[str] = []
    delta = abs(proposer_score - adversary_score)

    if len(score_history) >= 1:
        prev = score_history[-1]
        prev_p = prev.get("proposer_score", 0)
        prev_a = prev.get("adversary_score", 0)

        if (prev_p > prev_a and adversary_score > proposer_score) or (
            prev_a > prev_p and proposer_score > adversary_score
        ):
            events.append("LEAD CHANGE!")

        p_swing = abs(proposer_score - prev_p)
        a_swing = abs(adversary_score - prev_a)
        if p_swing > 0.5 or a_swing > 0.5:
            events.append("SCORE SHIFT!")

    if delta < 0.15:
        events.append("NEAR TIE!")

    return events


def _compute_momentum(
    score_history: list[dict],
    proposer_score: float,
    adversary_score: float,
) -> dict:
    if not score_history:
        return {"proposer": proposer_score, "adversary": adversary_score}
    prev = score_history[-1]
    return {
        "proposer": proposer_score - prev.get("proposer_score", 0),
        "adversary": adversary_score - prev.get("adversary_score", 0),
    }


def compute_scoreboard(state: DebateState) -> dict:
    """Compute scoreboard data. Returns state updates dict."""
    verdict_data = state.get("final_verdict") or {}
    scores = verdict_data.get("scores", {})
    current_round = state["round"]
    score_history = state.get("score_history", [])

    proposer_score = scores.get("proposer", {}).get("weighted_total", 0.0)
    adversary_score = scores.get("adversary", {}).get("weighted_total", 0.0)
    delta = proposer_score - adversary_score

    events = _detect_events(score_history, proposer_score, adversary_score)

    new_entry = {
        "round": current_round,
        "proposer_score": proposer_score,
        "adversary_score": adversary_score,
        "delta": delta,
    }
    updated_history = score_history + [new_entry]

    momentum = _compute_momentum(score_history, proposer_score, adversary_score)

    return {
        "score_history": updated_history,
        "momentum": momentum,
        "dramatic_events": events,
    }
