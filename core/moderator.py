"""Moderator logic — pure generator, no UI. Yields structured events."""

from __future__ import annotations

import json
import re
from typing import Any, Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import prompts.moderator as moderator_prompts
from config import (
    HITL_MIN_ROUND_FOR_SMART_PAUSE,
    HITL_SCORE_CLOSENESS_THRESHOLD,
    MODERATOR_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from state import DebateState

Event = dict[str, Any]

_client: ChatOpenAI | None = None


def _get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        _client = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=MODERATOR_MODEL,
            temperature=0.1,
        )
    return _client


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _safe_parse(raw: str, current_round: int, max_rounds: int) -> dict:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decision = "end" if current_round >= max_rounds else "continue"
        blank_scores = {
            k: 3.0
            for k in [
                "constraint_adherence",
                "technical_feasibility",
                "operational_complexity",
                "scalability_fit",
                "evidence_quality",
                "cost_efficiency",
                "weighted_total",
            ]
        }
        return {
            "scores": {
                "proposer": blank_scores.copy(),
                "adversary": blank_scores.copy(),
            },
            "decision": decision,
            "winner": None,
            "hitl_question": None,
            "reasoning": "Parse failure — fallback scoring applied.",
        }


def update_proposal_scores(
    proposals: list[dict], verdict: dict, current_round: int
) -> list[dict]:
    """Return a new proposals list with scores injected for current round."""
    scores = verdict.get("scores", {})
    result = []
    for p in proposals:
        if p["round"] == current_round:
            agent_key = p["agent"]
            agent_scores = scores.get(agent_key, {})
            result.append({**p, "score": agent_scores.get("weighted_total")})
        else:
            result.append(p)
    return result


def maybe_force_hitl(verdict: dict, current_round: int) -> dict:
    """Programmatic HITL fallback: force hitl when scores are very close."""
    decision = verdict.get("decision", "continue")
    if decision != "continue":
        return verdict

    if current_round < HITL_MIN_ROUND_FOR_SMART_PAUSE:
        return verdict

    scores = verdict.get("scores", {})
    p_total = scores.get("proposer", {}).get("weighted_total", 0)
    a_total = scores.get("adversary", {}).get("weighted_total", 0)
    delta = abs(p_total - a_total)

    if delta < HITL_SCORE_CLOSENESS_THRESHOLD:
        p_scores = scores.get("proposer", {})
        a_scores = scores.get("adversary", {})
        max_diff_dim = ""
        max_diff = 0
        for dim in [
            "constraint_adherence", "technical_feasibility", "operational_complexity",
            "scalability_fit", "evidence_quality", "cost_efficiency",
        ]:
            diff = abs(p_scores.get(dim, 0) - a_scores.get(dim, 0))
            if diff > max_diff:
                max_diff = diff
                max_diff_dim = dim

        question = (
            f"Scores are very close (Proposer: {p_total:.2f}, Adversary: {a_total:.2f}). "
            f"The biggest gap is in '{max_diff_dim.replace('_', ' ')}'. "
            f"Which matters more to you for this project — and is there anything the agents are missing?"
        )
        return {
            **verdict,
            "decision": "hitl",
            "hitl_question": question,
        }

    return verdict


def _score_round(client: ChatOpenAI, state: DebateState, hitl_answer: str | None = None) -> dict:
    human_prompt = moderator_prompts.build_moderator_prompt(
        problem=state["problem"],
        proposer_solution=state["last_proposer_solution"],
        adversary_solution=state["last_adversary_solution"],
        current_round=state["round"],
        max_rounds=state["max_rounds"],
        user_memory=state.get("user_memory", {}),
        hitl_answer=hitl_answer,
        enriched_context=state.get("enriched_context", ""),
    )
    response = client.invoke(
        [
            SystemMessage(
                content="You are an impartial architecture moderator. Return only valid JSON."
            ),
            HumanMessage(content=human_prompt),
        ]
    )
    return _safe_parse(response.content, state["round"], state["max_rounds"])


class ModeratorEngine:
    """Runs moderator scoring as a generator yielding events."""

    def __init__(self) -> None:
        self.last_verdict: dict | None = None

    def score(
        self, state: DebateState, hitl_answer: str | None = None
    ) -> Generator[Event, None, None]:
        client = _get_client()
        current_round = state["round"]

        yield {"type": "status", "agent": "moderator", "message": "Scoring..."}

        verdict = _score_round(client, state, hitl_answer=hitl_answer or state.get("hitl_answer"))
        self.last_verdict = verdict

        scores = verdict.get("scores", {})
        p_total = scores.get("proposer", {}).get("weighted_total", 0)
        a_total = scores.get("adversary", {}).get("weighted_total", 0)

        yield {
            "type": "score",
            "verdict": verdict,
            "round": current_round,
            "proposer_score": p_total,
            "adversary_score": a_total,
            "decision": verdict.get("decision", "continue"),
            "reasoning": verdict.get("reasoning", ""),
            "winner": verdict.get("winner"),
            "dimension_scores": {
                "proposer": {k: v for k, v in scores.get("proposer", {}).items() if k != "weighted_total"},
                "adversary": {k: v for k, v in scores.get("adversary", {}).items() if k != "weighted_total"},
            },
        }
