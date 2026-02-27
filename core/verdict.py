"""Verdict logic — pure generator, no UI. Yields structured events."""

from __future__ import annotations

from typing import Any, Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import MODERATOR_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from memory import manager as memory_manager
from state import DebateState

Event = dict[str, Any]


class VerdictEngine:
    """Generates final verdict brief as a generator yielding events."""

    def __init__(self) -> None:
        self.last_brief: str | None = None
        self.last_memory: dict | None = None
        self.last_memory_diff: dict | None = None

    def run(self, state: DebateState) -> Generator[Event, None, None]:
        final_verdict = state.get("final_verdict") or {}
        winner = final_verdict.get("winner", "proposer")
        score_history = state.get("score_history", [])

        # Identify winning solution
        winning_solution = ""
        for p in reversed(state["proposals"]):
            if p["agent"] == winner:
                winning_solution = p["solution"]
                break
        if not winning_solution and state["proposals"]:
            winning_solution = state["proposals"][-1]["solution"]

        score_summary = ""
        if score_history:
            last = score_history[-1]
            score_summary = (
                f"\nFINAL SCORES: Proposer {last.get('proposer_score', 0):.2f} | "
                f"Adversary {last.get('adversary_score', 0):.2f}\n"
            )

        client = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=MODERATOR_MODEL,
        )

        brief_prompt = (
            f"ARCHITECTURE PROBLEM:\n{state['problem']}\n\n"
            f"WINNING SOLUTION ({winner}):\n{winning_solution}\n\n"
            f"MODERATOR REASONING:\n{final_verdict.get('reasoning', '')}\n"
            f"{score_summary}\n"
            "Write a final architecture brief with these exact sections using Markdown:\n"
            "## Recommended Architecture\n"
            "## Why the Alternative Was Rejected\n"
            "## Key Risks to Monitor\n"
            "## Open Assumptions\n"
            "## Next Steps"
        )

        messages = [
            SystemMessage(
                content="You are Gemini, producing a concise final architecture brief for a technical audience."
            ),
            HumanMessage(content=brief_prompt),
        ]

        yield {
            "type": "stream_start",
            "agent": "verdict",
            "model": MODERATOR_MODEL,
            "round": state["round"],
        }

        chunks: list[str] = []
        for chunk in client.stream(messages):
            token = chunk.content or ""
            chunks.append(token)
            yield {"type": "token", "agent": "verdict", "token": token}

        brief = "".join(chunks).strip()
        self.last_brief = brief

        yield {
            "type": "stream_end",
            "agent": "verdict",
            "solution": brief,
            "round": state["round"],
        }

        # Update memory
        yield {"type": "status", "agent": "verdict", "message": "Updating memory..."}

        old_memory = dict(state.get("user_memory", {}))
        updated_memory = memory_manager.update(
            memory=state.get("user_memory", {}),
            session_problem=state["problem"],
            final_solution=winning_solution,
            moderator_client=client,
        )
        self.last_memory = updated_memory

        # Compute memory diff
        diff = _compute_memory_diff(old_memory, updated_memory)
        self.last_memory_diff = diff

        yield {
            "type": "verdict_done",
            "brief": brief,
            "winner": winner,
            "memory_diff": diff,
            "final_scores": score_history[-1] if score_history else {},
        }


def _compute_memory_diff(old: dict, new: dict) -> dict:
    """Compute what changed between old and new memory."""
    diff = {"added": {}, "updated": {}}

    list_fields = {
        "cloud_providers", "preferred_languages", "deployment_env",
        "off_limits", "vector_dbs_allowed", "notes",
    }
    scalar_fields = {"team_size", "domain", "budget_sensitivity"}

    for field in list_fields:
        old_items = set(old.get(field, []))
        new_items = set(new.get(field, []))
        added = new_items - old_items
        if added:
            diff["added"][field] = list(added)

    for field in scalar_fields:
        old_val = old.get(field)
        new_val = new.get(field)
        if old_val is None and new_val is not None:
            diff["updated"][field] = new_val

    return diff
