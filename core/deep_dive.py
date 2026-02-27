"""Deep dive logic — pure generator, no UI. Yields structured events."""

from __future__ import annotations

from typing import Any, Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    ADVERSARY_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PROPOSER_MODEL,
)
from state import DebateState
from tools.search import format_results_for_prompt, run_search

Event = dict[str, Any]


class DeepDiveEngine:
    """Runs deep dive sub-rounds as generators yielding events."""

    def __init__(self) -> None:
        self.last_proposer_result: dict | None = None
        self.last_adversary_result: dict | None = None

    def run_proposer(self, state: DebateState) -> Generator[Event, None, None]:
        topic = state.get("pending_deep_dive", "the contested topic")
        current_round = state["round"]

        yield {"type": "status", "agent": "proposer", "message": f"Researching deep dive: {topic}"}

        results = run_search(f"{topic} architecture best practices", max_results=3)
        search_context = format_results_for_prompt(results) if results else ""

        yield {
            "type": "search_done",
            "agent": "proposer",
            "sources": len(results),
            "queries": 1,
        }

        client = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=PROPOSER_MODEL,
        )

        system = (
            "You are GPT-5.2. This is a DEEP DIVE — a focused mini-round on one specific topic. "
            "Be concise and technical. Address ONLY the topic below. Cite sources as [1], [2]."
        )
        human = (
            f"DEEP DIVE TOPIC: {topic}\n\n"
            f"Your current proposal context:\n{state['last_proposer_solution'][:500]}\n\n"
        )
        if search_context:
            human += f"RESEARCH:\n{search_context}\n\n"
        human += (
            "Provide your focused analysis on this specific topic. "
            "Be specific with service names, configurations, and trade-offs. Keep it under 500 words."
        )

        messages = [SystemMessage(content=system), HumanMessage(content=human)]

        yield {
            "type": "stream_start",
            "agent": "proposer",
            "model": PROPOSER_MODEL,
            "round": current_round,
            "is_deep_dive": True,
            "topic": topic,
        }

        chunks: list[str] = []
        for chunk in client.stream(messages):
            token = chunk.content or ""
            chunks.append(token)
            yield {"type": "token", "agent": "proposer", "token": token}

        response = "".join(chunks).strip()

        yield {
            "type": "stream_end",
            "agent": "proposer",
            "solution": response,
            "round": current_round,
            "is_deep_dive": True,
        }

        enriched_solution = (
            state["last_proposer_solution"]
            + f"\n\n--- DEEP DIVE: {topic} ---\n"
            + response
        )

        self.last_proposer_result = {"last_proposer_solution": enriched_solution}

    def run_adversary(self, state: DebateState) -> Generator[Event, None, None]:
        topic = state.get("pending_deep_dive", "the contested topic")
        current_round = state["round"]

        yield {"type": "status", "agent": "adversary", "message": f"Researching deep dive: {topic}"}

        results = run_search(f"{topic} failures problems limitations", max_results=3)
        search_context = format_results_for_prompt(results) if results else ""

        yield {
            "type": "search_done",
            "agent": "adversary",
            "sources": len(results),
            "queries": 1,
        }

        client = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=ADVERSARY_MODEL,
        )

        system = (
            "You are Claude Sonnet 4.5. This is a DEEP DIVE — a focused mini-round on one specific topic. "
            "Be concise, evidence-driven, and ruthless. Address ONLY the topic below. Cite sources as [1], [2]."
        )
        human = (
            f"DEEP DIVE TOPIC: {topic}\n\n"
            f"Proposer's position:\n{state['last_proposer_solution'][-1000:]}\n\n"
        )
        if search_context:
            human += f"RESEARCH:\n{search_context}\n\n"
        human += (
            "Provide your focused critique and alternative analysis on this specific topic. "
            "Cite evidence. Keep it under 500 words."
        )

        messages = [SystemMessage(content=system), HumanMessage(content=human)]

        yield {
            "type": "stream_start",
            "agent": "adversary",
            "model": ADVERSARY_MODEL,
            "round": current_round,
            "is_deep_dive": True,
            "topic": topic,
        }

        chunks: list[str] = []
        for chunk in client.stream(messages):
            token = chunk.content or ""
            chunks.append(token)
            yield {"type": "token", "agent": "adversary", "token": token}

        response = "".join(chunks).strip()

        yield {
            "type": "stream_end",
            "agent": "adversary",
            "solution": response,
            "round": current_round,
            "is_deep_dive": True,
        }

        enriched_solution = (
            state["last_adversary_solution"]
            + f"\n\n--- DEEP DIVE: {topic} ---\n"
            + response
        )

        self.last_adversary_result = {
            "last_adversary_solution": enriched_solution,
            "pending_deep_dive": None,
        }
