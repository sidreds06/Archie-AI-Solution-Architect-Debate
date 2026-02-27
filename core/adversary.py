"""Adversary logic — pure generator, no UI. Yields structured events."""

from __future__ import annotations

import json
import re
from typing import Any, Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import prompts.adversary as adversary_prompts
from config import ADVERSARY_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from state import DebateState
from tools.search import format_results_for_prompt, run_search

Event = dict[str, Any]

_client: ChatOpenAI | None = None


def _get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        _client = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=ADVERSARY_MODEL,
        )
    return _client


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _generate_search_queries(client: ChatOpenAI, proposer_solution: str) -> list[str]:
    prompt = adversary_prompts.build_query_generation_prompt(proposer_solution)
    response = client.invoke([HumanMessage(content=prompt)])
    raw = _strip_fences(response.content.strip())
    try:
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [str(q) for q in queries[:5]]
    except json.JSONDecodeError:
        pass
    return []


def _extract_request(text: str) -> tuple[str, dict | None]:
    lines = text.rstrip().split("\n")
    for i in range(len(lines) - 1, max(len(lines) - 5, -1), -1):
        line = lines[i].strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                req = json.loads(line)
                if "request" in req:
                    clean = "\n".join(lines[:i]).rstrip()
                    return clean, {
                        "agent": "adversary",
                        "request_type": req["request"],
                        "detail": req.get("topic") or req.get("reason", ""),
                    }
            except json.JSONDecodeError:
                pass
    return text, None


class AdversaryEngine:
    """Runs adversary logic as a generator yielding events."""

    def __init__(self) -> None:
        self.last_result: dict | None = None

    def run(self, state: DebateState) -> Generator[Event, None, None]:
        client = _get_client()
        current_round = state["round"]
        proposer_solution = state["last_proposer_solution"]
        user_memory = state.get("user_memory", {})
        momentum = state.get("momentum")
        enriched_context = state.get("enriched_context", "")

        from prompts.proposer import build_debate_history
        debate_history = build_debate_history(
            state["proposals"], state.get("score_history", []), current_round
        )

        # --- Generate search queries ---
        yield {"type": "status", "agent": "adversary", "message": "Analyzing proposal..."}
        queries = _generate_search_queries(client, proposer_solution)

        # --- Execute searches ---
        search_context = "No search queries generated."
        total_sources = 0
        if queries:
            yield {"type": "status", "agent": "adversary", "message": "Gathering evidence..."}
            all_results = []
            for i, query in enumerate(queries):
                yield {
                    "type": "search_start",
                    "agent": "adversary",
                    "query": query,
                    "index": i + 1,
                    "total": len(queries),
                }
                results = run_search(query, max_results=5, search_depth="advanced")
                block = f"Query: {query}\n{format_results_for_prompt(results)}"
                all_results.append(block)
                total_sources += len(results)
            search_context = "\n---\n".join(all_results)
            yield {
                "type": "search_done",
                "agent": "adversary",
                "sources": total_sources,
                "queries": len(queries),
            }

        # --- Build prompt ---
        system_prompt = adversary_prompts.build_system_prompt(
            user_memory,
            momentum=momentum,
            current_round=current_round,
            max_rounds=state["max_rounds"],
        )
        human_prompt = adversary_prompts.build_critique_prompt(
            user_memory=user_memory,
            problem=state["problem"],
            proposer_solution=proposer_solution,
            search_context=search_context,
            enriched_context=enriched_context,
            user_interjection=state.get("user_interjection"),
            debate_history=debate_history,
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        # --- Stream response ---
        yield {
            "type": "stream_start",
            "agent": "adversary",
            "model": ADVERSARY_MODEL,
            "round": current_round,
        }

        chunks: list[str] = []
        for chunk in client.stream(messages):
            token = chunk.content or ""
            chunks.append(token)
            yield {"type": "token", "agent": "adversary", "token": token}

        solution = "".join(chunks).strip()
        solution, request = _extract_request(solution)

        yield {
            "type": "stream_end",
            "agent": "adversary",
            "solution": solution,
            "round": current_round,
            "sources": total_sources,
        }

        # Build state updates
        new_entry = {
            "agent": "adversary",
            "model": ADVERSARY_MODEL,
            "round": current_round,
            "solution": solution,
            "score": None,
        }

        updates: dict = {
            "last_adversary_solution": solution,
            "proposals": state["proposals"] + [new_entry],
            "search_metadata": state.get("search_metadata", []) + [{
                "agent": "adversary",
                "round": current_round,
                "queries": queries,
                "source_count": total_sources,
            }],
        }
        if request:
            updates["agent_requests"] = state.get("agent_requests", []) + [request]

        self.last_result = updates
