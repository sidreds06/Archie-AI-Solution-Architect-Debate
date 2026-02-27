"""Proposer logic — pure generator, no UI. Yields structured events."""

from __future__ import annotations

import json
import re
from typing import Any, Generator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import prompts.proposer as proposer_prompts
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROPOSER_MODEL
from state import DebateState
from tools.search import format_results_for_prompt, run_search, search_for_case_studies

Event = dict[str, Any]

_client: ChatOpenAI | None = None


def _get_client() -> ChatOpenAI:
    global _client
    if _client is None:
        _client = ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            model=PROPOSER_MODEL,
        )
    return _client


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _generate_search_queries(client: ChatOpenAI, problem: str) -> list[str]:
    prompt = proposer_prompts.build_search_query_prompt(problem)
    response = client.invoke([HumanMessage(content=prompt)])
    raw = _strip_fences(response.content.strip())
    try:
        queries = json.loads(raw)
        if isinstance(queries, list):
            return [str(q) for q in queries[:2]]
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
                        "agent": "proposer",
                        "request_type": req["request"],
                        "detail": req.get("topic") or req.get("reason", ""),
                    }
            except json.JSONDecodeError:
                pass
    return text, None


class ProposerEngine:
    """Runs proposer logic as a generator yielding events."""

    def __init__(self) -> None:
        self.last_result: dict | None = None

    def run(self, state: DebateState) -> Generator[Event, None, None]:
        client = _get_client()
        current_round = state["round"]
        user_memory = state.get("user_memory", {})
        momentum = state.get("momentum")
        enriched_context = state.get("enriched_context", "")

        debate_history = proposer_prompts.build_debate_history(
            state["proposals"], state.get("score_history", []), current_round
        )

        system_prompt = proposer_prompts.build_system_prompt(
            user_memory,
            momentum=momentum,
            current_round=current_round,
            max_rounds=state["max_rounds"],
        )

        # --- Search ---
        yield {"type": "status", "agent": "proposer", "message": "Researching..."}
        queries = _generate_search_queries(client, state["problem"])

        search_context = ""
        total_sources = 0
        if queries:
            all_results = []
            for i, query in enumerate(queries):
                yield {
                    "type": "search_start",
                    "agent": "proposer",
                    "query": query,
                    "index": i + 1,
                    "total": len(queries),
                }
                results = search_for_case_studies(query, max_results=5)
                if not results:
                    results = run_search(query, max_results=5)
                block = f"Query: {query}\n{format_results_for_prompt(results)}"
                all_results.append(block)
                total_sources += len(results)
            search_context = "\n---\n".join(all_results)
            yield {
                "type": "search_done",
                "agent": "proposer",
                "sources": total_sources,
                "queries": len(queries),
            }

        # --- Build prompt ---
        if current_round == 1:
            human_prompt = proposer_prompts.build_round1_prompt(
                state["problem"],
                enriched_context=enriched_context,
                search_context=search_context,
            )
        else:
            human_prompt = proposer_prompts.build_revision_prompt(
                problem=state["problem"],
                adversary_solution=state["last_adversary_solution"],
                hitl_answer=state.get("hitl_answer"),
                enriched_context=enriched_context,
                search_context=search_context,
                user_interjection=state.get("user_interjection"),
                debate_history=debate_history,
            )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        # --- Stream response ---
        yield {
            "type": "stream_start",
            "agent": "proposer",
            "model": PROPOSER_MODEL,
            "round": current_round,
        }

        chunks: list[str] = []
        for chunk in client.stream(messages):
            token = chunk.content or ""
            chunks.append(token)
            yield {"type": "token", "agent": "proposer", "token": token}

        solution = "".join(chunks).strip()
        solution, request = _extract_request(solution)

        yield {
            "type": "stream_end",
            "agent": "proposer",
            "solution": solution,
            "round": current_round,
            "sources": total_sources,
        }

        # Build state updates
        new_entry = {
            "agent": "proposer",
            "model": PROPOSER_MODEL,
            "round": current_round,
            "solution": solution,
            "score": None,
        }

        updates: dict = {
            "last_proposer_solution": solution,
            "proposals": state["proposals"] + [new_entry],
            "user_interjection": None,
            "search_metadata": state.get("search_metadata", []) + [{
                "agent": "proposer",
                "round": current_round,
                "queries": queries,
                "source_count": total_sources,
            }],
        }
        if request:
            updates["agent_requests"] = state.get("agent_requests", []) + [request]

        self.last_result = updates
