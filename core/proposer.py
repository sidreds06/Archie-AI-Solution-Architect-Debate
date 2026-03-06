"""Proposer logic — tool-calling agent, yields structured events."""

from __future__ import annotations

import json
from typing import Any, Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import prompts.proposer as proposer_prompts
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROPOSER_MODEL
from state import DebateState
from tools.agent_tools import deep_dive, web_search

Event = dict[str, Any]

_TOOLS = [web_search, deep_dive]

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


def _execute_tool(tool_call: dict) -> str:
    tool_map = {t.name: t for t in _TOOLS}
    func = tool_map.get(tool_call["name"])
    if func is None:
        return f"Unknown tool: {tool_call['name']}"
    return func.invoke(tool_call["args"])


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
    """Runs proposer logic as a tool-calling agent yielding events."""

    def __init__(self) -> None:
        self.last_result: dict | None = None

    def run(self, state: DebateState) -> Generator[Event, None, None]:
        client = _get_client()
        bound = client.bind_tools(_TOOLS)
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

        if current_round == 1:
            human_prompt = proposer_prompts.build_round1_prompt(
                state["problem"],
                enriched_context=enriched_context,
            )
        else:
            human_prompt = proposer_prompts.build_revision_prompt(
                problem=state["problem"],
                adversary_solution=state["last_adversary_solution"],
                hitl_answer=state.get("hitl_answer"),
                enriched_context=enriched_context,
                user_interjection=state.get("user_interjection"),
                debate_history=debate_history,
            )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

        # Agent tool-calling loop
        while True:
            response: AIMessage = bound.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                yield {
                    "type": "tool_call",
                    "agent": "proposer",
                    "tool": tc["name"],
                    "args": tc["args"],
                }
                result = _execute_tool(tc)
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )
                yield {
                    "type": "tool_result",
                    "agent": "proposer",
                    "tool": tc["name"],
                }

        # Stream final response
        yield {
            "type": "stream_start",
            "agent": "proposer",
            "model": PROPOSER_MODEL,
            "round": current_round,
        }

        chunks: list[str] = []
        for chunk in client.stream(messages[:-1]):
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
        }

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
        }
        if request:
            updates["agent_requests"] = state.get("agent_requests", []) + [request]

        self.last_result = updates
