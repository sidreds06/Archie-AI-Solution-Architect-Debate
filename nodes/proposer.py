import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

import prompts.proposer as proposer_prompts
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROPOSER_MODEL
from state import DebateState
from tools.search import format_results_for_prompt, run_search, search_for_case_studies
from ui.streaming import stream_to_panel

console = Console()

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
    """Generate 2 search queries for reference architectures and best practices."""
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


def _run_searches(queries: list[str]) -> str:
    """Execute searches and aggregate results."""
    all_results = []
    for query in queries:
        console.print(f"  [dim]Researching: {query}[/dim]")
        results = search_for_case_studies(query, max_results=5)
        if not results:
            results = run_search(query, max_results=5)
        block = f"Query: {query}\n{format_results_for_prompt(results)}"
        all_results.append(block)
    if all_results:
        count = sum(block.count("[") for block in all_results)
        console.print(f"  [dim]Found {count} sources across {len(queries)} searches[/dim]")
    return "\n---\n".join(all_results)


def _extract_request(text: str) -> tuple[str, dict | None]:
    """Strip trailing JSON request from agent response."""
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


def proposer(state: DebateState) -> dict:
    """LangGraph node. GPT-5.2 generates or revises an architecture proposal with evidence."""
    client = _get_client()
    current_round = state["round"]
    user_memory = state.get("user_memory", {})
    momentum = state.get("momentum")
    enriched_context = state.get("enriched_context", "")

    # Build debate history from all prior rounds
    debate_history = proposer_prompts.build_debate_history(
        state["proposals"], state.get("score_history", []), current_round
    )

    system_prompt = proposer_prompts.build_system_prompt(
        user_memory,
        momentum=momentum,
        current_round=current_round,
        max_rounds=state["max_rounds"],
    )

    # Search for evidence
    with console.status("[bold blue]Proposer researching...[/bold blue]", spinner="dots"):
        queries = _generate_search_queries(client, state["problem"])
    search_context = ""
    if queries:
        search_context = _run_searches(queries)

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
    title = f"[bold blue]Proposer — {PROPOSER_MODEL} — Round {current_round}[/bold blue]"

    solution = stream_to_panel(client, messages, title, "blue")
    solution = solution.strip()

    # Extract optional agent request
    solution, request = _extract_request(solution)

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
        "user_interjection": None,  # consumed
    }
    if request:
        updates["agent_requests"] = state.get("agent_requests", []) + [request]

    return updates
