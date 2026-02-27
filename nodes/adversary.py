import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

import prompts.adversary as adversary_prompts
from config import ADVERSARY_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from state import DebateState
from tools.search import format_results_for_prompt, run_search
from ui.streaming import stream_to_panel

console = Console()

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
    """Step 1: Generate 5 targeted search queries (3 attack + 2 defend)."""
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


def _run_all_searches(queries: list[str]) -> str:
    """Execute Tavily searches for each query, aggregate results."""
    all_results = []
    for query in queries:
        console.print(f"  [dim]Searching: {query}[/dim]")
        results = run_search(query, max_results=5, search_depth="advanced")
        block = f"Query: {query}\n{format_results_for_prompt(results)}"
        all_results.append(block)
    if all_results:
        total_sources = sum(
            1 for block in all_results for line in block.split("\n") if line.strip().startswith("[")
        )
        console.print(f"  [dim]Found {total_sources} sources across {len(queries)} searches[/dim]")
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
                        "agent": "adversary",
                        "request_type": req["request"],
                        "detail": req.get("topic") or req.get("reason", ""),
                    }
            except json.JSONDecodeError:
                pass
    return text, None


def adversary(state: DebateState) -> dict:
    """LangGraph node. Two-step: search then critique with evidence."""
    client = _get_client()
    current_round = state["round"]
    proposer_solution = state["last_proposer_solution"]
    user_memory = state.get("user_memory", {})
    momentum = state.get("momentum")
    enriched_context = state.get("enriched_context", "")

    # Build debate history from all prior rounds
    from prompts.proposer import build_debate_history
    debate_history = build_debate_history(
        state["proposals"], state.get("score_history", []), current_round
    )

    # Step 1: Generate targeted queries
    with console.status("[bold green]Adversary analyzing proposal...[/bold green]", spinner="dots"):
        queries = _generate_search_queries(client, proposer_solution)

    # Step 1b: Execute searches
    search_context = "No search queries generated."
    if queries:
        with console.status("[bold green]Adversary gathering evidence...[/bold green]", spinner="dots"):
            search_context = _run_all_searches(queries)

    # Step 2: Full critique + counter-proposal with streaming
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
    title = f"[bold green]Adversary — {ADVERSARY_MODEL} — Round {current_round}[/bold green]"

    solution = stream_to_panel(client, messages, title, "green")
    solution = solution.strip()

    # Extract optional agent request
    solution, request = _extract_request(solution)

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
    }
    if request:
        updates["agent_requests"] = state.get("agent_requests", []) + [request]

    return updates
