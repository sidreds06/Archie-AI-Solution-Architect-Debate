from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console

from config import (
    ADVERSARY_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    PROPOSER_MODEL,
)
from state import DebateState
from tools.search import format_results_for_prompt, run_search
from ui.streaming import stream_to_panel

console = Console()


def _get_proposer_client() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=PROPOSER_MODEL,
    )


def _get_adversary_client() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=ADVERSARY_MODEL,
    )


def deep_dive_proposer(state: DebateState) -> dict:
    """LangGraph node. Proposer's focused response on the deep-dive topic."""
    topic = state.get("pending_deep_dive", "the contested topic")
    current_round = state["round"]

    console.print(f"\n[bold cyan]── DEEP DIVE: {topic} ──[/bold cyan]\n")

    # Quick targeted search
    with console.status("[bold blue]Proposer researching deep dive topic...[/bold blue]", spinner="dots"):
        results = run_search(f"{topic} architecture best practices", max_results=3)
    search_context = format_results_for_prompt(results) if results else ""

    client = _get_proposer_client()
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
    title = f"[bold blue]Proposer DEEP DIVE — {PROPOSER_MODEL} — Round {current_round}[/bold blue]"

    response = stream_to_panel(client, messages, title, "cyan")

    # Append deep dive context to the existing solution
    enriched_solution = (
        state["last_proposer_solution"]
        + f"\n\n--- DEEP DIVE: {topic} ---\n"
        + response.strip()
    )

    return {"last_proposer_solution": enriched_solution}


def deep_dive_adversary(state: DebateState) -> dict:
    """LangGraph node. Adversary's focused response on the deep-dive topic."""
    topic = state.get("pending_deep_dive", "the contested topic")
    current_round = state["round"]

    # Quick targeted search
    with console.status("[bold green]Adversary researching deep dive topic...[/bold green]", spinner="dots"):
        results = run_search(f"{topic} failures problems limitations", max_results=3)
    search_context = format_results_for_prompt(results) if results else ""

    client = _get_adversary_client()
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
    title = f"[bold green]Adversary DEEP DIVE — {ADVERSARY_MODEL} — Round {current_round}[/bold green]"

    response = stream_to_panel(client, messages, title, "cyan")

    # Append deep dive context to the existing solution
    enriched_solution = (
        state["last_adversary_solution"]
        + f"\n\n--- DEEP DIVE: {topic} ---\n"
        + response.strip()
    )

    return {
        "last_adversary_solution": enriched_solution,
        "pending_deep_dive": None,  # consumed
    }
