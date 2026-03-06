from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config import MODERATOR_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from memory import manager as memory_manager
from state import DebateState
from ui.banners import session_footer, verdict_banner
from ui.streaming import stream_to_panel

console = Console()


def _get_moderator_client() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=MODERATOR_MODEL,
    )


def verdict(state: DebateState) -> dict:
    """LangGraph node. Generates the final architecture brief and updates memory."""
    final_verdict = state.get("final_verdict") or {}
    winner = final_verdict.get("winner", "proposer")
    score_history = state.get("score_history", [])

    # Identify winning solution text from the proposals list
    winning_solution = ""
    for p in reversed(state["proposals"]):
        if p["agent"] == winner:
            winning_solution = p["solution"]
            break
    if not winning_solution and state["proposals"]:
        winning_solution = state["proposals"][-1]["solution"]

    # Get final scores
    p_score = 0.0
    a_score = 0.0
    if score_history:
        last = score_history[-1]
        p_score = last.get("proposer_score", 0.0)
        a_score = last.get("adversary_score", 0.0)

    # Print final scoreboard
    from nodes.moderator_subgraph import _render_scoreboard
    _render_scoreboard(score_history)

    # Winner announcement
    verdict_banner(winner, p_score, a_score)

    # Build score summary for the brief
    score_summary = f"\nFINAL SCORES: Proposer {p_score:.2f} | Adversary {a_score:.2f}\n"

    client = _get_moderator_client()

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

    title = "[bold white]Architecture Brief[/bold white]"
    brief = stream_to_panel(client, messages, title, "bright_white")
    brief = brief.strip()

    # Re-render with Markdown formatting
    console.print(
        Panel(
            Markdown(brief),
            title=title,
            border_style="bright_white",
        )
    )

    # Update persistent memory
    updated_memory = memory_manager.update(
        memory=state.get("user_memory", {}),
        session_problem=state["problem"],
        final_solution=winning_solution,
        moderator_client=client,
    )

    session_footer(state["round"], memory_updated=True)

    return {"user_memory": updated_memory}
