import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import prompts.moderator as moderator_prompts
from config import (
    HITL_MIN_ROUND_FOR_SMART_PAUSE,
    HITL_SCORE_CLOSENESS_THRESHOLD,
    MODERATOR_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from state import DebateState

console = Console()

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
    """Parse moderator JSON with a safe fallback."""
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


def _update_proposal_scores(
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


def _maybe_force_hitl(verdict: dict, current_round: int) -> dict:
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
        for dim in ["constraint_adherence", "technical_feasibility", "operational_complexity",
                     "scalability_fit", "evidence_quality", "cost_efficiency"]:
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
    """Build prompt, invoke LLM, parse result."""
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


def _display_verdict_panel(verdict: dict, current_round: int, title_extra: str = "") -> None:
    """Display the moderator's scoring panel."""
    decision = verdict.get("decision", "continue")
    p_total = verdict["scores"]["proposer"].get("weighted_total", 0)
    a_total = verdict["scores"]["adversary"].get("weighted_total", 0)
    panel_text = (
        f"Proposer score:  {p_total:.2f}    Adversary score: {a_total:.2f}\n"
        f"Decision: {decision.upper()}    Winner: {verdict.get('winner', 'TBD')}\n\n"
        f"{verdict.get('reasoning', '')}"
    )
    title_suffix = f" — {title_extra}" if title_extra else ""
    console.print(
        Panel(
            panel_text,
            title=f"[bold magenta]Moderator — {MODERATOR_MODEL} — Round {current_round}{title_suffix}[/bold magenta]",
            border_style="magenta",
        )
    )


def moderator(state: DebateState) -> dict:
    """LangGraph node. Scores both proposals, handles HITL inline, decides next action."""
    client = _get_client()
    current_round = state["round"]

    # --- First scoring pass ---
    with console.status(
        "[bold magenta]Moderator scoring...[/bold magenta]", spinner="dots"
    ):
        verdict = _score_round(client, state, hitl_answer=state.get("hitl_answer"))

    # Apply programmatic HITL fallback (only on first pass)
    verdict = _maybe_force_hitl(verdict, current_round)
    decision = verdict.get("decision", "continue")

    _display_verdict_panel(verdict, current_round)

    # --- Inline HITL: ask user and re-score, up to 2 times ---
    hitl_answer = state.get("hitl_answer")
    hitl_attempts = 0

    while decision == "hitl" and hitl_attempts < 2:
        hitl_attempts += 1
        question = verdict.get("hitl_question", "Please clarify a constraint.")

        console.print(f"\n[bold yellow]MODERATOR NEEDS YOUR INPUT[/bold yellow]")
        console.print(f"[yellow]{question}[/yellow]\n")
        hitl_answer = Prompt.ask("[bold yellow]Your answer[/bold yellow]")

        old_p = verdict["scores"]["proposer"].get("weighted_total", 0)
        old_a = verdict["scores"]["adversary"].get("weighted_total", 0)

        with console.status(
            "[bold magenta]Re-scoring with your input...[/bold magenta]", spinner="dots"
        ):
            verdict = _score_round(client, state, hitl_answer=hitl_answer)

        decision = verdict.get("decision", "continue")
        new_p = verdict["scores"]["proposer"].get("weighted_total", 0)
        new_a = verdict["scores"]["adversary"].get("weighted_total", 0)

        console.print(
            Panel(
                f"Before:  Proposer {old_p:.2f}  |  Adversary {old_a:.2f}\n"
                f"After:   Proposer {new_p:.2f}  |  Adversary {new_a:.2f}\n"
                f"Decision: {decision.upper()}\n\n"
                f"{verdict.get('reasoning', '')}",
                title="[bold yellow]Re-scored with your input[/bold yellow]",
                border_style="yellow",
            )
        )

    # If still "hitl" after max attempts, resolve to "continue"
    if decision == "hitl":
        verdict = {**verdict, "decision": "continue"}
        decision = "continue"

    # --- Build return updates ---
    updated_proposals = _update_proposal_scores(state["proposals"], verdict, current_round)

    updates: dict = {
        "proposals": updated_proposals,
        "final_verdict": verdict,
        "hitl_pending": None,
    }

    if hitl_answer is not None:
        updates["hitl_answer"] = hitl_answer

    if decision == "end":
        updates["debate_active"] = False
    else:
        updates["debate_active"] = True

    return updates
