"""Moderator subgraph — tool-calling agent with hitl tool.

Subgraph: START → evaluate → evaluate_router
           ├── think → tool_router → hitl (loop) → respond → END
           └── respond → END  (short-circuit for init/proposal_done)

The respond node absorbs: scoring display, scoreboard, request handling, round increment.
"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel

from rich.table import Table

import prompts.moderator as moderator_prompts
from config import (
    MAX_EXTRA_ROUNDS,
    MODERATOR_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from state import DebateState
from tools.agent_tools import hitl
from ui.dramatic import render_all_events, render_events_with_context

console = Console()

_TOOLS = [hitl]

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
            "reasoning": "Parse failure -- fallback scoring applied.",
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


def _display_verdict(verdict: dict, current_round: int) -> None:
    """Display the moderator's scoring as a structured, scannable output."""
    from ui.banners import score_comparison
    score_comparison(verdict, current_round, MODERATOR_MODEL)


def _detect_events(
    score_history: list[dict],
    proposer_score: float,
    adversary_score: float,
) -> list[str]:
    """Detect dramatic moments based on score changes."""
    events: list[str] = []
    delta = abs(proposer_score - adversary_score)

    if len(score_history) >= 1:
        prev = score_history[-1]
        prev_p = prev.get("proposer_score", 0)
        prev_a = prev.get("adversary_score", 0)

        if (prev_p > prev_a and adversary_score > proposer_score) or (
            prev_a > prev_p and proposer_score > adversary_score
        ):
            events.append("LEAD CHANGE!")

        p_swing = abs(proposer_score - prev_p)
        a_swing = abs(adversary_score - prev_a)
        if p_swing > 0.5 or a_swing > 0.5:
            events.append("SCORE SHIFT!")

    if delta < 0.15:
        events.append("NEAR TIE!")

    return events


def _compute_momentum(
    score_history: list[dict],
    proposer_score: float,
    adversary_score: float,
) -> dict:
    """Compute momentum based on score trends."""
    if not score_history:
        return {"proposer": proposer_score, "adversary": adversary_score}
    prev = score_history[-1]
    return {
        "proposer": proposer_score - prev.get("proposer_score", 0),
        "adversary": adversary_score - prev.get("adversary_score", 0),
    }


def _render_scoreboard(updated_history: list[dict]) -> None:
    """Render the Rich scoreboard table."""
    table = Table(
        title="[bold white]SCOREBOARD[/bold white]",
        border_style="bright_white",
        show_lines=True,
    )
    table.add_column("Round", style="white", justify="center", width=7)
    table.add_column("Proposer", style="blue", justify="center", width=12)
    table.add_column("Adversary", style="green", justify="center", width=12)
    table.add_column("Delta", style="yellow", justify="center", width=10)
    table.add_column("Trend", style="white", justify="center", width=7)

    for i, entry in enumerate(updated_history):
        if i == 0:
            trend = "—"
        else:
            prev_delta = updated_history[i - 1]["delta"]
            curr_delta = entry["delta"]
            if curr_delta > prev_delta + 0.1:
                trend = "[blue]▲ P[/blue]"
            elif curr_delta < prev_delta - 0.1:
                trend = "[green]▲ A[/green]"
            else:
                trend = "[yellow]=[/yellow]"

        delta_str = f"{entry['delta']:+.2f}"
        if entry["delta"] > 0:
            delta_str = f"[blue]{delta_str}[/blue]"
        elif entry["delta"] < 0:
            delta_str = f"[green]{delta_str}[/green]"

        table.add_row(
            str(entry["round"]),
            f"{entry['proposer_score']:.2f}",
            f"{entry['adversary_score']:.2f}",
            delta_str,
            trend,
        )

    console.print()
    console.print(table)


def _handle_requests(state: DebateState) -> dict:
    """Process structured requests from agents. Returns updates dict."""
    requests = state.get("agent_requests", [])
    if not requests:
        return {"agent_requests": []}

    updates: dict = {"agent_requests": []}
    events: list[str] = list(state.get("dramatic_events", []))

    for req in requests:
        req_type = req.get("request_type", "")
        agent = req.get("agent", "unknown")
        detail = req.get("detail", "")

        if req_type == "agree" and agent == "adversary":
            events.append("ADVERSARY CONCEDES!")
            updates["debate_active"] = False
            updates["final_verdict"] = {
                **state.get("final_verdict", {}),
                "decision": "end",
                "winner": "proposer",
                "reasoning": f"Adversary conceded: {detail}",
            }
            console.print(f"\n[bold green]Adversary concedes: {detail}[/bold green]")

        elif req_type == "extra_round":
            current_max = state["max_rounds"]
            if current_max < state.get("max_rounds", 5) + MAX_EXTRA_ROUNDS:
                events.append("EXTRA ROUND GRANTED!")
                updates["max_rounds"] = current_max + 1
                console.print(
                    f"\n[bold yellow]Extra round granted (requested by {agent}): {detail}[/bold yellow]"
                )

        elif req_type == "pivot" and agent == "proposer":
            events.append("PROPOSER PIVOTS STRATEGY!")
            console.print(f"\n[bold blue]Proposer pivoting strategy: {detail}[/bold blue]")

    if events != list(state.get("dramatic_events", [])):
        new_events = [e for e in events if e not in state.get("dramatic_events", [])]
        if new_events:
            render_all_events(new_events)
        updates["dramatic_events"] = events

    return updates


# ------------------------------------------------------------------
# Subgraph nodes
# ------------------------------------------------------------------


def evaluate(state: DebateState) -> dict:
    """Entry point: decide if we need the LLM or can short-circuit."""
    phase = state.get("debate_phase", "init")

    if phase == "init":
        # No scoring needed — just route to proposer
        return {"debate_phase": "need_proposal", "_agent_messages": []}
    elif phase == "proposal_done":
        # No scoring needed — just route to adversary
        return {"debate_phase": "need_critique", "_agent_messages": []}
    else:
        # critique_done → need LLM scoring
        return {"debate_phase": "need_scoring", "_agent_messages": []}


def evaluate_router(state: DebateState) -> str:
    """Route based on debate_phase after evaluate."""
    phase = state.get("debate_phase", "")
    if phase in ("need_proposal", "need_critique"):
        return "respond"  # short-circuit — no LLM call needed
    return "think"


def moderator_think(state: DebateState) -> dict:
    """Call moderator LLM with hitl tool bound."""
    messages = state.get("_agent_messages", [])

    if not messages:
        human_prompt = moderator_prompts.build_moderator_prompt(
            problem=state["problem"],
            proposer_solution=state["last_proposer_solution"],
            adversary_solution=state["last_adversary_solution"],
            current_round=state["round"],
            max_rounds=state["max_rounds"],
            user_memory=state.get("user_memory", {}),
            hitl_answer=state.get("hitl_answer"),
            enriched_context=state.get("enriched_context", ""),
        )
        messages = [
            SystemMessage(
                content="You are an impartial architecture moderator. Return only valid JSON. "
                "You have a tool called 'hitl' that lets you ask the user a clarifying question. "
                "Use it when scores are close, agents debate unstated constraints, or you need "
                "clarification on a specific requirement."
            ),
            HumanMessage(content=human_prompt),
        ]

    client = _get_client()
    bound = client.bind_tools(_TOOLS)

    with console.status("[bold magenta]Moderator scoring...[/bold magenta]", spinner="dots"):
        response: AIMessage = bound.invoke(messages)

    messages.append(response)
    return {"_agent_messages": messages}


def mod_tool_router(state: DebateState) -> str:
    """Route: if hitl tool was called → hitl node, else → respond."""
    messages = state.get("_agent_messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "hitl"
    return "respond"


def exec_hitl(state: DebateState) -> dict:
    """Execute hitl tool call (prompts user) and append ToolMessage."""
    messages = list(state.get("_agent_messages", []))
    last_msg = messages[-1]

    for tc in last_msg.tool_calls:
        if tc["name"] == "hitl":
            result = hitl.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

            # Record in HITL history
            hitl_history = list(state.get("hitl_history", []))
            hitl_history.append({
                "round": state["round"],
                "question": tc["args"].get("question", ""),
                "answer": result,
            })
            return {
                "_agent_messages": messages,
                "hitl_answer": result,
                "hitl_history": hitl_history,
            }

    return {"_agent_messages": messages}


def respond(state: DebateState) -> dict:
    """Post-processing: parse verdict, update scores, render scoreboard, handle requests, round increment."""
    phase = state.get("debate_phase", "")
    current_round = state["round"]

    # Short-circuit: no LLM call was made, just routing
    if phase in ("need_proposal", "need_critique"):
        return {"_agent_messages": []}

    # Parse the moderator's LLM response
    messages = state.get("_agent_messages", [])
    raw_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            raw_content = msg.content
            break

    verdict = _safe_parse(raw_content, current_round, state["max_rounds"])

    # Hard enforcement: if we've reached max_rounds, force end regardless of LLM decision
    if current_round >= state["max_rounds"]:
        verdict["decision"] = "end"

    # If moderator decided HITL but didn't call the tool, execute it manually
    extra_updates: dict = {}
    if verdict.get("decision") == "hitl" and verdict.get("hitl_question"):
        from tools.agent_tools import hitl as hitl_tool

        answer = hitl_tool.invoke({"question": verdict["hitl_question"]})

        # Record in HITL history
        hitl_history = list(state.get("hitl_history", []))
        hitl_history.append({
            "round": current_round,
            "question": verdict["hitl_question"],
            "answer": answer,
        })

        # Re-invoke moderator with the answer
        client = _get_client()
        bound = client.bind_tools(_TOOLS)
        re_prompt = (
            f"The user answered your HITL question.\n"
            f"Question: {verdict['hitl_question']}\n"
            f"Answer: {answer}\n\n"
            f"Now re-score both proposals incorporating this new information. "
            f"Return the same JSON format."
        )
        re_messages = list(messages) + [HumanMessage(content=re_prompt)]
        with console.status("[bold magenta]Moderator re-scoring...[/bold magenta]", spinner="dots"):
            response = bound.invoke(re_messages)
        re_content = response.content if isinstance(response, AIMessage) and response.content else ""
        verdict = _safe_parse(re_content, current_round, state["max_rounds"])

        # Re-enforce max_rounds after re-scoring
        if current_round >= state["max_rounds"]:
            verdict["decision"] = "end"

        extra_updates = {"hitl_answer": answer, "hitl_history": hitl_history}

    # Display structured verdict
    _display_verdict(verdict, current_round)

    # Update proposal scores
    updated_proposals = _update_proposal_scores(state["proposals"], verdict, current_round)

    # --- Scoreboard ---
    scores = verdict.get("scores", {})
    score_history = state.get("score_history", [])
    proposer_score = scores.get("proposer", {}).get("weighted_total", 0.0)
    adversary_score = scores.get("adversary", {}).get("weighted_total", 0.0)
    delta = proposer_score - adversary_score

    events = _detect_events(score_history, proposer_score, adversary_score)
    new_entry = {
        "round": current_round,
        "proposer_score": proposer_score,
        "adversary_score": adversary_score,
        "delta": delta,
    }
    updated_history = score_history + [new_entry]
    _render_scoreboard(updated_history)

    if events:
        render_events_with_context(events, proposer_score, adversary_score)

    momentum = _compute_momentum(score_history, proposer_score, adversary_score)

    # --- Request handling ---
    request_updates = _handle_requests(state)

    # --- Determine next phase ---
    decision = verdict.get("decision", "continue")

    # Check if debate should end
    debate_active = state.get("debate_active", True)
    if decision == "end":
        debate_active = False
    if request_updates.get("debate_active") is not None:
        debate_active = request_updates["debate_active"]

    if not debate_active:
        next_phase = "done"
    else:
        # Round increment: advance round and offer user a chance to inject direction
        new_round = current_round + 1
        max_rounds = state["max_rounds"]
        if request_updates.get("max_rounds"):
            max_rounds = request_updates["max_rounds"]

        next_phase = "need_proposal"

        updates = {
            "proposals": updated_proposals,
            "final_verdict": verdict,
            "score_history": updated_history,
            "momentum": momentum,
            "dramatic_events": events,
            "debate_active": debate_active,
            "debate_phase": next_phase,
            "hitl_pending": None,
            "round": new_round,
            "_agent_messages": [],
        }
        updates.update({k: v for k, v in request_updates.items() if k != "debate_active"})
        updates.update(extra_updates)
        return updates

    # Debate ending — no round increment
    updates = {
        "proposals": updated_proposals,
        "final_verdict": verdict,
        "score_history": updated_history,
        "momentum": momentum,
        "dramatic_events": events,
        "debate_active": debate_active,
        "debate_phase": "done",
        "hitl_pending": None,
        "_agent_messages": [],
    }
    updates.update({k: v for k, v in request_updates.items() if k != "debate_active"})
    updates.update(extra_updates)
    return updates


# ------------------------------------------------------------------
# Parent graph router
# ------------------------------------------------------------------


def parent_router(state: DebateState) -> str:
    """Route from moderator subgraph to next parent node."""
    phase = state.get("debate_phase", "")
    if phase == "need_proposal":
        return "proposer"
    elif phase == "need_critique":
        return "adversary"
    else:  # "done"
        return "verdict"


# ------------------------------------------------------------------
# Build subgraph
# ------------------------------------------------------------------


def build_moderator_subgraph():
    """Build and compile the moderator subgraph."""
    builder = StateGraph(DebateState)

    builder.add_node("evaluate", evaluate)
    builder.add_node("think", moderator_think)
    builder.add_node("hitl", exec_hitl)
    builder.add_node("respond", respond)

    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges("evaluate", evaluate_router, {
        "think": "think",
        "respond": "respond",
    })
    builder.add_conditional_edges("think", mod_tool_router, {
        "hitl": "hitl",
        "respond": "respond",
    })
    builder.add_edge("hitl", "think")
    builder.add_edge("respond", END)

    return builder.compile()
