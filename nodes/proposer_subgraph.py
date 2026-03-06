"""Proposer subgraph — tool-calling agent with visible tool nodes.

Subgraph: START → think → tool_router → web_search/deep_dive (loop) → respond → END
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

import prompts.proposer as proposer_prompts
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROPOSER_MODEL
from state import DebateState
from tools.agent_tools import deep_dive, web_search
from ui.streaming import stream_to_panel

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


# ------------------------------------------------------------------
# Subgraph nodes
# ------------------------------------------------------------------


def think(state: DebateState) -> dict:
    """Call LLM with bound tools. Store AIMessage in _agent_messages."""
    messages = state.get("_agent_messages", [])

    # First call — build prompt from state
    if not messages:
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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

    client = _get_client()
    bound = client.bind_tools(_TOOLS)
    response: AIMessage = bound.invoke(messages)
    messages.append(response)

    return {"_agent_messages": messages}


def tool_router(state: DebateState) -> str:
    """Route to the appropriate tool node or to respond."""
    messages = state.get("_agent_messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        tool_name = messages[-1].tool_calls[0]["name"]
        if tool_name == "deep_dive":
            return "deep_dive"
        return "web_search"
    return "respond"


def exec_web_search(state: DebateState) -> dict:
    """Execute web_search tool calls and append ToolMessages."""
    messages = list(state.get("_agent_messages", []))
    last_msg = messages[-1]
    tool_map = {t.name: t for t in _TOOLS}

    from rich.console import Console
    from rich.live import Live
    from rich.spinner import Spinner

    console = Console()

    for tc in last_msg.tool_calls:
        if tc["name"] != "web_search":
            continue
        query = tc["args"].get("query", "")
        with Live(
            Spinner("dots", text=f"[dim]Proposer searching: {query}...[/dim]"),
            console=console, transient=True,
        ):
            result = tool_map["web_search"].invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        console.print(f"  [dim]Searched: {query}[/dim]")

    return {"_agent_messages": messages}


def exec_deep_dive(state: DebateState) -> dict:
    """Execute deep_dive tool calls and append ToolMessages."""
    messages = list(state.get("_agent_messages", []))
    last_msg = messages[-1]
    tool_map = {t.name: t for t in _TOOLS}

    from rich.console import Console
    from rich.live import Live
    from rich.spinner import Spinner

    console = Console()

    for tc in last_msg.tool_calls:
        if tc["name"] != "deep_dive":
            continue
        topic = tc["args"].get("topic", "")
        with Live(
            Spinner("dots", text=f"[dim]Proposer deep diving: {topic}...[/dim]"),
            console=console, transient=True,
        ):
            result = tool_map["deep_dive"].invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        console.print(f"  [dim]Deep dive: {topic}[/dim]")

    return {"_agent_messages": messages}


def respond(state: DebateState) -> dict:
    """Stream final response and update state."""
    from ui.banners import phase_banner, round_header

    messages = state.get("_agent_messages", [])
    current_round = state["round"]
    client = _get_client()

    # Round header on round 1 (adversary doesn't print it again)
    if current_round == 1 or not state.get("last_adversary_solution"):
        round_header(current_round, state["max_rounds"])

    phase_banner("Proposer", "is building an architecture...", "blue")

    # Stream a fresh final response for the Rich panel display
    # Drop the last AIMessage (tool-calling response) and re-stream
    stream_messages = [m for m in messages if not (isinstance(m, AIMessage) and not m.content)]
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].content:
        # Last message has content — re-stream from messages without it
        stream_messages = messages[:-1]
    else:
        stream_messages = messages

    title = f"[bold blue]Proposer — {PROPOSER_MODEL} — Round {current_round}[/bold blue]"
    solution = stream_to_panel(client, stream_messages, title, "blue")
    solution = solution.strip()

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
        "user_interjection": None,
        "debate_phase": "proposal_done",
        "_agent_messages": [],  # clear for next subgraph call
    }
    if request:
        updates["agent_requests"] = state.get("agent_requests", []) + [request]

    return updates


# ------------------------------------------------------------------
# Build subgraph
# ------------------------------------------------------------------


def build_proposer_subgraph():
    """Build and compile the proposer subgraph."""
    builder = StateGraph(DebateState)

    builder.add_node("think", think)
    builder.add_node("web_search", exec_web_search)
    builder.add_node("deep_dive", exec_deep_dive)
    builder.add_node("respond", respond)

    builder.add_edge(START, "think")
    builder.add_conditional_edges("think", tool_router, {
        "web_search": "web_search",
        "deep_dive": "deep_dive",
        "respond": "respond",
    })
    builder.add_edge("web_search", "think")
    builder.add_edge("deep_dive", "think")
    builder.add_edge("respond", END)

    return builder.compile()
