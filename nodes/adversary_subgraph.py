"""Adversary subgraph — tool-calling agent with visible tool nodes.

Subgraph: START → think → tool_router → web_search/deep_dive (loop) → respond → END
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

import prompts.adversary as adversary_prompts
from config import ADVERSARY_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
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
            model=ADVERSARY_MODEL,
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
                        "agent": "adversary",
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

        from prompts.proposer import build_debate_history

        debate_history = build_debate_history(
            state["proposals"], state.get("score_history", []), current_round
        )

        system_prompt = adversary_prompts.build_system_prompt(
            user_memory,
            momentum=momentum,
            current_round=current_round,
            max_rounds=state["max_rounds"],
        )
        human_prompt = adversary_prompts.build_critique_prompt(
            user_memory=user_memory,
            problem=state["problem"],
            proposer_solution=state["last_proposer_solution"],
            search_context="Use the web_search tool to gather evidence before responding.",
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
            Spinner("dots", text=f"[dim]Adversary searching: {query}...[/dim]"),
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
            Spinner("dots", text=f"[dim]Adversary deep diving: {topic}...[/dim]"),
            console=console, transient=True,
        ):
            result = tool_map["deep_dive"].invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        console.print(f"  [dim]Deep dive: {topic}[/dim]")

    return {"_agent_messages": messages}


def respond(state: DebateState) -> dict:
    """Stream final response and update state."""
    from ui.banners import phase_banner

    messages = state.get("_agent_messages", [])
    current_round = state["round"]
    client = _get_client()

    phase_banner("Adversary", "is preparing a critique...", "green")

    # Stream a fresh final response
    stream_messages = [m for m in messages if not (isinstance(m, AIMessage) and not m.content)]
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].content:
        stream_messages = messages[:-1]
    else:
        stream_messages = messages

    title = f"[bold green]Adversary — {ADVERSARY_MODEL} — Round {current_round}[/bold green]"
    solution = stream_to_panel(client, stream_messages, title, "green")
    solution = solution.strip()

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
        "debate_phase": "critique_done",
        "_agent_messages": [],
    }
    if request:
        updates["agent_requests"] = state.get("agent_requests", []) + [request]

    return updates


# ------------------------------------------------------------------
# Build subgraph
# ------------------------------------------------------------------


def build_adversary_subgraph():
    """Build and compile the adversary subgraph."""
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
