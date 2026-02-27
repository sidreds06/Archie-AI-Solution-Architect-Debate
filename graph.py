from langgraph.graph import END, START, StateGraph

from nodes.adversary import adversary
from nodes.deep_dive import deep_dive_adversary, deep_dive_proposer
from nodes.interviewer import interviewer
from nodes.loader import load_memory
from nodes.moderator import moderator
from nodes.proposer import proposer
from nodes.request_handler import handle_agent_requests
from nodes.round_increment import round_increment
from nodes.scoreboard import update_scoreboard
from nodes.verdict import verdict
from state import DebateState


def route_after_requests(state: DebateState) -> str:
    """
    Conditional edge router called after the request handler.
    HITL is now handled inline by the moderator, so this only routes
    to deep_dive, round_increment, or verdict.
    """
    # Deep dive takes priority if requested
    if state.get("pending_deep_dive"):
        return "deep_dive_proposer"

    # Check if debate ended (from agent "agree" or moderator "end")
    if not state.get("debate_active", True):
        return "verdict"

    verdict_data = state.get("final_verdict") or {}
    decision = verdict_data.get("decision", "continue")

    if decision == "end":
        return "verdict"
    else:
        # "continue" or any resolved post-hitl decision
        return "round_increment"


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Flow:
      START → load_memory → interviewer → proposer → adversary → moderator (+ inline HITL)
                                                                    ↓
                                                                scoreboard
                                                                    ↓
                                                            request_handler
                                                                    ↓
                                                          route_after_requests()
                                                          ├── deep_dive_proposer → deep_dive_adversary → moderator
                                                          ├── round_increment → proposer
                                                          └── verdict → END
    """
    builder = StateGraph(DebateState)

    # Register all nodes
    builder.add_node("load_memory", load_memory)
    builder.add_node("interviewer", interviewer)
    builder.add_node("proposer", proposer)
    builder.add_node("adversary", adversary)
    builder.add_node("moderator", moderator)
    builder.add_node("scoreboard", update_scoreboard)
    builder.add_node("request_handler", handle_agent_requests)
    builder.add_node("round_increment", round_increment)
    builder.add_node("deep_dive_proposer", deep_dive_proposer)
    builder.add_node("deep_dive_adversary", deep_dive_adversary)
    builder.add_node("verdict", verdict)

    # Linear chain: START → load_memory → interviewer → proposer → adversary → moderator
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "interviewer")
    builder.add_edge("interviewer", "proposer")
    builder.add_edge("proposer", "adversary")
    builder.add_edge("adversary", "moderator")

    # Moderator → scoreboard → request_handler (linear)
    builder.add_edge("moderator", "scoreboard")
    builder.add_edge("scoreboard", "request_handler")

    # Conditional routing after request processing
    builder.add_conditional_edges(
        "request_handler",
        route_after_requests,
        {
            "deep_dive_proposer": "deep_dive_proposer",
            "round_increment": "round_increment",
            "verdict": "verdict",
        },
    )

    # Deep dive sub-flow → loops back to moderator for re-scoring
    builder.add_edge("deep_dive_proposer", "deep_dive_adversary")
    builder.add_edge("deep_dive_adversary", "moderator")

    # After round increment, loop back to proposer for the next round
    builder.add_edge("round_increment", "proposer")

    # Verdict is terminal
    builder.add_edge("verdict", END)

    return builder.compile()
