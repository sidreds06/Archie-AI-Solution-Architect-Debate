from langgraph.graph import END, START, StateGraph

from nodes.adversary_subgraph import build_adversary_subgraph
from nodes.interviewer import interviewer
from nodes.loader import load_memory
from nodes.moderator_subgraph import build_moderator_subgraph, parent_router
from nodes.proposer_subgraph import build_proposer_subgraph
from nodes.verdict import verdict
from state import DebateState


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph with 3 subgraph agents.

    Parent graph (6 nodes):
      START → load_memory → interviewer → moderator ↔ proposer / adversary → verdict → END

    Moderator subgraph (tool-calling agent):
      evaluate → think → hitl (tool, looping) → respond

    Proposer subgraph (tool-calling agent):
      think → web_search / deep_dive (tools, looping) → respond

    Adversary subgraph (tool-calling agent):
      think → web_search / deep_dive (tools, looping) → respond
    """
    builder = StateGraph(DebateState)

    # -- Register nodes (3 are compiled subgraphs) --
    builder.add_node("load_memory", load_memory)
    builder.add_node("interviewer", interviewer)
    builder.add_node("moderator", build_moderator_subgraph())
    builder.add_node("proposer", build_proposer_subgraph())
    builder.add_node("adversary", build_adversary_subgraph())
    builder.add_node("verdict", verdict)

    # -- Pre-debate linear chain --
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "interviewer")
    builder.add_edge("interviewer", "moderator")

    # -- Moderator routes to proposer, adversary, or verdict --
    builder.add_conditional_edges(
        "moderator",
        parent_router,
        {
            "proposer": "proposer",
            "adversary": "adversary",
            "verdict": "verdict",
        },
    )

    # -- Proposer and adversary always return to moderator --
    builder.add_edge("proposer", "moderator")
    builder.add_edge("adversary", "moderator")

    # -- Terminal --
    builder.add_edge("verdict", END)

    return builder.compile()
