from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


def _replace_messages(left: list, right: list) -> list:
    """Reducer: always replace agent messages (no accumulation)."""
    return right


class DebateState(TypedDict):
    # --- Core ---
    problem: str
    round: int
    max_rounds: int
    proposals: list[dict]          # each: {agent, model, round, solution, score}
    last_proposer_solution: str
    last_adversary_solution: str
    hitl_pending: str | None
    hitl_answer: str | None
    user_memory: dict
    debate_active: bool
    final_verdict: dict | None

    # --- Interview ---
    enriched_context: str           # assembled Q&A from pre-debate interview

    # --- Scoreboard ---
    score_history: list[dict]       # [{round, proposer_score, adversary_score, delta}]

    # --- Dynamic graph ---
    agent_requests: list[dict]      # [{agent, request_type, detail}]

    # --- Momentum / personality ---
    momentum: dict                  # {"proposer": float, "adversary": float}
    dramatic_events: list[str]      # event strings for rendering

    # --- User input between rounds ---
    user_interjection: str | None   # typed by user at round checkpoint

    # --- Hub-spoke orchestration ---
    debate_phase: str               # current phase for moderator hub routing

    # --- UI metadata (used by Gradio smart UI) ---
    search_metadata: list[dict]     # [{agent, round, queries: [str], source_count: int}]
    timing: dict                    # {"debate_start": float, "round_times": [float]}
    token_counts: dict              # {"proposer": [int], "adversary": [int]} per round
    hitl_history: list[dict]        # [{round, question, answer, scores_before, scores_after}]

    # --- Internal: agent loop messages (reset per subgraph call) ---
    _agent_messages: Annotated[list, _replace_messages]
