"""Debate engine — orchestrates the full debate flow as generator methods.

Each method yields structured event dicts that the UI layer consumes.
No Rich, no console, no direct printing. Pure logic + event emission.
"""

from __future__ import annotations

import time
from typing import Any, Generator

from memory import manager as memory_manager
from state import DebateState

from core.interviewer import InterviewEngine
from core.proposer import ProposerEngine
from core.adversary import AdversaryEngine
from core.moderator import ModeratorEngine
from core.scoreboard import compute_scoreboard
from core.request_handler import process_requests, route_after_requests
from core.deep_dive import DeepDiveEngine
from core.verdict import VerdictEngine


Event = dict[str, Any]


class DebateEngine:
    """Stateful debate orchestrator.

    Usage from Gradio handlers:
        engine = DebateEngine(problem, max_rounds)
        # Interview phase
        for event in engine.start_interview():
            yield update_ui(event)
        for event in engine.submit_interview_answer(answer):
            yield update_ui(event)
        # ... repeat until interview done ...
        # Debate rounds
        for event in engine.run_round():
            yield update_ui(event)
        # ... handle HITL, checkpoints, etc ...
    """

    def __init__(self, problem: str, max_rounds: int = 5) -> None:
        self.state: DebateState = {
            "problem": problem,
            "round": 1,
            "max_rounds": max_rounds,
            "proposals": [],
            "last_proposer_solution": "",
            "last_adversary_solution": "",
            "hitl_pending": None,
            "hitl_answer": None,
            "user_memory": {},
            "debate_active": True,
            "final_verdict": None,
            "enriched_context": "",
            "score_history": [],
            "agent_requests": [],
            "pending_deep_dive": None,
            "deep_dives_used": 0,
            "momentum": {"proposer": 0.0, "adversary": 0.0},
            "dramatic_events": [],
            "user_interjection": None,
            # UI metadata
            "search_metadata": [],
            "timing": {"debate_start": time.time(), "round_times": []},
            "token_counts": {"proposer": [], "adversary": []},
            "hitl_history": [],
        }
        self._interview = InterviewEngine()
        self._proposer = ProposerEngine()
        self._adversary = AdversaryEngine()
        self._moderator = ModeratorEngine()
        self._deep_dive = DeepDiveEngine()
        self._verdict = VerdictEngine()
        self._interview_qa: list[tuple[str, str]] = []
        self._pending_hitl_verdict: dict | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def load_memory(self) -> dict:
        """Load memory and return it. Also sets it in state."""
        mem = memory_manager.load()
        self.state["user_memory"] = mem
        return mem

    # ------------------------------------------------------------------
    # Interview phase
    # ------------------------------------------------------------------
    def start_interview(self) -> Generator[Event, None, None]:
        """Generate the first interview question."""
        yield {"type": "phase_change", "phase": "interview"}
        for event in self._interview.generate_question(
            self.state["problem"],
            self.state["user_memory"],
            self._interview_qa,
            question_index=0,
        ):
            yield event

    def submit_interview_answer(self, answer: str) -> Generator[Event, None, None]:
        """Process an answer and generate the next question or finish."""
        last_q = self._interview.last_question
        if last_q and answer.strip():
            self._interview_qa.append((last_q, answer.strip()))

        idx = len(self._interview_qa)
        for event in self._interview.generate_question(
            self.state["problem"],
            self.state["user_memory"],
            self._interview_qa,
            question_index=idx,
        ):
            if event["type"] == "interview_done":
                # Assemble enriched context
                lines = [f"Q: {q}\nA: {a}" for q, a in self._interview_qa if a]
                self.state["enriched_context"] = "\n\n".join(lines)
            yield event

    def skip_interview(self) -> None:
        """Skip interview and move to debate."""
        self.state["enriched_context"] = ""

    # ------------------------------------------------------------------
    # Debate round
    # ------------------------------------------------------------------
    def run_round(self, direction: str = "") -> Generator[Event, None, None]:
        """Run one full round: proposer → adversary → moderator → scoreboard → requests.

        Yields events throughout. If HITL is needed, yields a hitl_request
        event and stops — caller must then call run_hitl_rescore().
        """
        round_start = time.time()
        current_round = self.state["round"]
        max_rounds = self.state["max_rounds"]

        if direction.strip():
            self.state["user_interjection"] = direction.strip()

        yield {"type": "round_start", "round": current_round, "max_rounds": max_rounds}

        # --- Proposer ---
        yield {"type": "phase_change", "phase": "proposer"}
        proposer_tokens = 0
        for event in self._proposer.run(self.state):
            if event["type"] == "token":
                proposer_tokens += 1
            yield event
        # Apply proposer updates to state
        if self._proposer.last_result:
            self._apply_updates(self._proposer.last_result)
        self.state["token_counts"]["proposer"].append(proposer_tokens)

        # --- Adversary ---
        yield {"type": "phase_change", "phase": "adversary"}
        adversary_tokens = 0
        for event in self._adversary.run(self.state):
            if event["type"] == "token":
                adversary_tokens += 1
            yield event
        # Apply adversary updates
        if self._adversary.last_result:
            self._apply_updates(self._adversary.last_result)
        self.state["token_counts"]["adversary"].append(adversary_tokens)

        # --- Moderator ---
        yield {"type": "phase_change", "phase": "moderator"}
        for event in self._moderator.score(self.state):
            yield event

        # Apply moderator updates
        if self._moderator.last_verdict:
            verdict = self._moderator.last_verdict
            from core.moderator import update_proposal_scores, maybe_force_hitl

            verdict = maybe_force_hitl(verdict, current_round)
            updated_proposals = update_proposal_scores(
                self.state["proposals"], verdict, current_round
            )
            self.state["proposals"] = updated_proposals
            self.state["final_verdict"] = verdict

            decision = verdict.get("decision", "continue")

            # Check for HITL
            if decision == "hitl":
                self._pending_hitl_verdict = verdict
                self.state["hitl_pending"] = verdict.get("hitl_question", "")
                yield {
                    "type": "hitl_request",
                    "question": verdict.get("hitl_question", "Please clarify a constraint."),
                    "scores": verdict.get("scores", {}),
                    "round": current_round,
                }
                # Don't continue — caller handles HITL
                return

            if decision == "end":
                self.state["debate_active"] = False

        # --- Scoreboard ---
        yield from self._run_scoreboard()

        # --- Request handler ---
        yield from self._run_request_handler()

        # Record round time
        self.state["timing"]["round_times"].append(time.time() - round_start)

        # Clear consumed fields
        self.state["user_interjection"] = None

        # Yield routing decision
        route = route_after_requests(self.state)
        yield {"type": "route", "decision": route}

        # If deep dive was requested, run it
        if route == "deep_dive":
            yield from self._run_deep_dive()

    def run_hitl_rescore(self, answer: str) -> Generator[Event, None, None]:
        """Re-score after HITL answer, potentially loop for another HITL."""
        self.state["hitl_answer"] = answer
        current_round = self.state["round"]

        old_scores = {}
        if self._pending_hitl_verdict:
            old_scores = self._pending_hitl_verdict.get("scores", {})

        # Re-score
        for event in self._moderator.score(self.state, hitl_answer=answer):
            yield event

        if self._moderator.last_verdict:
            verdict = self._moderator.last_verdict
            from core.moderator import update_proposal_scores

            updated_proposals = update_proposal_scores(
                self.state["proposals"], verdict, current_round
            )
            self.state["proposals"] = updated_proposals
            self.state["final_verdict"] = verdict

            new_scores = verdict.get("scores", {})

            # Record HITL impact
            self.state["hitl_history"].append({
                "round": current_round,
                "question": self.state.get("hitl_pending", ""),
                "answer": answer,
                "scores_before": old_scores,
                "scores_after": new_scores,
            })

            yield {
                "type": "hitl_rescore",
                "old_scores": old_scores,
                "new_scores": new_scores,
                "verdict": verdict,
            }

            decision = verdict.get("decision", "continue")

            # If still HITL, yield another request
            if decision == "hitl":
                self._pending_hitl_verdict = verdict
                yield {
                    "type": "hitl_request",
                    "question": verdict.get("hitl_question", "Please clarify."),
                    "scores": new_scores,
                    "round": current_round,
                }
                return

            if decision == "end":
                self.state["debate_active"] = False

        self._pending_hitl_verdict = None
        self.state["hitl_pending"] = None

        # Continue with scoreboard + requests
        yield from self._run_scoreboard()
        yield from self._run_request_handler()

        route = route_after_requests(self.state)
        yield {"type": "route", "decision": route}

        if route == "deep_dive":
            yield from self._run_deep_dive()

    # ------------------------------------------------------------------
    # Round increment
    # ------------------------------------------------------------------
    def increment_round(self) -> Event:
        """Advance round counter. Returns round info."""
        self.state["round"] += 1
        return {
            "type": "round_increment",
            "round": self.state["round"],
            "max_rounds": self.state["max_rounds"],
        }

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    def run_verdict(self) -> Generator[Event, None, None]:
        """Generate final verdict brief."""
        yield {"type": "phase_change", "phase": "verdict"}
        for event in self._verdict.run(self.state):
            yield event
        # Update memory
        if self._verdict.last_memory:
            self.state["user_memory"] = self._verdict.last_memory

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def is_debate_over(self) -> bool:
        if not self.state["debate_active"]:
            return True
        verdict = self.state.get("final_verdict") or {}
        return verdict.get("decision") == "end"

    @property
    def current_round(self) -> int:
        return self.state["round"]

    @property
    def max_rounds(self) -> int:
        return self.state["max_rounds"]

    def get_session_stats(self) -> dict:
        """Return session statistics for the verdict view."""
        timing = self.state.get("timing", {})
        elapsed = time.time() - timing.get("debate_start", time.time())
        search_meta = self.state.get("search_metadata", [])
        total_searches = sum(m.get("source_count", 0) for m in search_meta)

        return {
            "duration_seconds": elapsed,
            "rounds_played": self.state["round"],
            "max_rounds": self.state["max_rounds"],
            "total_searches": total_searches,
            "hitl_count": len(self.state.get("hitl_history", [])),
            "deep_dives": self.state.get("deep_dives_used", 0),
            "proposer_tokens": self.state.get("token_counts", {}).get("proposer", []),
            "adversary_tokens": self.state.get("token_counts", {}).get("adversary", []),
        }

    def _apply_updates(self, updates: dict) -> None:
        """Merge node return dict into state."""
        for key, value in updates.items():
            self.state[key] = value

    def _run_scoreboard(self) -> Generator[Event, None, None]:
        """Compute scoreboard and yield events."""
        result = compute_scoreboard(self.state)
        self._apply_updates(result)
        yield {
            "type": "scoreboard",
            "score_history": self.state["score_history"],
            "momentum": self.state["momentum"],
            "dramatic_events": result.get("dramatic_events", []),
        }

    def _run_request_handler(self) -> Generator[Event, None, None]:
        """Process agent requests and yield events."""
        result = process_requests(self.state)
        # Yield individual request events before applying
        for req in self.state.get("agent_requests", []):
            yield {
                "type": "agent_request",
                "request_type": req.get("request_type", ""),
                "agent": req.get("agent", ""),
                "detail": req.get("detail", ""),
            }
        self._apply_updates(result)

    def _run_deep_dive(self) -> Generator[Event, None, None]:
        """Run deep dive sub-round."""
        topic = self.state.get("pending_deep_dive", "")
        yield {"type": "deep_dive_start", "topic": topic}

        # Proposer deep dive
        yield {"type": "phase_change", "phase": "deep_dive_proposer"}
        for event in self._deep_dive.run_proposer(self.state):
            yield event
        if self._deep_dive.last_proposer_result:
            self._apply_updates(self._deep_dive.last_proposer_result)

        # Adversary deep dive
        yield {"type": "phase_change", "phase": "deep_dive_adversary"}
        for event in self._deep_dive.run_adversary(self.state):
            yield event
        if self._deep_dive.last_adversary_result:
            self._apply_updates(self._deep_dive.last_adversary_result)

        # Re-score with moderator
        yield {"type": "phase_change", "phase": "moderator"}
        for event in self._moderator.score(self.state):
            yield event
        if self._moderator.last_verdict:
            from core.moderator import update_proposal_scores
            verdict = self._moderator.last_verdict
            updated = update_proposal_scores(
                self.state["proposals"], verdict, self.state["round"]
            )
            self.state["proposals"] = updated
            self.state["final_verdict"] = verdict

            if verdict.get("decision") == "end":
                self.state["debate_active"] = False

        yield from self._run_scoreboard()
        yield from self._run_request_handler()

        route = route_after_requests(self.state)
        yield {"type": "route", "decision": route}
