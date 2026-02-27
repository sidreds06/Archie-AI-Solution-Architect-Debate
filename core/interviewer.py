"""Interviewer logic — pure generator, no UI. Yields structured events."""

from __future__ import annotations

import json
import re
from typing import Any, Generator

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import INTERVIEWER_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from prompts.interviewer import build_first_question_prompt, build_followup_prompt

Event = dict[str, Any]

_MAX_QUESTIONS = 6

_FALLBACK_QUESTIONS = [
    "What is your expected scale (users, requests/sec, data volume)?",
    "Do you have budget constraints or strong cost sensitivity?",
    "What is your team's size and primary technical expertise?",
]


def _get_client() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=INTERVIEWER_MODEL,
    )


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _parse_question_response(raw: str) -> dict | None:
    cleaned = _strip_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "done" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


class InterviewEngine:
    """Runs interviewer logic as a generator yielding question events."""

    def __init__(self) -> None:
        self.last_question: str | None = None
        self._client: ChatOpenAI | None = None
        self._used_fallback = False

    def generate_question(
        self,
        problem: str,
        user_memory: dict,
        qa_history: list[tuple[str, str]],
        question_index: int,
    ) -> Generator[Event, None, None]:
        """Generate the next interview question (or finish).

        Yields:
            interview_question: a new question for the user
            interview_done: interview is complete
            interview_fallback: using fallback questions
        """
        if self._client is None:
            self._client = _get_client()

        # Check if we've hit the max
        if question_index >= _MAX_QUESTIONS:
            self.last_question = None
            yield {"type": "interview_done"}
            return

        # Generate question via LLM
        if question_index == 0:
            prompt = build_first_question_prompt(problem, user_memory)
        else:
            prompt = build_followup_prompt(problem, user_memory, qa_history)

        yield {"type": "status", "agent": "interviewer", "message": "Thinking..."}
        response = self._client.invoke([HumanMessage(content=prompt)])
        parsed = _parse_question_response(response.content.strip())

        # Fallback: if LLM fails on first question, use fallback questions
        if parsed is None and question_index == 0:
            self._used_fallback = True
            for i, fallback_q in enumerate(_FALLBACK_QUESTIONS):
                self.last_question = fallback_q
                yield {
                    "type": "interview_question",
                    "question": fallback_q,
                    "index": i + 1,
                    "is_fallback": True,
                }
                # For fallback, we yield all questions at once
                # The handler needs to collect all answers
            return

        # If parse failed on a later question, just stop
        if parsed is None:
            self.last_question = None
            yield {"type": "interview_done"}
            return

        # If LLM says done, stop
        if parsed.get("done", False) or parsed.get("question") is None:
            self.last_question = None
            yield {"type": "interview_done"}
            return

        question = parsed["question"]
        self.last_question = question
        yield {
            "type": "interview_question",
            "question": question,
            "index": question_index + 1,
            "is_fallback": False,
        }
