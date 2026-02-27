import json
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import INTERVIEWER_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from prompts.interviewer import build_first_question_prompt, build_followup_prompt
from state import DebateState

console = Console()

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
    """Parse a {"question": ..., "done": ...} response. Returns None on failure."""
    cleaned = _strip_fences(raw)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "done" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def interviewer(state: DebateState) -> dict:
    """LangGraph node. Adaptive interview — asks questions one at a time,
    adapting each question based on prior answers."""
    console.print(
        Panel(
            "[bold yellow]Before the debate begins, let me ask a few clarifying questions "
            "to ensure the agents address your actual needs.[/bold yellow]",
            title="[bold yellow]Requirements Interview[/bold yellow]",
            border_style="yellow",
        )
    )

    client = _get_client()
    user_memory = state.get("user_memory", {})
    problem = state["problem"]

    qa_history: list[tuple[str, str]] = []

    for i in range(_MAX_QUESTIONS):
        # Generate the next question (or decide to stop)
        if i == 0:
            prompt = build_first_question_prompt(problem, user_memory)
        else:
            prompt = build_followup_prompt(problem, user_memory, qa_history)

        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            response = client.invoke([HumanMessage(content=prompt)])

        parsed = _parse_question_response(response.content.strip())

        # Fallback: if LLM fails on first question, use fallback questions
        if parsed is None and i == 0:
            console.print("[dim]Using standard questions...[/dim]")
            for j, fallback_q in enumerate(_FALLBACK_QUESTIONS, 1):
                console.print(f"\n[yellow]Q{j}:[/yellow] {fallback_q}")
                answer = Prompt.ask(f"  [bold yellow]A{j}[/bold yellow]")
                qa_history.append((fallback_q, answer.strip()))
            break

        # If parse failed on a later question, just stop
        if parsed is None:
            break

        # If LLM says done, stop
        if parsed.get("done", False) or parsed.get("question") is None:
            break

        question = parsed["question"]
        console.print(f"\n[yellow]Q{i + 1}:[/yellow] {question}")
        answer = Prompt.ask(f"  [bold yellow]A{i + 1}[/bold yellow]")
        qa_history.append((question, answer.strip()))

    # Assemble enriched context from Q&A
    lines = [f"Q: {q}\nA: {a}" for q, a in qa_history if a]
    enriched_context = "\n\n".join(lines)

    console.print(
        Panel(
            "[bold green]Requirements captured. Starting the debate![/bold green]",
            border_style="green",
        )
    )

    return {"enriched_context": enriched_context}
