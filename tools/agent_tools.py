"""LangChain @tool functions for proposer and adversary agents.

These tools are bound to agents via bind_tools() so they can autonomously
search the web and deep-dive into topics during their response generation.
"""

from langchain_core.tools import tool

from tools.search import (
    format_results_for_prompt,
    run_search,
    search_for_case_studies,
    search_for_failures,
)


@tool
def web_search(query: str, search_type: str = "general") -> str:
    """Search the web for architecture-related information.

    Use this when you need evidence, case studies, failure reports, or
    technical references to strengthen your argument.

    Args:
        query: Search query string — be specific, name services and patterns.
        search_type: One of 'general', 'failures', or 'case_studies'.
    """
    if search_type == "failures":
        results = search_for_failures(query, max_results=5)
    elif search_type == "case_studies":
        results = search_for_case_studies(query, max_results=5)
    else:
        results = run_search(query, max_results=5, search_depth="advanced")
    return format_results_for_prompt(results)


@tool
def hitl(question: str) -> str:
    """Ask the user a clarifying question about their architecture requirements.

    Use this when:
    - Scores are very close and you need the user to break the tie
    - Agents are debating an unstated constraint or preference
    - You need clarification on a specific requirement

    Args:
        question: The question to ask the user. Must be specific and focused.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    console.print()
    console.print(
        Panel(
            f"[yellow]{question}[/yellow]",
            title="[bold yellow]Moderator Question[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    return Prompt.ask("[bold yellow]Your answer[/bold yellow]")


@tool
def deep_dive(topic: str) -> str:
    """Perform a focused deep dive on a specific architectural topic.

    Use this when a particular aspect needs deeper technical analysis —
    e.g., a specific failure mode, scaling pattern, or technology comparison.
    Runs multiple targeted searches and returns consolidated results.

    Args:
        topic: The specific topic to research deeply.
    """
    queries = [
        f"{topic} architecture best practices production",
        f"{topic} failure case study post-mortem",
        f"{topic} comparison alternatives benchmarks",
    ]
    blocks = []
    for q in queries:
        results = run_search(q, max_results=5, search_depth="advanced")
        blocks.append(f"Query: {q}\n{format_results_for_prompt(results)}")
    return "\n---\n".join(blocks)
