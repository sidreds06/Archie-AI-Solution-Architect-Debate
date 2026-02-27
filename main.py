import argparse
import sys
from pathlib import Path

# load_dotenv MUST run before any project imports that read os.environ at import time
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import ADVERSARY_MODEL, DEFAULT_MAX_ROUNDS, MODERATOR_MODEL, PROPOSER_MODEL
from graph import build_graph
from memory import manager as memory_manager

console = Console()


def print_header() -> None:
    console.print(
        Panel(
            f"[bold blue]Proposer:[/bold blue]   {PROPOSER_MODEL}\n"
            f"[bold green]Adversary:[/bold green]  {ADVERSARY_MODEL}\n"
            f"[bold magenta]Moderator:[/bold magenta]  {MODERATOR_MODEL}",
            title="[bold white]Archie — AI Solution Architect Debate[/bold white]",
            border_style="white",
        )
    )


def print_loaded_memory(memory: dict) -> None:
    lines = []
    if memory.get("cloud_providers"):
        lines.append(f"Cloud: {', '.join(memory['cloud_providers'])}")
    if memory.get("deployment_env"):
        lines.append(f"Deployment: {', '.join(memory['deployment_env'])}")
    if memory.get("domain"):
        lines.append(f"Domain: {memory['domain']}")
    if lines:
        console.print("[dim]Loaded memory: " + " | ".join(lines) + "[/dim]")
    else:
        console.print("[dim]No prior memory loaded.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archie: AI Architecture Debate CLI")
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help=f"Number of debate rounds (default: {DEFAULT_MAX_ROUNDS})",
    )
    args = parser.parse_args()

    print_header()

    problem = Prompt.ask("\n[bold]Describe your architecture problem[/bold]")
    if not problem.strip():
        console.print("[red]No problem provided. Exiting.[/red]")
        sys.exit(1)

    initial_memory = memory_manager.load()
    print_loaded_memory(initial_memory)
    console.print()

    initial_state = {
        "problem": problem,
        "round": 1,
        "max_rounds": args.rounds,
        "proposals": [],
        "last_proposer_solution": "",
        "last_adversary_solution": "",
        "hitl_pending": None,
        "hitl_answer": None,
        "user_memory": initial_memory,
        "debate_active": True,
        "final_verdict": None,
        # Interview
        "enriched_context": "",
        # Scoreboard
        "score_history": [],
        # Dynamic graph
        "agent_requests": [],
        "pending_deep_dive": None,
        "deep_dives_used": 0,
        # Momentum
        "momentum": {"proposer": 0.0, "adversary": 0.0},
        "dramatic_events": [],
        # User interjection
        "user_interjection": None,
    }

    app = build_graph()
    app.invoke(initial_state)


if __name__ == "__main__":
    # Ensure the archie/ directory is on sys.path so all imports resolve
    sys.path.insert(0, str(Path(__file__).parent))
    main()
