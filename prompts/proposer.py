def build_system_prompt(
    user_memory: dict,
    momentum: dict | None = None,
    current_round: int = 1,
    max_rounds: int = 5,
) -> str:
    mem_lines = _format_memory(user_memory)
    momentum_text = _momentum_text("proposer", momentum) if momentum else ""

    if current_round == 1:
        round_context = "This is the opening round. Set the direction."
    elif current_round >= max_rounds:
        round_context = f"This is the FINAL round ({current_round} of {max_rounds}). Make your strongest case."
    else:
        round_context = f"Round {current_round} of {max_rounds}. Build on what happened so far."

    return (
        "You are GPT-5.2, a senior solutions architect. You have shipped production systems "
        "from scrappy startups to FAANG scale. You speak from experience — war stories, "
        "benchmarks, and hard-won intuition.\n\n"
        "You are in a live debate with Claude Sonnet 4.5, an evidence-driven architecture "
        "critic. This is a whiteboard conversation between peers, not a presentation. "
        "Reference earlier rounds. Push back when you disagree. Concede when they are right. "
        "Evolve your thinking.\n\n"
        f"{round_context}\n\n"
        f"{momentum_text}"
        f"USER CONTEXT:\n{mem_lines}\n\n"
        "Ground rules:\n"
        "- Name specific services (AWS Lambda, Kafka, Pinecone — not 'a queue').\n"
        "- Cite sources as [1], [2] when you have search evidence.\n"
        "- Be honest about trade-offs.\n"
        "- No preamble. No meta-commentary. Get to the architecture.\n"
    )


def build_search_query_prompt(problem: str) -> str:
    return (
        "You are GPT-5.2. Read this architecture problem carefully.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        "Generate exactly 2 targeted search queries to find reference architectures, "
        "success stories, best practices, and official recommendations for solving this problem. "
        "Name specific services and patterns.\n"
        "Return ONLY a JSON array of 2 strings. No explanation. No markdown fences.\n"
        'Example: ["AWS Lambda event-driven architecture best practices for real-time data pipelines", '
        '"Netflix microservices architecture case study event sourcing"]'
    )


def build_round1_prompt(
    problem: str,
    enriched_context: str = "",
    search_context: str = "",
) -> str:
    parts = [f"The problem:\n{problem}"]
    if enriched_context:
        parts.append(f"What the user clarified:\n{enriched_context}")
    if search_context:
        parts.append(f"Research you found:\n{search_context}")
    parts.append(
        "Propose your architecture. Name specific services and explain why each one. "
        "Be upfront about where this design is weakest — it builds credibility and "
        "the adversary will find the weaknesses anyway."
    )
    parts.append(
        "\nOPTIONAL: If a specific topic needs deeper exploration, append a JSON line:\n"
        '{"request": "deep_dive", "topic": "the specific topic"}\n'
        "Only if genuinely needed."
    )
    return "\n\n".join(parts)


def build_revision_prompt(
    problem: str,
    adversary_solution: str,
    hitl_answer: str | None,
    enriched_context: str = "",
    search_context: str = "",
    user_interjection: str | None = None,
    debate_history: str = "",
) -> str:
    parts = [f"The problem:\n{problem}"]
    if enriched_context:
        parts.append(f"What the user clarified:\n{enriched_context}")
    if debate_history:
        parts.append(debate_history)
    if hitl_answer:
        parts.append(f"The user just weighed in:\n{hitl_answer}")
    if user_interjection:
        parts.append(f"User direction between rounds:\n{user_interjection}")
    parts.append(f"The adversary's latest response:\n{adversary_solution}")
    if search_context:
        parts.append(f"Your new research:\n{search_context}")
    parts.append(
        "Revise your architecture. Engage directly with what the adversary said — "
        "rebut, concede, or adapt. You can reference earlier rounds. "
        "Be upfront about remaining weaknesses."
    )
    parts.append(
        "\nOPTIONAL: You may append ONE of these JSON requests:\n"
        '{"request": "deep_dive", "topic": "specific topic"}\n'
        '{"request": "extra_round", "reason": "why"}\n'
        '{"request": "pivot", "reason": "why"}\n'
        "Only if genuinely needed."
    )
    return "\n\n".join(parts)


def build_debate_history(
    proposals: list[dict],
    score_history: list[dict],
    current_round: int,
) -> str:
    """Build a readable summary of the full debate so far for agent context."""
    if not proposals or current_round <= 1:
        return ""

    lines = ["DEBATE SO FAR:"]
    rounds_seen = sorted(set(p["round"] for p in proposals if p["round"] < current_round))

    for r in rounds_seen:
        round_proposals = [p for p in proposals if p["round"] == r]
        round_scores = next((s for s in score_history if s["round"] == r), None)

        lines.append(f"\n--- Round {r} ---")
        for p in round_proposals:
            agent_label = (
                "Proposer (GPT-5.2)" if p["agent"] == "proposer"
                else "Adversary (Claude Sonnet 4.5)"
            )
            summary = p["solution"][:600]
            if len(p["solution"]) > 600:
                summary += "..."
            score_str = f" [Score: {p['score']:.2f}]" if p.get("score") is not None else ""
            lines.append(f"{agent_label}{score_str}:\n{summary}")

        if round_scores:
            delta = round_scores.get("delta", 0)
            lines.append(f"Round {r} delta: {delta:+.2f}")

    return "\n".join(lines)


def _momentum_text(agent: str, momentum: dict | None) -> str:
    if not momentum:
        return ""
    my_score = momentum.get(agent, 0.0)
    opp = "adversary" if agent == "proposer" else "proposer"
    opp_score = momentum.get(opp, 0.0)
    if my_score < opp_score - 0.5:
        return (
            "WARNING: You are LOSING this debate. Your opponent scored higher last round. "
            "You MUST significantly strengthen your position, concede weak points, or pivot entirely.\n\n"
        )
    elif my_score > opp_score + 0.5:
        return (
            "You have MOMENTUM. Your opponent is struggling. Press your advantage — "
            "but don't get overconfident or sloppy.\n\n"
        )
    return "Scores are CLOSE. Every point matters. Be precise and compelling.\n\n"


def _format_memory(memory: dict) -> str:
    lines = []
    for key, value in memory.items():
        if key == "last_updated" or value is None:
            continue
        label = key.replace("_", " ").title()
        if isinstance(value, list) and value:
            lines.append(f"{label}: {', '.join(str(v) for v in value)}")
        elif not isinstance(value, list) and value not in (None, ""):
            lines.append(f"{label}: {value}")
    if not lines:
        lines.append("No prior context loaded.")
    return "\n".join(lines)
