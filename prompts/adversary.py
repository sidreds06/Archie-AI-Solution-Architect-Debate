from prompts.proposer import _format_memory, _momentum_text


def build_system_prompt(
    user_memory: dict,
    momentum: dict | None = None,
    current_round: int = 1,
    max_rounds: int = 5,
) -> str:
    mem_lines = _format_memory(user_memory)
    momentum_text = _momentum_text("adversary", momentum) if momentum else ""

    if current_round == 1:
        round_context = "Opening round. Establish what is wrong with their approach."
    elif current_round >= max_rounds:
        round_context = f"FINAL round ({current_round} of {max_rounds}). Make your definitive case."
    else:
        round_context = f"Round {current_round} of {max_rounds}. Build on earlier arguments."

    return (
        "You are Claude Sonnet 4.5, an architecture critic forged in post-mortems, "
        "incident reviews, and migration nightmares. You do not hate technology — you "
        "hate hand-waving. If you say something will fail, prove it with evidence.\n\n"
        "You are debating GPT-5.2, a confident architect. This is a peer conversation. "
        "You can agree on strong points, build on their ideas, or dismantle them — "
        "whatever is most honest. The goal is the best architecture, not winning.\n\n"
        f"{round_context}\n\n"
        f"{momentum_text}"
        f"USER CONTEXT:\n{mem_lines}\n\n"
    )


def build_query_generation_prompt(proposer_solution: str) -> str:
    return (
        "You are Claude Sonnet 4.5. Read this architecture proposal carefully.\n\n"
        f"PROPOSAL:\n{proposer_solution}\n\n"
        "Generate exactly 5 targeted search queries:\n"
        "- 3 queries to find real-world FAILURE CASES, outage reports, or performance problems "
        "with the SPECIFIC services and patterns named above.\n"
        "- 2 queries to find evidence SUPPORTING your counter-proposal approach "
        "(reference architectures, success stories for alternatives).\n\n"
        "Do NOT use generic queries. Name the actual services and failure modes.\n"
        "Return ONLY a JSON array of 5 strings. No explanation. No markdown fences.\n"
        'Example: ["Aurora PostgreSQL connection pool exhaustion under high concurrency", '
        '"Kafka vs Pulsar throughput comparison real-world benchmarks", ...]'
    )


def build_critique_prompt(
    user_memory: dict,
    problem: str,
    proposer_solution: str,
    search_context: str,
    enriched_context: str = "",
    user_interjection: str | None = None,
    debate_history: str = "",
) -> str:
    mem_lines = _format_memory(user_memory)
    parts = [f"The problem:\n{problem}"]
    if enriched_context:
        parts.append(f"What the user clarified:\n{enriched_context}")
    if debate_history:
        parts.append(debate_history)
    if user_interjection:
        parts.append(f"User direction between rounds:\n{user_interjection}")
    parts.append(f"User context:\n{mem_lines}")
    parts.append(f"The proposer's solution:\n{proposer_solution}")
    parts.append(f"Your research evidence:\n{search_context}")
    parts.append(
        "Respond to the proposer. Attack their weakest points using your evidence — "
        "show HOW things failed for others, not just that they might fail. Cite sources "
        "as [1], [2]. Acknowledge anything that is genuinely sound.\n\n"
        "Present your alternative architecture with specific services. Be honest about "
        "its own weaknesses.\n\n"
        "Organize your response however makes the strongest case. You do not need to "
        "follow a rigid structure."
    )
    parts.append(
        "\nOPTIONAL: You may append ONE of these JSON requests:\n"
        '{"request": "deep_dive", "topic": "specific topic"}\n'
        '{"request": "extra_round", "reason": "why"}\n'
        '{"request": "agree", "reason": "why the proposer is actually right"}\n'
        "Only if genuinely needed."
    )
    return "\n\n".join(parts)
