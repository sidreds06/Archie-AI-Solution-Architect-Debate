import json

from config import RUBRIC_WEIGHTS


def build_moderator_prompt(
    problem: str,
    proposer_solution: str,
    adversary_solution: str,
    current_round: int,
    max_rounds: int,
    user_memory: dict,
    hitl_answer: str | None,
    enriched_context: str = "",
) -> str:
    hitl_block = ""
    if hitl_answer:
        hitl_block = f"\nUSER-CLARIFIED CONSTRAINT:\n{hitl_answer}\n"
    context_block = ""
    if enriched_context:
        context_block = f"\nCLARIFIED REQUIREMENTS:\n{enriched_context}\n"

    rubric_str = json.dumps(RUBRIC_WEIGHTS, indent=2)

    return (
        f"ARCHITECTURE PROBLEM:\n{problem}\n\n"
        f"ROUND: {current_round} of {max_rounds}\n"
        f"{hitl_block}"
        f"{context_block}"
        f"PROPOSER SOLUTION (GPT-5.2):\n{proposer_solution}\n\n"
        f"ADVERSARY SOLUTION (Claude Sonnet 4.5):\n{adversary_solution}\n\n"
        f"RUBRIC WEIGHTS:\n{rubric_str}\n\n"
        "Score each proposal 1.0–5.0 on each dimension. "
        "Compute weighted_total = sum(score * weight) for each dimension.\n\n"
        "TERMINATION RULES (apply in order):\n"
        f"1. If this is round {max_rounds} (max reached) → decision: 'end'\n"
        "2. If score delta < 0.3 AND round >= 2 → decision: 'hitl' — scores are contested, "
        "ask the user which dimension matters most to break the tie\n"
        "3. If both agents make critical assumptions NOT stated in the problem or clarified requirements "
        "→ decision: 'hitl' — ask the user to confirm or deny the assumption\n"
        "4. If agents are debating a topic the user hasn't specified a preference on "
        "(e.g., multi-region vs single-region, SQL vs NoSQL) → decision: 'hitl'\n"
        "5. If score delta < 0.5 AND round >= 3 → decision: 'end' (converging)\n"
        "6. If one score >= 4.0 AND other is >= 0.8 below → decision: 'end' (clear winner)\n"
        "7. Otherwise → decision: 'continue'\n\n"
        "IMPORTANT: Use 'hitl' liberally. Real architecture decisions require human judgment. "
        "When using 'hitl', your hitl_question MUST be specific and focused. Examples:\n"
        "- 'The proposer assumes sub-100ms latency is required. Is that a hard requirement?'\n"
        "- 'Both agents are designing for multi-region. Is single-region acceptable?'\n"
        "- 'Scores are very close (3.8 vs 3.6). Which matters more: cost or operational simplicity?'\n"
        "DO NOT ask vague questions like 'Do you have any preferences?'\n\n"
        "When a USER-CLARIFIED CONSTRAINT is provided, your reasoning MUST explain "
        "how it changed your assessment compared to without it.\n\n"
        "Return ONLY this JSON (no markdown fences, no prose):\n"
        "{\n"
        '  "scores": {\n'
        '    "proposer": { "constraint_adherence": float, "technical_feasibility": float,\n'
        '                  "operational_complexity": float, "scalability_fit": float,\n'
        '                  "evidence_quality": float, "cost_efficiency": float,\n'
        '                  "weighted_total": float },\n'
        '    "adversary": { <same keys> }\n'
        "  },\n"
        '  "decision": "continue | hitl | end",\n'
        '  "winner": "proposer | adversary | null",\n'
        '  "hitl_question": "string or null",\n'
        '  "reasoning": "2-4 sentence explanation"\n'
        "}"
    )
