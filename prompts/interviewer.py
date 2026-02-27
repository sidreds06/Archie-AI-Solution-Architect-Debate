from prompts.proposer import _format_memory


def build_first_question_prompt(problem: str, user_memory: dict) -> str:
    """Prompt for the very first clarifying question — no prior Q&A yet."""
    mem_lines = _format_memory(user_memory)
    return (
        "You are an architecture requirements analyst preparing for a debate between "
        "two AI architects. Your job is to identify the single most critical ambiguity "
        "in the user's problem.\n\n"
        f"USER'S PROBLEM:\n{problem}\n\n"
        f"WHAT WE ALREADY KNOW (from past sessions):\n{mem_lines}\n\n"
        "Ask ONE clarifying question — the most important thing you need to know "
        "to produce a good architecture proposal. Be specific and actionable.\n\n"
        "DO NOT ask about things already stated in the problem or user memory.\n\n"
        "Return ONLY this JSON (no markdown fences, no prose):\n"
        '{"question": "your specific question here", "done": false}'
    )


def build_followup_prompt(
    problem: str, user_memory: dict, qa_history: list[tuple[str, str]]
) -> str:
    """Prompt for follow-up questions, with full Q&A history for context."""
    mem_lines = _format_memory(user_memory)

    qa_block = ""
    for i, (q, a) in enumerate(qa_history, 1):
        qa_block += f"Q{i}: {q}\nA{i}: {a}\n\n"

    return (
        "You are an architecture requirements analyst. You are interviewing a user "
        "before a debate between two AI architects.\n\n"
        f"USER'S PROBLEM:\n{problem}\n\n"
        f"WHAT WE ALREADY KNOW (from past sessions):\n{mem_lines}\n\n"
        f"QUESTIONS AND ANSWERS SO FAR:\n{qa_block}\n"
        "Based on everything above, decide:\n"
        "- If you have enough context to run a high-quality architecture debate, "
        'return: {"question": null, "done": true}\n'
        "- If there is still a critical ambiguity, ask ONE more question.\n\n"
        "IMPORTANT:\n"
        "- Consider what the user's answers IMPLY, not just what they said literally. "
        "For example, if they said '3-person startup', don't ask about budget — it's implied.\n"
        "- DO NOT re-ask anything already answered or covered by prior answers.\n"
        "- DO NOT ask about things stated in the problem or user memory.\n"
        "- Aim for 3-5 total questions, but stop earlier if you have enough.\n\n"
        "Return ONLY this JSON (no markdown fences, no prose):\n"
        '{"question": "your specific question here", "done": false}\n'
        "or\n"
        '{"question": null, "done": true}'
    )
