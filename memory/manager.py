import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Allow running from archie/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MEMORY_FILE_PATH

_DEFAULT_MEMORY: dict = {}


def load() -> dict:
    """Read memory.json. If missing or malformed, create it with defaults and return defaults."""
    path = Path(MEMORY_FILE_PATH)
    if not path.exists():
        save(_DEFAULT_MEMORY.copy())
        return _DEFAULT_MEMORY.copy()
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        save(_DEFAULT_MEMORY.copy())
        return _DEFAULT_MEMORY.copy()


def save(memory: dict) -> None:
    """Write memory to disk. Always stamps last_updated with today's date."""
    memory["last_updated"] = date.today().isoformat()
    path = Path(MEMORY_FILE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def _additive_merge(base: dict, diff: dict) -> dict:
    """
    Merge diff into base without overwriting existing information.
    Handles arbitrary keys (free-form schema):
    - None values in diff: skip.
    - List values: append unique new items only.
    - Scalar values: set only if key is absent or None in base.
    Returns a new dict; does not mutate base.
    """
    result = {**base}

    for key, value in diff.items():
        if key == "last_updated" or value is None:
            continue
        if isinstance(value, list):
            existing = set(result.get(key, []))
            new_items = [item for item in value if item not in existing]
            result[key] = result.get(key, []) + new_items
        else:
            if result.get(key) is None:
                result[key] = value

    return result


def update(
    memory: dict,
    session_problem: str,
    final_solution: str,
    moderator_client: Any,
) -> dict:
    """
    Call Gemini to extract new preference information from this session.
    Merge additively into current memory. Save. Return updated dict.

    moderator_client: a langchain_openai.ChatOpenAI instance pointed at Gemini.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "You are a memory extraction assistant. "
        "Given the user's architecture problem, the winning solution, and the "
        "existing memory JSON, extract ONLY new facts, preferences, and constraints "
        "not already captured.\n\n"
        "Return a JSON object using whatever key names best describe what you found "
        "(e.g. 'cloud_providers', 'team_size', 'budget', 'off_limits', 'preferred_stack', "
        "'domain', 'constraints', 'notes', or anything else relevant). "
        "You are free to invent new keys — do not feel constrained by any prior schema.\n\n"
        "For list values, return only NEW items not already in memory. "
        "For scalar values, return the value only if it was explicitly stated and is not "
        "already in memory. Return null for any field with no new information. "
        "Return ONLY valid JSON with no markdown fences."
    )

    human = (
        f"EXISTING MEMORY:\n{json.dumps(memory, indent=2)}\n\n"
        f"SESSION PROBLEM:\n{session_problem}\n\n"
        f"WINNING SOLUTION:\n{final_solution}\n\n"
        "Extract new information to add to memory."
    )

    response = moderator_client.invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    raw = response.content.strip()

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        diff = json.loads(raw)
    except json.JSONDecodeError:
        diff = {}

    updated = _additive_merge(memory, diff)
    save(updated)
    return updated
