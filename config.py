import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# OpenRouter
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

# Model strings — must use full OpenRouter format
PROPOSER_MODEL: str = "openai/gpt-5.2"
ADVERSARY_MODEL: str = "anthropic/claude-sonnet-4-5"
MODERATOR_MODEL: str = "google/gemini-2.5-flash"
INTERVIEWER_MODEL: str = "google/gemini-2.5-flash"

# Debate defaults
DEFAULT_MAX_ROUNDS: int = 5

# Memory
MEMORY_FILE_PATH: Path = Path(__file__).parent / "memory" / "memory.json"

# Rubric weights — must sum to 1.0
RUBRIC_WEIGHTS: dict[str, float] = {
    "constraint_adherence": 0.25,
    "technical_feasibility": 0.20,
    "operational_complexity": 0.20,
    "scalability_fit": 0.15,
    "evidence_quality": 0.10,
    "cost_efficiency": 0.10,
}

# Smart pause thresholds
HITL_SCORE_CLOSENESS_THRESHOLD: float = 0.3
HITL_MIN_ROUND_FOR_SMART_PAUSE: int = 2

# Dynamic graph caps
MAX_DEEP_DIVES: int = 1
MAX_EXTRA_ROUNDS: int = 2

# Search
SEARCH_MAX_RESULTS: int = 5
PROPOSER_SEARCH_QUERIES: int = 2
ADVERSARY_SEARCH_QUERIES: int = 5
