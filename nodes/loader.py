from memory import manager as memory_manager
from state import DebateState


def load_memory(state: DebateState) -> dict:
    """LangGraph node. Loads memory.json into state."""
    memory = memory_manager.load()
    return {"user_memory": memory}
