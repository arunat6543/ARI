"""Brain package -- LLM backends for Ari."""

from ari.config import cfg


def create_brain():
    """Create a brain instance based on cfg['brain']['engine'].

    Returns ClaudeClient, GemmaClient, or GeminiClient depending on config.
    """
    engine = cfg["brain"].get("engine", "claude")

    if engine == "gemma":
        from ari.brain.gemma_client import GemmaClient
        return GemmaClient()
    elif engine == "gemini":
        from ari.brain.gemini_client import GeminiClient
        return GeminiClient()
    else:
        from ari.brain.claude_client import ClaudeClient
        return ClaudeClient()
