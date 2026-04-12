"""Abstract base class for Ari's brain backends.

All brain implementations (Claude, Gemma/Ollama, etc.) must implement
this interface so the daemon can switch between them via config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Optional


class Brain(ABC):
    """Base class for LLM brain backends."""

    @abstractmethod
    def ask(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Send a prompt and return (reply_text, session_id)."""

    @abstractmethod
    def ask_streaming(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Send a prompt and yield complete sentences as they arrive."""

    @abstractmethod
    def ask_and_speak(
        self,
        text: str,
        speaker,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Ask and speak the response with minimum latency. Returns full reply."""
