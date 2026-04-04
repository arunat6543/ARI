"""Abstract base class for Ari skills.

Every skill that Ari can perform (vision, camera movement, conversation,
etc.) inherits from :class:`Skill` and implements ``can_handle`` and
``handle``.

The ``context`` dict passed to ``handle`` carries shared resources so that
skills do not need to construct their own::

    context = {
        "speaker":       Speaker,        # ari.audio.speaker.Speaker
        "claude":        ClaudeClient,   # ari.brain.claude_client.ClaudeClient
        "camera_client": FifoClient,     # ari.ipc.fifo.FifoClient
        "scanner":       PersonScanner,  # ari.vision.scanner.PersonScanner
        "session_id":    str | None,     # current Claude conversation session
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Skill(ABC):
    """Base class for all Ari skills."""

    name: str

    @abstractmethod
    def can_handle(self, intent: dict) -> bool:
        """Return True if this skill should handle the given intent."""
        ...

    @abstractmethod
    def handle(self, text: str, intent: dict, context: dict) -> None:
        """Execute the skill.

        Parameters
        ----------
        text:
            The original user utterance.
        intent:
            The intent dict produced by :func:`ari.brain.intent.detect_intent`.
        context:
            Shared resources (speaker, claude client, camera, etc.).
        """
        ...
