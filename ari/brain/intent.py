"""Intent detection for Ari.

Classifies user utterances into one of several intent categories so the
main loop can dispatch to the right skill handler.

Usage::

    from ari.brain.intent import detect_intent
    result = detect_intent("look left please")
    # {"intent": "camera_direction", "direction": "left"}
"""

from __future__ import annotations

import logging
from typing import Optional

from ari.config import cfg

log = logging.getLogger(__name__)


def contains_phrase(text: str, phrases: list[str]) -> bool:
    """Return True if *text* contains any of the given *phrases*.

    Comparison is case-insensitive.
    """
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in phrases)


def detect_intent(text: str) -> dict:
    """Classify *text* into an intent dict.

    Returns a dict with at least an ``"intent"`` key.  Possible intents:

    * ``"sleep"`` -- the user wants to end the conversation.
    * ``"camera_direction"`` -- the user asked to move the camera.
      Includes a ``"direction"`` key (e.g. ``"left"``, ``"home"``).
    * ``"find_person"`` -- the user wants Ari to find someone.
    * ``"vision"`` -- the user wants Ari to look at / describe something.
    * ``"conversation"`` -- general chat (fallback).
    """
    text_lower = text.lower()

    # --- Sleep ----------------------------------------------------------- #
    sleep_phrases: list[str] = cfg["wake"]["sleep_phrases"]
    if contains_phrase(text, sleep_phrases):
        return {"intent": "sleep"}

    # --- Camera direction ------------------------------------------------ #
    camera_directions: dict[str, str] = cfg["vision"]["camera_directions"]
    for phrase, direction in camera_directions.items():
        if phrase.lower() in text_lower:
            log.debug("Intent: camera_direction -> %s", direction)
            return {"intent": "camera_direction", "direction": direction}

    # --- Find person ----------------------------------------------------- #
    find_keywords: list[str] = cfg["vision"]["find_keywords"]
    if contains_phrase(text, find_keywords):
        return {"intent": "find_person"}

    # --- Vision (general) ------------------------------------------------ #
    vision_keywords: list[str] = cfg["vision"]["keywords"]
    if contains_phrase(text, vision_keywords):
        return {"intent": "vision"}

    # --- Default: conversation ------------------------------------------- #
    return {"intent": "conversation"}
