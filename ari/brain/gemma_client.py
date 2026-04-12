"""Gemma/Ollama client for Ari brain -- local LLM inference.

Uses Ollama HTTP API directly via urllib (no extra pip packages).
Streams responses for low-latency TTS.

Usage::

    client = GemmaClient()
    reply, sid = client.ask("Hello")

    for sentence in client.ask_streaming("Tell me a story"):
        speaker.speak(sentence)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from typing import Generator, Optional

from ari.brain.base import Brain
from ari.config import cfg

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class GemmaClient(Brain):
    """Local LLM brain via Ollama HTTP API."""

    def __init__(self) -> None:
        brain_cfg = cfg["brain"]
        gemma_cfg = brain_cfg.get("gemma", {})

        self._host: str = gemma_cfg.get("host", "http://localhost:11434")
        self._model: str = gemma_cfg.get("model", "gemma3:1b")
        self._timeout: int = int(gemma_cfg.get("timeout", 30))
        self._system_prompt: str = brain_cfg["system_prompt"]

    # -- Brain interface: ask (blocking) ---------------------------------------

    def ask(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Send a prompt and return (reply_text, None).

        Ollama does not have persistent sessions, so session_id is ignored
        and always returns None.
        """
        try:
            reply = self._generate(text, stream=False)
            return (reply or "(empty response)", None)
        except Exception as e:
            log.error("Gemma ask error: %s", e)
            return ("Sorry, something went wrong on my end.", None)

    # -- Brain interface: ask_streaming ----------------------------------------

    def ask_streaming(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response and yield complete sentences."""
        buffer = ""

        try:
            for token in self._generate_stream(text):
                buffer += token

                # Check for sentence boundary
                while True:
                    match = _SENTENCE_RE.search(buffer)
                    if match:
                        sentence = buffer[:match.start() + 1].strip()
                        buffer = buffer[match.end():]
                        if sentence and len(sentence) > 5:
                            yield sentence
                    else:
                        break

            # Yield remaining text
            remaining = buffer.strip()
            if remaining:
                yield remaining

        except Exception as e:
            log.error("Gemma streaming error: %s", e)

    # -- Brain interface: ask_and_speak ----------------------------------------

    def ask_and_speak(
        self,
        text: str,
        speaker,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Stream Gemma output and speak each sentence immediately."""
        full_reply = []
        first = True

        for sentence in self.ask_streaming(text, image_path, session_id):
            full_reply.append(sentence)
            if first:
                log.info("First sentence ready, speaking...")
                first = False
            speaker._speak_stream(sentence)

        reply = " ".join(full_reply)
        return reply if reply else "Sorry, I didn't get a response."

    # -- Ollama HTTP API -------------------------------------------------------

    def _generate(self, prompt: str, stream: bool = False) -> str:
        """Non-streaming generate via Ollama API."""
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "system": self._system_prompt,
            "stream": False,
            "options": {"num_ctx": 2048},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError as e:
            log.error("Ollama connection error: %s", e)
            raise

    def _generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Streaming generate -- yields tokens as they arrive."""
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "system": self._system_prompt,
            "stream": True,
            "options": {"num_ctx": 2048},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            return
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as e:
            log.error("Ollama stream error: %s", e)
            raise
