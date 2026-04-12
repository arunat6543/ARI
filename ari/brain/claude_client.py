"""Claude CLI client for Ari's brain.

Two modes:
  - ask(): standard request-response (cold-starts CLI each time)
  - ask_streaming(): reads output in real-time, yields sentences as they complete
    so TTS can start before the full response is ready

Usage::

    claude = ClaudeClient()

    # Standard
    reply, sid = claude.ask("Hello")

    # Streaming — feed to TTS sentence by sentence
    for sentence in claude.ask_streaming("Tell me a story"):
        speaker.speak(sentence)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from typing import Generator, Optional

from ari.brain.base import Brain
from ari.config import cfg
from ari.vision.camera import image_to_base64

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ClaudeClient(Brain):
    """Wrapper around Claude CLI with streaming support."""

    def __init__(self) -> None:
        brain_cfg = cfg["brain"]
        self.cli_path: str = os.path.expanduser(brain_cfg["claude_cli"])
        self.model: str = brain_cfg["model"]
        self.timeout: int = int(brain_cfg["timeout"])
        self.system_prompt: str = brain_cfg["system_prompt"]
        self._last_session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._last_session_id

    # -- Standard ask (blocking) -----------------------------------------------

    def ask(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Send a prompt and return (reply_text, session_id)."""
        prompt = self._build_prompt(text, image_path)
        cmd = self._build_command(session_id, output_format="json")

        try:
            result = subprocess.run(
                cmd, input=prompt,
                capture_output=True, text=True,
                timeout=self.timeout, env=os.environ,
            )
        except subprocess.TimeoutExpired:
            log.warning("Claude CLI timed out")
            return ("Sorry, I took too long to think.", None)

        if result.returncode != 0:
            log.error("Claude CLI error (rc=%d): %s",
                      result.returncode, result.stderr.strip()[:200])
            return ("Sorry, something went wrong on my end.", None)

        reply, sid = self._parse_json_response(result.stdout)
        self._last_session_id = sid
        return reply, sid

    # -- Streaming ask (yields sentences) --------------------------------------

    def ask_streaming(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Send a prompt and yield complete sentences as they arrive.

        This starts the CLI, reads stdout character by character, and
        yields each sentence as soon as it's complete (ends with . ! or ?).
        TTS can start on the first sentence while Claude is still generating.
        """
        prompt = self._build_prompt(text, image_path)
        cmd = self._build_command(session_id, output_format="text")

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=os.environ,
            )
            proc.stdin.write(prompt)
            proc.stdin.close()

            # Read output and buffer until sentence boundary
            buffer = ""
            for char in iter(lambda: proc.stdout.read(1), ""):
                buffer += char

                # Check for sentence boundary
                if char in ".!?" and len(buffer.strip()) > 5:
                    # Peek at next char — if it's space or EOF, yield sentence
                    next_char = proc.stdout.read(1)
                    if next_char == "" or next_char in " \n\t":
                        sentence = buffer.strip()
                        if sentence:
                            yield sentence
                        buffer = ""
                    else:
                        buffer += next_char

            # Yield any remaining text
            remaining = buffer.strip()
            if remaining:
                yield remaining

            proc.wait(timeout=5)

        except subprocess.TimeoutExpired:
            log.error("Claude streaming timed out")
            try:
                proc.kill()
            except Exception:
                pass
        except Exception as e:
            log.error("Claude streaming error: %s", e)

    def ask_and_speak(
        self,
        text: str,
        speaker,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Ask Claude and speak the response with minimum latency.

        Streams Claude's output, and as soon as the first sentence is
        complete, starts speaking it while Claude continues generating.

        Returns the full reply text.
        """
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

    # -- Simple ask ------------------------------------------------------------

    def ask_simple(self, text: str) -> str:
        """One-shot ask, returns only reply text."""
        reply, _ = self.ask(text)
        return reply

    # -- Internals -------------------------------------------------------------

    def _build_prompt(self, text: str, image_path: Optional[str] = None) -> str:
        """Build prompt text, optionally with base64 image."""
        if image_path:
            b64 = image_to_base64(image_path)
            if b64:
                return f"[image data:image/jpeg;base64,{b64}]\n{text}"
        return text

    def _build_command(
        self,
        session_id: Optional[str] = None,
        output_format: str = "text",
    ) -> list[str]:
        """Build CLI argument list."""
        cmd = [
            self.cli_path,
            "-p",
            "--bare",
            "--model", self.model,
            "--tools", "",
            "--system-prompt", self.system_prompt,
        ]
        if output_format == "json":
            cmd.append("--output-format")
            cmd.append("json")
        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    @staticmethod
    def _parse_json_response(stdout: str) -> tuple[str, Optional[str]]:
        """Extract reply and session_id from JSON output."""
        try:
            data = json.loads(stdout)
            reply = data.get("result", "").strip()
            sid = data.get("session_id")
            return (reply or "(empty response)", sid)
        except (json.JSONDecodeError, TypeError):
            text = stdout.strip()
            return (text or "(empty response)", None)
