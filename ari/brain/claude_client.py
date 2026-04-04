"""Claude CLI client for Ari's brain.

Wraps the Claude CLI binary to send prompts (with optional images and
session continuations) and parse structured JSON responses.

Usage::

    from ari.brain.claude_client import ClaudeClient
    claude = ClaudeClient()
    reply, sid = claude.ask("What do you see?", image_path="/tmp/snap.jpg")
    reply2, sid = claude.ask("Tell me more", session_id=sid)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Optional

from ari.config import cfg
from ari.vision.camera import image_to_base64

log = logging.getLogger(__name__)


class ClaudeClient:
    """Thin wrapper around the Claude CLI for conversational queries."""

    def __init__(self) -> None:
        brain_cfg = cfg["brain"]
        self.cli_path: str = os.path.expanduser(brain_cfg["claude_cli"])
        self.model: str = brain_cfg["model"]
        self.timeout: int = int(brain_cfg["timeout"])
        self.system_prompt: str = brain_cfg["system_prompt"]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def ask(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Send a prompt to Claude and return (reply_text, session_id).

        Parameters
        ----------
        text:
            The user's text prompt.
        image_path:
            Optional path to a JPEG image.  If provided the image is
            Base64-encoded and prepended to the prompt so Claude can
            "see" it.
        session_id:
            Optional session ID from a previous call.  When provided the
            conversation is resumed (``--resume``).

        Returns
        -------
        tuple[str, str | None]
            The assistant's reply text and the session ID (which can be
            passed back to continue the conversation).  On failure the
            reply is a short error message and session_id may be ``None``.
        """
        # Build the prompt, optionally prepending the image data.
        prompt = text
        if image_path:
            b64 = image_to_base64(image_path)
            if b64:
                prompt = f"[image data:image/jpeg;base64,{b64}]\n{text}"

        cmd = self._build_command(prompt, session_id)
        log.debug("Claude CLI command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            log.warning("Claude CLI timed out after %ds", self.timeout)
            return ("Sorry, I took too long to think. Try again.", None)

        if result.returncode != 0:
            log.error("Claude CLI error (rc=%d): %s", result.returncode, result.stderr.strip())
            return ("Sorry, something went wrong on my end.", None)

        return self._parse_response(result.stdout)

    def ask_simple(self, text: str) -> str:
        """One-shot ask without session tracking or images.

        Returns only the reply text (no session ID).
        """
        reply, _ = self.ask(text)
        return reply

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _build_command(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> list[str]:
        """Assemble the CLI argument list."""
        cmd = [
            self.cli_path,
            "-p", prompt,
            "--bare",
            "--model", self.model,
            "--tools", "",
            "--system-prompt", self.system_prompt,
            "--output-format", "json",
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    @staticmethod
    def _parse_response(stdout: str) -> tuple[str, Optional[str]]:
        """Extract reply text and session_id from CLI JSON output.

        The CLI ``--output-format json`` emits a JSON object with at
        least ``result`` (the assistant text) and ``session_id``.
        """
        try:
            data = json.loads(stdout)
            reply = data.get("result", "").strip()
            sid = data.get("session_id")
            return (reply or "(empty response)", sid)
        except (json.JSONDecodeError, TypeError):
            # Fall back to treating the raw output as plain text.
            text = stdout.strip()
            return (text or "(empty response)", None)
