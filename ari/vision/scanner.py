"""Person scanner for Ari — sweeps the pan-tilt mount and uses Claude Vision.

Scans through a set of configurable positions, captures an image at each,
and asks Claude (via the CLI) whether a person is visible.  If found,
fine-tunes the pan position based on where in the frame the person appears.

Usage::

    from ari.vision.scanner import PersonScanner
    scanner = PersonScanner(camera_client)
    result = scanner.find_person()
    if result:
        pan, tilt, description = result
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Tuple

from ari.config import cfg
from ari.vision.camera import image_to_base64


# Type alias for the FIFO-based camera client.  Any object that
# implements ``send(cmd: str) -> str | None`` will work.  In practice
# this is typically a thin wrapper around the named pipe.
class _FifoClientProtocol:
    """Minimal protocol expected from the camera daemon client."""
    def send(self, cmd: str) -> str | None: ...


class PersonScanner:
    """Scan the room with the pan-tilt mount to locate a person."""

    def __init__(self, camera_client: _FifoClientProtocol) -> None:
        """
        Parameters
        ----------
        camera_client:
            An object with a ``send(cmd)`` method that writes *cmd* to the
            camera daemon FIFO and returns the status response (or ``None``).
        """
        self._client = camera_client

        vcfg = cfg["vision"]
        self._scan_positions: list[list[int]] = vcfg["scan_positions"]
        self._settle_time: float = vcfg["scan_settle_time"]
        self._capture_wait: float = vcfg["scan_capture_wait"]
        self._fine_tune_offset: int = vcfg["fine_tune_offset"]

        bcfg = cfg["brain"]
        self._claude_cli: str = bcfg["claude_cli"]
        self._claude_model: str = bcfg["model"]
        self._claude_timeout: int = bcfg["timeout"]

    # -- Public API -----------------------------------------------------------

    def find_person(self) -> Tuple[int, int, str] | None:
        """Sweep through scan positions looking for a person.

        Returns
        -------
        tuple of (pan_us, tilt_us, description) if a person is found,
        or ``None`` if the sweep completes without a detection.
        """
        for pan_us, tilt_us in self._scan_positions:
            # Move camera to scan position
            self._client.send(f"set {pan_us} {tilt_us}")
            time.sleep(self._settle_time)

            # Capture an image
            filepath = f"/tmp/ari_scan_{pan_us}.jpg"
            self._client.send(f"capture {filepath}")
            time.sleep(self._capture_wait)

            b64 = image_to_base64(filepath)
            if b64 is None:
                continue

            # Ask Claude whether a person is present
            response = self.ask_vision(
                "Is there a person or human visible in this image?", b64
            )

            lines = response.strip().split("\n")
            first_line = lines[0].strip().upper() if lines else ""

            if not first_line.startswith("YES"):
                continue

            # Person found -- fine-tune the pan position
            frame_pos = lines[1].strip().upper() if len(lines) > 1 else "CENTER"
            description = lines[2].strip() if len(lines) > 2 else "I found someone!"

            if "LEFT" in frame_pos:
                pan_us += self._fine_tune_offset
            elif "RIGHT" in frame_pos:
                pan_us -= self._fine_tune_offset

            # Centre on the person
            self._client.send(f"set {pan_us} {tilt_us}")
            time.sleep(1.0)

            return pan_us, tilt_us, description

        # Nobody found -- return home
        self._client.send("home")
        time.sleep(1.0)
        return None

    def ask_vision(self, prompt: str, image_b64: str) -> str:
        """Send an image + prompt to Claude CLI for vision analysis.

        The model is instructed to reply with a strict YES/NO format:
          Line 1: YES or NO
          Line 2: LEFT, CENTER, or RIGHT (if YES)
          Line 3: brief description (if YES)

        Returns ``"NO"`` on any error so the scan can continue.
        """
        system_prompt = (
            "You are analyzing a camera image for person detection. "
            "Reply with ONLY 'YES' or 'NO' on the first line. "
            "If YES, on the second line say where in frame: LEFT, CENTER, or RIGHT. "
            "On the third line, briefly describe the person. Nothing else."
        )

        try:
            result = subprocess.run(
                [
                    self._claude_cli,
                    "-p", "--bare",
                    "--model", self._claude_model,
                    "--tools", "",
                    "--system-prompt", system_prompt,
                ],
                input=(
                    f"[Image as base64 JPEG: data:image/jpeg;base64,{image_b64}]"
                    f"\n\n{prompt}"
                ),
                capture_output=True,
                text=True,
                timeout=self._claude_timeout,
                env=os.environ,
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            return "NO"
