"""Person and object scanner for Ari.

Two scanning modes:
  1. **Live scan** (fast, ~5 seconds) — uses OpenCV face detection at 30 FPS
     while smoothly panning the camera. No API calls needed for detection.
  2. **Claude scan** (slower, ~40 seconds) — captures photos at each position
     and asks Claude Vision. Use when live scan isn't available.

After finding a target, Claude Vision is used to describe what was found.

Usage::

    from ari.vision.scanner import PersonScanner
    scanner = PersonScanner(camera_client)
    result = scanner.find_person()  # uses live scan by default
    if result:
        pan, tilt, description = result
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Optional, Tuple

from ari.config import cfg
from ari.vision.camera import image_to_base64

logger = logging.getLogger(__name__)


class PersonScanner:
    """Scan the room with the pan-tilt mount to locate a person or object."""

    def __init__(self, camera_client=None, servo=None) -> None:
        """
        Parameters
        ----------
        camera_client:
            Object with a ``send(cmd)`` method for the camera daemon FIFO.
        servo:
            Optional PanTilt instance for direct servo control (used in live scan).
        """
        self._client = camera_client
        self._servo = servo

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

    def find_person(self) -> Optional[Tuple[int, int, str]]:
        """Find a person by scanning with live video detection.

        Smoothly pans the camera while running face detection on every frame
        at ~30 FPS. Much faster than the Claude-based scan.

        Returns (pan_us, tilt_us, description) or None.
        """
        try:
            return self._live_scan()
        except Exception as e:
            logger.warning("Live scan failed (%s), falling back to Claude scan", e)
            return self._claude_scan()

    def _live_scan(self) -> Optional[Tuple[int, int, str]]:
        """Scan using real-time OpenCV face detection + picamera2."""
        from ari.vision.detector import FaceDetector, LiveScanner

        logger.info("Starting live scan...")

        # Stop camera daemon temporarily (we need direct camera access)
        self._client.send("position")  # save current position
        # Note: camera daemon uses rpicam-still which is separate from picamera2
        # They can coexist as long as we don't capture simultaneously

        detector = FaceDetector()
        scanner = LiveScanner(detector, resolution=(640, 480))

        try:
            scanner.start()

            # Use direct servo if available, otherwise create one
            servo = self._servo
            if servo is None:
                from ari.hardware.servo import PanTilt
                servo = PanTilt()

            # Hold servos during scan so they don't drift between positions
            servo.hold()

            # Pan through positions smoothly
            for pan_us, tilt_us in self._scan_positions:
                logger.info("Live scan: panning to %d", pan_us)
                servo.set_position(pan_us, tilt_us)

                # Check for detection while settling
                detection = scanner.wait_for_detection(
                    timeout=self._settle_time + 1.0,
                    label="person"
                )

                if detection:
                    logger.info("Person found at pan=%d! Position: %s",
                                pan_us, detection.position_in_frame)

                    if detection.position_in_frame == "LEFT":
                        pan_us += self._fine_tune_offset
                    elif detection.position_in_frame == "RIGHT":
                        pan_us -= self._fine_tune_offset

                    servo.set_position(pan_us, tilt_us)
                    time.sleep(0.5)
                    scanner.capture_frame_as_jpeg("/tmp/ari_found.jpg")

                    servo.release()
                    scanner.stop()
                    return pan_us, tilt_us, "I found someone!"

            # Not found
            logger.info("Live scan: no person found")
            servo.home()
            servo.release()
            scanner.stop()
            return None

        except Exception as e:
            logger.error("Live scan error: %s", e)
            if servo:
                servo.release()
            scanner.stop()
            raise

    def _claude_scan(self) -> Optional[Tuple[int, int, str]]:
        """Fallback: scan using Claude Vision at each position (slower)."""
        logger.info("Starting Claude-based scan...")

        for pan_us, tilt_us in self._scan_positions:
            self._client.send(f"set {pan_us} {tilt_us}")
            time.sleep(self._settle_time)

            filepath = f"/tmp/ari_scan_{pan_us}.jpg"
            self._client.send(f"capture {filepath}")
            time.sleep(self._capture_wait)

            b64 = image_to_base64(filepath)
            if b64 is None:
                continue

            response = self.ask_vision(
                "Is there a person or human visible in this image?", b64
            )

            lines = response.strip().split("\n")
            first_line = lines[0].strip().upper() if lines else ""

            if not first_line.startswith("YES"):
                continue

            frame_pos = lines[1].strip().upper() if len(lines) > 1 else "CENTER"
            description = lines[2].strip() if len(lines) > 2 else "I found someone!"

            if "LEFT" in frame_pos:
                pan_us += self._fine_tune_offset
            elif "RIGHT" in frame_pos:
                pan_us -= self._fine_tune_offset

            self._client.send(f"set {pan_us} {tilt_us}")
            time.sleep(1.0)

            # Clean up scan images
            for p, _ in self._scan_positions:
                try:
                    os.unlink(f"/tmp/ari_scan_{p}.jpg")
                except OSError:
                    pass

            return pan_us, tilt_us, description

        self._client.send("home")
        time.sleep(1.0)
        return None

    def ask_vision(self, prompt: str, image_b64: str) -> str:
        """Send an image + prompt to Claude CLI for vision analysis."""
        system_prompt = (
            "You are analyzing a camera image for person detection. "
            "Reply with ONLY 'YES' or 'NO' on the first line. "
            "If YES, on the second line say where in frame: LEFT, CENTER, or RIGHT. "
            "On the third line, briefly describe the person. Nothing else."
        )

        try:
            result = subprocess.run(
                [self._claude_cli, "-p", "--bare",
                 "--model", self._claude_model, "--tools", "",
                 "--system-prompt", system_prompt],
                input=f"[Image as base64 JPEG: data:image/jpeg;base64,{image_b64}]\n\n{prompt}",
                capture_output=True, text=True,
                timeout=self._claude_timeout, env=os.environ,
            )
            return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            return "NO"
