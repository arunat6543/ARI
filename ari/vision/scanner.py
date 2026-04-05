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
from ari.vision.recognizer import PersonRecognizer, extract_person_name

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
        self._recognizer = PersonRecognizer()

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

    @property
    def recognizer(self) -> PersonRecognizer:
        """Access the person recognizer for registration etc."""
        return self._recognizer

    def find_person(self, target_name: str | None = None) -> Optional[Tuple[int, int, str]]:
        """Find a person by scanning with live video detection.

        Args:
            target_name: if set, only stop when this specific person is found
                        (uses reference photo comparison via Claude Vision)

        Returns (pan_us, tilt_us, description) or None.
        """
        try:
            return self._live_scan(target_name=target_name)
        except Exception as e:
            logger.warning("Live scan failed (%s), falling back to Claude scan", e)
            return self._claude_scan()

    def _live_scan(self, target_name: str | None = None) -> Optional[Tuple[int, int, str]]:
        """Scan using real-time YOLO detection + optional identity verification."""
        from ari.vision.detector import YoloDetector, LiveScanner

        logger.info("Starting live scan (target=%s)...", target_name or "any person")

        detector = YoloDetector(confidence=0.25)
        scanner = LiveScanner(detector, resolution=(640, 480))

        try:
            scanner.start()

            # Use direct servo if available, otherwise create one
            servo = self._servo
            if servo is None:
                from ari.hardware.servo import PanTilt
                servo = PanTilt()

            # Hold servos during scan
            servo.hold()

            # Pan through positions
            for pan_us, tilt_us in self._scan_positions:
                logger.info("Live scan: panning to %d", pan_us)
                servo.set_position(pan_us, tilt_us)
                time.sleep(1)  # let image stabilize

                # Check for person detection (5 seconds per position for YOLO)
                for _ in range(25):
                    time.sleep(0.2)
                    detections = [d for d in scanner.latest_detections
                                  if d.label == "person"]

                    if not detections:
                        continue

                    det = detections[0]
                    logger.info("Person detected at pan=%d (%s, %d%%)",
                                pan_us, det.position_in_frame, det.confidence * 100)

                    # If looking for a specific person, verify identity
                    if target_name and self._recognizer.has_reference(target_name):
                        frame = scanner.latest_frame
                        if frame is not None:
                            identified = self._recognizer.identify(frame, target_name)
                            if not identified:
                                logger.info("Person found but not %s, continuing...",
                                            target_name)
                                break  # not the right person, try next position
                            logger.info("Confirmed: this is %s!", target_name)

                    # Fine-tune pan
                    if det.position_in_frame == "LEFT":
                        pan_us += self._fine_tune_offset
                    elif det.position_in_frame == "RIGHT":
                        pan_us -= self._fine_tune_offset

                    servo.set_position(pan_us, tilt_us)
                    time.sleep(0.5)
                    scanner.capture_frame_as_jpeg("/tmp/ari_found.jpg")

                    desc = f"I found {target_name}!" if target_name else "I found someone!"
                    servo.release()
                    scanner.stop()
                    return pan_us, tilt_us, desc

            # Not found
            logger.info("Live scan: %s not found",
                        target_name or "no person")
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
