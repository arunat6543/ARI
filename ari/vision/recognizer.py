"""Person recognition for Ari — identifies specific people using Claude Vision.

Flow:
  1. YOLO detects a person in frame
  2. Crop the person from the frame
  3. Send cropped image + reference photo to Claude
  4. Claude says if it's a match

Reference photos are stored in faces/ directory:
  faces/arun.jpg
  faces/aadi.jpg
  faces/shruthi.jpg

Usage::

    from ari.vision.recognizer import PersonRecognizer
    recognizer = PersonRecognizer()
    recognizer.register("arun")  # captures and saves reference photo

    # Later, during scanning:
    name = recognizer.identify(frame)  # returns "arun", "aadi", or None
"""

from __future__ import annotations

import logging
import os
import subprocess

import cv2
import numpy as np

from ari.config import cfg
from ari.vision.camera import image_to_base64

logger = logging.getLogger(__name__)

FACES_DIR = os.path.expanduser("~/ari-assistant/faces")


class PersonRecognizer:
    """Identify specific people by comparing against stored reference photos."""

    def __init__(self):
        self._claude_cli = os.path.expanduser(cfg["brain"]["claude_cli"])
        self._claude_model = cfg["brain"]["model"]
        self._claude_timeout = cfg["brain"]["timeout"]
        self._known_people = self._load_known_people()

    def _load_known_people(self) -> dict[str, str]:
        """Load reference photos from faces/ directory.

        Returns dict of {name: filepath}.
        """
        people = {}
        if not os.path.exists(FACES_DIR):
            os.makedirs(FACES_DIR, exist_ok=True)
            return people

        for f in os.listdir(FACES_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(f)[0].lower()
                people[name] = os.path.join(FACES_DIR, f)
                logger.info("Loaded reference photo: %s", name)

        return people

    @property
    def known_names(self) -> list[str]:
        """List of registered people."""
        return list(self._known_people.keys())

    def has_reference(self, name: str) -> bool:
        """Check if we have a reference photo for this person."""
        return name.lower() in self._known_people

    def register(self, name: str, frame: np.ndarray) -> str:
        """Save a reference photo for a person.

        Args:
            name: person's name (lowercase)
            frame: RGB image containing the person

        Returns:
            Path to saved reference photo.
        """
        name = name.lower()
        filepath = os.path.join(FACES_DIR, f"{name}.jpg")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr)
        self._known_people[name] = filepath
        logger.info("Registered reference photo for: %s", name)
        return filepath

    def identify(self, frame: np.ndarray, target_name: str | None = None) -> str | None:
        """Identify who is in the frame.

        Args:
            frame: RGB image containing a person
            target_name: if set, only check if this specific person matches

        Returns:
            Name of identified person, or None if no match.
        """
        # Save current frame as temp file
        temp_path = "/tmp/ari_identify.jpg"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, bgr)
        current_b64 = image_to_base64(temp_path)
        if not current_b64:
            return None

        # If looking for a specific person
        if target_name:
            target_name = target_name.lower()
            if target_name not in self._known_people:
                logger.warning("No reference photo for: %s", target_name)
                return None
            if self._compare(current_b64, target_name):
                return target_name
            return None

        # Otherwise, try all known people
        for name in self._known_people:
            if self._compare(current_b64, name):
                return name

        return None

    def _compare(self, current_b64: str, name: str) -> bool:
        """Compare current image against a reference photo using Claude.

        Returns True if Claude thinks it's the same person.
        """
        ref_path = self._known_people[name]
        ref_b64 = image_to_base64(ref_path)
        if not ref_b64:
            return False

        prompt = (
            f"I have two images. The first is a reference photo of a person named '{name}'. "
            f"The second is a live camera image. "
            f"Is the person in the live image the SAME person as in the reference photo? "
            f"Reply with ONLY 'YES' or 'NO' on the first line. Nothing else."
            f"\n\n[Reference photo of {name}: data:image/jpeg;base64,{ref_b64}]"
            f"\n\n[Live camera image: data:image/jpeg;base64,{current_b64}]"
        )

        try:
            result = subprocess.run(
                [self._claude_cli, "-p", "--bare",
                 "--model", self._claude_model, "--tools", "",
                 "--system-prompt",
                 "You are comparing two photos to identify if they show the same person. "
                 "Focus on facial features, hair, build, and other identifying characteristics. "
                 "Reply ONLY 'YES' or 'NO'."],
                input=prompt,
                capture_output=True, text=True,
                timeout=self._claude_timeout, env=os.environ,
            )
            answer = result.stdout.strip().upper()
            logger.info("Compare %s: %s", name, answer)
            return answer.startswith("YES")
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error("Claude compare error: %s", e)
            return False


def extract_person_name(text: str) -> str | None:
    """Extract the person's name from a 'find X' command.

    Examples:
        "find arun" → "arun"
        "find aadi" → "aadi"
        "where is shruthi" → "shruthi"
        "find me" → None (not a specific person)
    """
    text = text.lower().strip()

    # Generic phrases that don't specify a person
    generic = ["find me", "find someone", "look for me", "can you see me"]
    if any(g in text for g in generic):
        return None

    # Extract name after keywords
    for prefix in ["find ", "where is ", "look for ", "where's "]:
        if prefix in text:
            name = text.split(prefix, 1)[1].strip().rstrip("?.!")
            # Remove common filler words
            name = name.replace("please", "").replace("can you", "").strip()
            if name and len(name) > 1:
                return name

    return None
