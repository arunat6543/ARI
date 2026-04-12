"""Lightweight wake word detection for Ari using openWakeWord.

Replaces Whisper-based wake word detection in sleep mode.
Uses ~50MB RAM and <6ms per prediction vs Whisper's 500MB and 2-3 seconds.

Usage::

    from ari.audio.wakeword import WakeWordDetector
    detector = WakeWordDetector()

    # Feed 80ms audio chunks (1280 samples at 16kHz)
    if detector.detect(audio_chunk_16k):
        print("Wake word detected!")
"""

from __future__ import annotations

import logging
import os
import glob

import numpy as np
from openwakeword.model import Model

from ari.config import cfg

log = logging.getLogger(__name__)


class WakeWordDetector:
    """Fast wake word detection using openWakeWord."""

    def __init__(self) -> None:
        wake_cfg = cfg.get("wake", {})
        self._model_name: str = wake_cfg.get("oww_model", "hey_jarvis")
        self._threshold: float = float(wake_cfg.get("oww_threshold", 0.5))

        log.info("Loading wake word model: %s (threshold=%.2f)",
                 self._model_name, self._threshold)

        # Find the model file
        model_path = self._find_model(self._model_name)
        if model_path:
            log.info("Loading custom model: %s", model_path)
            self._model = Model(wakeword_model_paths=[model_path])
        else:
            log.info("Using default models")
            self._model = Model()

        log.info("Wake word models loaded: %s", list(self._model.models.keys()))

    def _find_model(self, name: str) -> str | None:
        """Find a model file by name in known locations."""
        # Check common locations
        search_paths = [
            os.path.expanduser(f"~/ari-assistant/models/{name}.onnx"),
            os.path.expanduser(f"~/ari-assistant/models/{name}.tflite"),
        ]
        # Also search openwakeword resources
        import openwakeword
        oww_dir = os.path.dirname(openwakeword.__file__)
        resources = os.path.join(oww_dir, "resources", "models")
        search_paths.append(os.path.join(resources, f"{name}.onnx"))
        search_paths.append(os.path.join(resources, f"{name}.tflite"))

        for p in search_paths:
            if os.path.exists(p):
                return p

        return None

    def detect(self, audio_16k: np.ndarray) -> bool:
        """Check if wake word is present in audio chunk."""
        frame_size = 1280
        for i in range(0, len(audio_16k) - frame_size + 1, frame_size):
            frame = audio_16k[i:i + frame_size]
            prediction = self._model.predict(frame)

            score = prediction.get(self._model_name, 0)
            if score > self._threshold:
                log.info("Wake word '%s' detected (score=%.2f)",
                         self._model_name, score)
                self._model.reset()
                return True

        return False

    def reset(self) -> None:
        """Reset internal state after detection."""
        self._model.reset()
