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
        self._model = Model()
        log.info("Wake word models loaded: %s", list(self._model.models.keys()))

    def detect(self, audio_16k: np.ndarray) -> bool:
        """Check if wake word is present in audio chunk.

        Args:
            audio_16k: int16 numpy array at 16kHz. Should be ~80ms
                       (1280 samples) for best performance, but larger
                       chunks work too -- they're processed in 80ms frames.

        Returns:
            True if wake word detected above threshold.
        """
        # Process in 80ms frames (1280 samples at 16kHz)
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
