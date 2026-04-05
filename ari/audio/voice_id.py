"""Voice identification for Ari — identify who is speaking.

Uses resemblyzer (speaker embedding model) to create voiceprints
and compare against stored reference recordings.

Usage::

    from ari.audio.voice_id import VoiceID
    vid = VoiceID()

    # Register a voice (record 5 seconds)
    vid.register("arun", audio_16k)

    # Later, identify who is speaking
    name, confidence = vid.identify(audio_16k)
    # Returns ("arun", 0.85) or (None, 0.0)
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)

VOICES_DIR = os.path.expanduser("~/ari-assistant/voices")
SIMILARITY_THRESHOLD = 0.75  # minimum cosine similarity to match


class VoiceID:
    """Identify speakers by comparing voice embeddings."""

    def __init__(self):
        from resemblyzer import VoiceEncoder
        self._encoder = VoiceEncoder()
        self._voiceprints: dict[str, np.ndarray] = {}
        os.makedirs(VOICES_DIR, exist_ok=True)
        self._load_voiceprints()
        logger.info("VoiceID loaded: %d voices registered", len(self._voiceprints))

    def _load_voiceprints(self) -> None:
        """Load saved voiceprints from disk."""
        for f in os.listdir(VOICES_DIR):
            if f.endswith(".pkl"):
                name = f[:-4].lower()
                path = os.path.join(VOICES_DIR, f)
                try:
                    with open(path, "rb") as fh:
                        self._voiceprints[name] = pickle.load(fh)
                    logger.info("Loaded voiceprint: %s", name)
                except Exception as e:
                    logger.error("Failed to load voiceprint %s: %s", name, e)

    @property
    def known_voices(self) -> list[str]:
        return list(self._voiceprints.keys())

    def has_voiceprint(self, name: str) -> bool:
        return name.lower() in self._voiceprints

    def register(self, name: str, audio_16k: np.ndarray) -> float:
        """Create and save a voiceprint from audio.

        Args:
            name: person's name (lowercase)
            audio_16k: mono audio at 16kHz, float32 normalized (-1 to 1)
                       or int16

        Returns:
            Embedding norm (for sanity check — should be ~1.0)
        """
        name = name.lower()
        audio_f32 = self._to_float32(audio_16k)

        # Create embedding
        embedding = self._encoder.embed_utterance(audio_f32)

        # Save to disk
        path = os.path.join(VOICES_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(embedding, f)

        self._voiceprints[name] = embedding
        norm = float(np.linalg.norm(embedding))
        logger.info("Registered voiceprint for %s (norm=%.2f)", name, norm)
        return norm

    def identify(self, audio_16k: np.ndarray) -> tuple[str | None, float]:
        """Identify who is speaking.

        Args:
            audio_16k: mono audio at 16kHz

        Returns:
            (name, confidence) or (None, 0.0) if no match
        """
        if not self._voiceprints:
            return None, 0.0

        audio_f32 = self._to_float32(audio_16k)

        try:
            embedding = self._encoder.embed_utterance(audio_f32)
        except Exception as e:
            logger.error("Failed to create embedding: %s", e)
            return None, 0.0

        # Compare against all voiceprints
        best_name = None
        best_score = 0.0

        for name, ref_embedding in self._voiceprints.items():
            # Cosine similarity
            score = float(np.dot(embedding, ref_embedding) /
                          (np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)))
            logger.debug("Voice similarity %s: %.2f", name, score)

            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= SIMILARITY_THRESHOLD:
            logger.info("Voice identified: %s (%.0f%%)", best_name, best_score * 100)
            return best_name, best_score
        else:
            logger.info("Voice not matched (best: %s at %.0f%%)",
                        best_name, best_score * 100)
            return None, best_score

    @staticmethod
    def _to_float32(audio: np.ndarray) -> np.ndarray:
        """Convert audio to float32 normalized."""
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)
