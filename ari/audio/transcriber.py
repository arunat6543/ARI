"""Speech-to-text transcription for Ari using faster-whisper.

Usage::

    transcriber = Transcriber()
    text = transcriber.transcribe(audio_16k)
"""

from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from ari.config import cfg

log = logging.getLogger(__name__)


class Transcriber:
    """Wraps faster-whisper for single-call transcription."""

    def __init__(self) -> None:
        stt_cfg = cfg["stt"]

        model_size: str = stt_cfg.get("model", "base")
        device: str = stt_cfg.get("device", "cpu")
        compute_type: str = stt_cfg.get("compute_type", "int8")

        self._beam_size: int = int(stt_cfg.get("beam_size", 1))
        self._language: str = stt_cfg.get("language", "en")
        self._vad_filter: bool = bool(stt_cfg.get("vad_filter", True))

        log.info(
            "Loading Whisper model: size=%s device=%s compute=%s",
            model_size, device, compute_type,
        )
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        log.info("Whisper model loaded")

    def transcribe(self, audio_16k: np.ndarray) -> str:
        """Transcribe audio to text.

        Parameters
        ----------
        audio_16k:
            Numpy array at 16 kHz sample rate. Can be int16 or float32.
            If int16, it is converted to float32 in [-1, 1] automatically.

        Returns
        -------
        str
            Transcribed text (stripped, lowercased). Empty string if nothing
            was recognized.
        """
        if audio_16k.size == 0:
            return ""

        # faster-whisper expects float32 in [-1.0, 1.0].
        if audio_16k.dtype == np.int16:
            audio = audio_16k.astype(np.float32) / 32768.0
        elif audio_16k.dtype == np.float32:
            audio = audio_16k
        else:
            audio = audio_16k.astype(np.float32)

        segments, info = self._model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=self._language,
            vad_filter=self._vad_filter,
        )

        text = " ".join(seg.text for seg in segments).strip()
        log.info("Transcribed (%s, %.2fs): %s", info.language, info.duration, text)
        return text
