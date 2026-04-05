"""Text-to-speech output for Ari using Piper + aplay.

Optimized for low latency:
  - Streams Piper raw audio directly to aplay (no temp files)
  - Audio starts playing while Piper is still generating
  - speak_streaming() splits into sentences and plays each immediately

Usage::

    speaker = Speaker(mic=mic_instance)
    speaker.speak("Hello!")
    speaker.speak_streaming("Sentence one. And sentence two.")
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import TYPE_CHECKING

from ari.config import cfg

if TYPE_CHECKING:
    from ari.audio.microphone import Microphone

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Speaker:
    """Generate speech with Piper and play through ALSA.

    Uses streaming mode: Piper --output_raw | aplay
    Audio starts playing before Piper finishes generating.
    """

    def __init__(self, mic: Microphone | None = None) -> None:
        tts_cfg = cfg["tts"]
        audio_cfg = cfg["audio"]

        self._piper_bin: str = os.path.expanduser(str(tts_cfg.get("piper_bin", "piper")))
        self._model: str = os.path.expanduser(str(tts_cfg.get("model", "")))
        self._speaker_id: str = str(tts_cfg.get("speaker", "0"))
        self._length_scale: str = str(tts_cfg.get("length_scale", "1.0"))
        self._aplay_device: str = str(audio_cfg.get("aplay_device", "default"))

        # Detect sample rate from model config
        self._sample_rate = self._detect_sample_rate()

        self._mic = mic

        log.info("Speaker init: model=%s speaker=%s rate=%d aplay=%s",
                 os.path.basename(self._model), self._speaker_id,
                 self._sample_rate, self._aplay_device)

    def _detect_sample_rate(self) -> int:
        """Read sample rate from Piper model config JSON."""
        import json
        config_path = self._model + ".json"
        try:
            with open(config_path) as f:
                data = json.load(f)
            return data.get("audio", {}).get("sample_rate", 22050)
        except (FileNotFoundError, json.JSONDecodeError):
            return 22050

    # -- Mic muting -----------------------------------------------------------

    def _mute_mic(self) -> None:
        if self._mic is not None:
            self._mic.mute()

    def _unmute_mic(self) -> None:
        if self._mic is not None:
            self._mic.unmute()

    # -- Streaming speech (low latency) ----------------------------------------

    def _speak_stream(self, text: str) -> None:
        """Stream Piper raw output directly to aplay.

        Audio starts playing as soon as Piper generates the first chunk,
        before the full text is synthesized. This saves 1-2 seconds
        compared to generating a WAV file first.
        """
        try:
            piper = subprocess.Popen(
                [self._piper_bin, "--model", self._model,
                 "--speaker", self._speaker_id,
                 "--length-scale", self._length_scale,
                 "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            aplay = subprocess.Popen(
                ["aplay", "-r", str(self._sample_rate), "-f", "S16_LE",
                 "-t", "raw", "-c", "1", "-D", self._aplay_device],
                stdin=piper.stdout,
                stderr=subprocess.DEVNULL,
            )

            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()
            piper.wait(timeout=30)
            aplay.wait(timeout=30)

        except subprocess.TimeoutExpired:
            log.error("TTS stream timed out")
            try:
                piper.kill()
                aplay.kill()
            except Exception:
                pass
        except Exception as e:
            log.error("TTS stream error: %s", e)

    # -- Public API -----------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesize and play text (blocking). Mutes mic during playback."""
        text = text.strip()
        if not text:
            return

        log.info("Speaking: %s", text[:80])
        self._mute_mic()
        try:
            self._speak_stream(text)
        finally:
            time.sleep(0.2)  # small buffer after speaking
            self._unmute_mic()

    def speak_streaming(self, text: str) -> None:
        """Split text into sentences and speak each one immediately.

        Each sentence is streamed (Piper → aplay) so audio starts fast.
        Sentences are spoken sequentially but each one streams internally.
        """
        text = text.strip()
        if not text:
            return

        sentences = _split_sentences(text)
        if not sentences:
            return

        log.info("Streaming %d sentence(s): %s", len(sentences), text[:80])
        self._mute_mic()
        try:
            for sentence in sentences:
                self._speak_stream(sentence)
        finally:
            time.sleep(0.2)
            self._unmute_mic()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering empty strings."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip()]
