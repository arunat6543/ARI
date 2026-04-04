"""Text-to-speech output for Ari using Piper + aplay.

Supports both blocking ``speak()`` and low-latency ``speak_streaming()``
which pipelines sentence generation and playback.

Usage::

    speaker = Speaker(mic=mic_instance)
    speaker.speak("Hello, world!")
    speaker.speak_streaming("This is sentence one. And here is two.")
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from ari.config import cfg

if TYPE_CHECKING:
    from ari.audio.microphone import Microphone

log = logging.getLogger(__name__)

# Regex to split text into sentences at . ! ? followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Speaker:
    """Generate speech with Piper and play it through ALSA."""

    def __init__(self, mic: Microphone | None = None) -> None:
        tts_cfg = cfg["tts"]
        audio_cfg = cfg["audio"]

        self._piper_bin: str = str(tts_cfg.get("piper_bin", "piper"))
        self._model: str = str(tts_cfg.get("model", ""))
        self._speaker_id: str = str(tts_cfg.get("speaker", "0"))
        self._length_scale: str = str(tts_cfg.get("length_scale", "1.0"))
        self._sentence_pause: float = float(tts_cfg.get("sentence_pause", 0.0))
        self._aplay_device: str = str(audio_cfg.get("aplay_device", "default"))

        self._mic = mic

        log.info(
            "Speaker init: piper=%s model=%s speaker=%s aplay=%s",
            self._piper_bin, Path(self._model).name,
            self._speaker_id, self._aplay_device,
        )

    # -- Mic muting helpers ---------------------------------------------------

    def _mute_mic(self) -> None:
        """Mute the microphone if one is attached."""
        if self._mic is not None:
            self._mic.mute()

    def _unmute_mic(self) -> None:
        """Unmute the microphone if one is attached."""
        if self._mic is not None:
            self._mic.unmute()

    # -- Piper WAV generation -------------------------------------------------

    def _generate_wav(self, text: str, wav_path: str) -> bool:
        """Run Piper to synthesize *text* into a WAV file at *wav_path*.

        Returns True on success, False on failure.
        """
        cmd = [
            self._piper_bin,
            "--model", self._model,
            "--speaker", self._speaker_id,
            "--length-scale", self._length_scale,
            "--output_file", wav_path,
        ]
        log.debug("Piper cmd: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                log.error("Piper failed (rc=%d): %s", proc.returncode, proc.stderr.strip())
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error("Piper timed out for text: %s", text[:60])
            return False
        except FileNotFoundError:
            log.error("Piper binary not found: %s", self._piper_bin)
            return False

    # -- aplay ----------------------------------------------------------------

    def _play_wav(self, wav_path: str) -> None:
        """Play a WAV file through ALSA using aplay."""
        cmd = ["aplay", "-D", self._aplay_device, wav_path]
        log.debug("aplay cmd: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            log.error("aplay timed out for %s", wav_path)
        except FileNotFoundError:
            log.error("aplay not found")

    # -- Public API -----------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesize and play *text* as speech (blocking).

        Mutes the microphone during playback to prevent echo / feedback.
        """
        text = text.strip()
        if not text:
            return

        log.info("Speaking: %s", text[:80])
        self._mute_mic()

        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="ari_tts_")
        os.close(wav_fd)

        try:
            if self._generate_wav(text, wav_path):
                self._play_wav(wav_path)
        finally:
            self._unmute_mic()
            _remove_file(wav_path)

    def speak_streaming(self, text: str) -> None:
        """Synthesize and play *text* with pipelined sentence generation.

        Splits *text* into sentences, generates WAV files in parallel using
        a thread pool, and plays them sequentially as each becomes ready.
        The first sentence starts playing as soon as its WAV is generated,
        while later sentences are still being synthesized.
        """
        text = text.strip()
        if not text:
            return

        sentences = _split_sentences(text)
        if not sentences:
            return

        log.info("Streaming %d sentence(s): %s", len(sentences), text[:80])
        self._mute_mic()

        wav_paths: list[str] = []
        futures: list[Future[bool]] = []

        try:
            # Submit all sentence generations in parallel.
            with ThreadPoolExecutor(max_workers=min(len(sentences), 4)) as pool:
                for sentence in sentences:
                    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="ari_tts_")
                    os.close(wav_fd)
                    wav_paths.append(wav_path)
                    futures.append(pool.submit(self._generate_wav, sentence, wav_path))

                # Play each sentence as soon as its WAV is ready, in order.
                for i, future in enumerate(futures):
                    success = future.result()  # blocks until this sentence is done
                    if success:
                        self._play_wav(wav_paths[i])
                    else:
                        log.warning("Skipping sentence %d (generation failed)", i)
        finally:
            self._unmute_mic()
            for path in wav_paths:
                _remove_file(path)


# -- Module helpers -----------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, filtering out empty strings."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _remove_file(path: str) -> None:
    """Silently remove a file if it exists."""
    try:
        os.unlink(path)
    except OSError:
        pass
