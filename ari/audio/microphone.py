"""Microphone input for Ari.

Provides speech recording with silence detection, mute control (for
TTS playback), and resampling to 16 kHz for Whisper.

Usage::

    mic = Microphone()
    audio = mic.record_speech()
    if audio is not None:
        # audio is int16 numpy array at 16 kHz, ready for whisper
        ...
"""

from __future__ import annotations

import logging
import time
from threading import Lock

import numpy as np
import sounddevice as sd

from ari.config import cfg

log = logging.getLogger(__name__)


class Microphone:
    """Capture audio from the system microphone with speech detection."""

    def __init__(self) -> None:
        audio_cfg = cfg["audio"]

        self._device: int = audio_cfg.get("mic_device", 0)
        self._channels: int = self._detect_channels()
        self._sample_rate: int = int(audio_cfg.get("mic_sample_rate", 44100))
        self._whisper_rate: int = int(audio_cfg.get("whisper_sample_rate", 16000))
        self._silence_threshold: int = int(audio_cfg.get("silence_threshold", 400))
        self._conversation_silence: float = float(audio_cfg.get("conversation_silence", 1.5))
        self._max_duration: float = float(audio_cfg.get("max_recording_duration", 30))
        self._min_duration: float = float(audio_cfg.get("min_recording_duration", 0.3))
        self._no_speech_timeout: float = float(audio_cfg.get("no_speech_timeout", 8))

        self._muted = False
        self._mute_lock = Lock()

        log.info(
            "Microphone init: device=%s channels=%d rate=%d threshold=%d",
            self._device, self._channels, self._sample_rate, self._silence_threshold,
        )

    # -- Mute control (used during TTS playback) ------------------------------

    @property
    def is_muted(self) -> bool:
        """True when the mic is muted (e.g. during TTS playback)."""
        with self._mute_lock:
            return self._muted

    def mute(self) -> None:
        """Mute the microphone so record calls return silence / None."""
        with self._mute_lock:
            self._muted = True
        log.debug("Microphone muted")

    def unmute(self) -> None:
        """Unmute the microphone."""
        with self._mute_lock:
            self._muted = False
        log.debug("Microphone unmuted")

    # -- Channel auto-detection -----------------------------------------------

    def _detect_channels(self) -> int:
        """Query sounddevice for the number of input channels on our device."""
        try:
            info = sd.query_devices(self._device, kind="input")
            channels = int(info["max_input_channels"])  # type: ignore[index]
            log.info("Detected %d input channel(s) on device %s", channels, self._device)
            return max(channels, 1)
        except Exception:
            log.warning("Could not query device %s, defaulting to 1 channel", self._device)
            return 1

    # -- Low-level recording --------------------------------------------------

    def record_chunk(self, duration: float) -> np.ndarray:
        """Record a fixed-duration chunk of audio.

        Uses ``sd.InputStream`` with a callback that collects audio in the
        background, so it does not block the audio thread.

        Parameters
        ----------
        duration:
            Length of the recording in seconds.

        Returns
        -------
        numpy.ndarray
            Mono int16 audio at the native mic sample rate (channel 0).
        """
        if self.is_muted:
            # Return silence while muted.
            num_samples = int(self._sample_rate * duration)
            return np.zeros(num_samples, dtype=np.int16)

        frames: list[np.ndarray] = []

        def _callback(indata: np.ndarray, frame_count: int, time_info, status) -> None:
            if status:
                log.debug("InputStream status: %s", status)
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self._sample_rate,
            device=self._device,
            channels=self._channels,
            dtype="int16",
            callback=_callback,
        ):
            sd.sleep(int(duration * 1000))

        if not frames:
            return np.zeros(0, dtype=np.int16)

        audio = np.concatenate(frames, axis=0)
        # Take channel 0 only (mono).
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio

    # -- Speech detection -----------------------------------------------------

    @staticmethod
    def has_speech(audio: np.ndarray) -> bool:
        """Return True if *audio* contains speech above the silence threshold.

        Uses the instance threshold when called on an instance, but this is
        also usable as a static helper with a manual threshold check.
        """
        if audio.size == 0:
            return False
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        # We cannot access instance threshold here (static), so callers that
        # need the configured threshold should use _has_speech_threshold.
        return rms > 0

    def _has_speech_threshold(self, audio: np.ndarray) -> bool:
        """Return True if RMS of *audio* exceeds the configured threshold."""
        if audio.size == 0:
            return False
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        return rms > self._silence_threshold

    # -- Resampling -----------------------------------------------------------

    @staticmethod
    def resample(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Resample *audio* from *orig_rate* to *target_rate* via linear interpolation.

        Parameters
        ----------
        audio:
            1-D numpy array (any numeric dtype).
        orig_rate:
            Sample rate of the input audio.
        target_rate:
            Desired output sample rate.

        Returns
        -------
        numpy.ndarray
            Resampled audio with the same dtype as input.
        """
        if orig_rate == target_rate or audio.size == 0:
            return audio
        duration = audio.shape[0] / orig_rate
        target_len = int(duration * target_rate)
        if target_len == 0:
            return np.zeros(0, dtype=audio.dtype)
        # Linear interpolation indices.
        indices = np.linspace(0, audio.shape[0] - 1, target_len)
        resampled = np.interp(indices, np.arange(audio.shape[0]), audio.astype(np.float64))
        return resampled.astype(audio.dtype)

    # -- High-level speech recording ------------------------------------------

    def record_speech(self, stop_flag_fn=None) -> np.ndarray | None:
        """Record audio until speech followed by silence is detected.

        Waits for speech to begin, then records until
        ``conversation_silence`` seconds of quiet, or ``max_recording_duration``
        is reached.

        Parameters
        ----------
        stop_flag_fn:
            Optional callable returning True to abort recording early.

        Returns
        -------
        numpy.ndarray or None
            Resampled 16 kHz int16 mono audio ready for Whisper, or None if
            no speech was detected or recording was aborted.
        """
        if self.is_muted:
            return None

        chunk_sec = 0.3  # size of each recording chunk
        all_audio: list[np.ndarray] = []
        speech_started = False
        silence_start: float | None = None
        recording_start = time.monotonic()

        log.debug("record_speech: waiting for speech...")

        while True:
            # Check abort conditions.
            if stop_flag_fn is not None and stop_flag_fn():
                log.debug("record_speech: aborted by stop_flag_fn")
                return None
            if self.is_muted:
                log.debug("record_speech: mic muted, aborting")
                return None

            elapsed = time.monotonic() - recording_start

            chunk = self.record_chunk(chunk_sec)
            has_voice = self._has_speech_threshold(chunk)

            if not speech_started:
                # Waiting for speech to begin.
                if has_voice:
                    speech_started = True
                    silence_start = None
                    all_audio.append(chunk)
                    log.debug("record_speech: speech detected at %.1fs", elapsed)
                elif elapsed > self._no_speech_timeout:
                    log.debug("record_speech: no speech after %.1fs, giving up", elapsed)
                    return None
            else:
                # Currently recording speech.
                all_audio.append(chunk)

                if has_voice:
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif (time.monotonic() - silence_start) >= self._conversation_silence:
                        log.debug("record_speech: silence detected, stopping")
                        break

                if elapsed >= self._max_duration:
                    log.debug("record_speech: max duration reached")
                    break

        if not all_audio:
            return None

        audio = np.concatenate(all_audio)
        duration = audio.shape[0] / self._sample_rate

        if duration < self._min_duration:
            log.debug("record_speech: too short (%.2fs), discarding", duration)
            return None

        # Resample to 16 kHz for Whisper.
        audio_16k = self.resample(audio, self._sample_rate, self._whisper_rate)
        log.info("record_speech: captured %.2fs of speech (%d samples at 16kHz)", duration, audio_16k.shape[0])
        return audio_16k
