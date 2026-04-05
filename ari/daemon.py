#!/usr/bin/env python3 -u
"""Ari Voice Daemon -- the main orchestrator for the Ari robot assistant.

A thin state machine that coordinates microphone input, speech recognition,
intent detection, Claude conversations, camera control, and text-to-speech.

States::

    SLEEPING       -- passively listening for wake phrases
    AWAKE          -- actively listening, transcribing, and responding
    SHUTTING_DOWN  -- cleanup and exit

Control via FIFO ``/tmp/ari_voice_cmd``::

    wake     -- force wake up
    sleep    -- force sleep
    status   -- write current state to status file
    quit     -- graceful shutdown

Usage::

    from ari.daemon import AriDaemon
    daemon = AriDaemon()
    daemon.run()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from enum import Enum
from pathlib import Path

from ari.config import cfg
from ari.audio.microphone import Microphone
from ari.audio.transcriber import Transcriber
from ari.audio.speaker import Speaker
from ari.audio.voice_id import VoiceID
from ari.brain.claude_client import ClaudeClient
from ari.brain.intent import detect_intent, contains_phrase
from ari.vision.scanner import PersonScanner
from ari.vision.camera import capture_and_resize, image_to_base64
from ari.ipc.fifo import FifoClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class State(Enum):
    """Daemon lifecycle states."""
    SLEEPING = "sleeping"
    AWAKE = "awake"
    SHUTTING_DOWN = "shutting_down"


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class AriDaemon:
    """Main Ari voice daemon -- thin orchestrator around the component modules."""

    def __init__(self) -> None:
        # -- IPC paths --------------------------------------------------------
        ipc_cfg = cfg["ipc"]
        self._voice_fifo: str = ipc_cfg["voice_fifo"]
        self._status_path: str = ipc_cfg["voice_status"]
        self._camera_fifo: str = ipc_cfg["camera_fifo"]

        # -- Temp directory ---------------------------------------------------
        self._temp_dir: str = ipc_cfg.get("temp_dir", "/tmp/ari")
        os.makedirs(self._temp_dir, exist_ok=True)

        # -- Components -------------------------------------------------------
        print("Loading microphone...", flush=True)
        self.mic = Microphone()

        print("Loading transcriber (Whisper)...", flush=True)
        self.transcriber = Transcriber()

        self.speaker = Speaker(mic=self.mic)
        self.claude = ClaudeClient()
        self.camera = FifoClient(self._camera_fifo)
        self.scanner = PersonScanner(self.camera)

        print("Loading voice ID...", flush=True)
        try:
            self.voice_id = VoiceID()
            print(f"  Registered voices: {self.voice_id.known_voices}", flush=True)
        except Exception as exc:
            log.warning("Voice ID failed to load: %s", exc)
            self.voice_id = None

        # -- Wake / sleep config ----------------------------------------------
        wake_cfg = cfg["wake"]
        self._wake_phrases: list[str] = wake_cfg["phrases"]
        self._sleep_phrases: list[str] = wake_cfg["sleep_phrases"]
        self._wake_check_duration: float = float(wake_cfg.get("check_duration", 3))
        self._silence_timeout: float = float(wake_cfg.get("silence_timeout", 60))
        self._startup_msg: str = wake_cfg.get(
            "startup_message",
            "Ari is ready. Say, listen to me Ari, when you want to talk.",
        )
        self._wake_msg: str = wake_cfg.get("wake_message", "I'm listening!")
        self._sleep_msg: str = wake_cfg.get(
            "sleep_message",
            "Okay, going back to sleep. Call me when you need me!",
        )
        self._timeout_msg: str = wake_cfg.get(
            "timeout_message",
            "I'll go back to sleep now. Call me if you need me!",
        )
        self._goodbye_msg: str = wake_cfg.get("goodbye_message", "Goodbye!")

        # -- Runtime state ----------------------------------------------------
        self._state = State.SLEEPING
        self._last_speech_time: float = 0.0
        self._session_id: str | None = None
        self._current_speaker: str | None = None  # identified by voice ID

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, new: State) -> None:
        old = self._state
        self._state = new
        if old != new:
            log.info("State: %s -> %s", old.value, new.value)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Start the daemon -- blocking until shutdown."""
        print("=" * 50, flush=True)
        print("  Ari Voice Daemon", flush=True)
        print('  Wake: "Listen to me, Ari"', flush=True)
        print('  Sleep: "Go back to sleep"', flush=True)
        print("=" * 50, flush=True)
        print(flush=True)
        print(f"Control pipe: {self._voice_fifo}", flush=True)
        print(flush=True)

        # Start FIFO listener in a background thread.
        fifo_thread = threading.Thread(target=self.fifo_listener, daemon=True)
        fifo_thread.start()

        # Announce startup.
        self.speaker.speak(self._startup_msg)

        try:
            while self._state != State.SHUTTING_DOWN:
                if self._state == State.SLEEPING:
                    self.sleep_loop()
                elif self._state == State.AWAKE:
                    self.awake_loop()
        except KeyboardInterrupt:
            print("\nInterrupted by user", flush=True)
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ #
    # Sleep loop
    # ------------------------------------------------------------------ #

    def sleep_loop(self) -> None:
        """Passive mode -- listen for wake phrases only."""
        print("Sleeping -- waiting for wake word...", flush=True)
        self.write_status("sleeping")

        while self._state == State.SLEEPING:
            # Skip recording while mic is muted (e.g. during TTS playback).
            if self.mic.is_muted:
                time.sleep(0.5)
                continue

            # Record a short chunk and check for speech.
            try:
                audio = self.mic.record_chunk(self._wake_check_duration)
            except Exception as exc:
                print(f"  Mic error: {exc}", flush=True)
                time.sleep(1)
                continue

            if not self.mic._has_speech_threshold(audio):
                continue

            # Speech detected -- transcribe and check for wake phrase.
            audio_16k = Microphone.resample(
                audio, self.mic._sample_rate, self.mic._whisper_rate,
            )
            text = self.transcriber.transcribe(audio_16k)
            if text:
                print(f"  [sleep heard]: \"{text}\"", flush=True)

            if text and contains_phrase(text, self._wake_phrases):
                print(f"Wake word detected: \"{text}\"", flush=True)
                self.state = State.AWAKE
                self._last_speech_time = time.time()
                return

    # ------------------------------------------------------------------ #
    # Awake loop
    # ------------------------------------------------------------------ #

    def awake_loop(self) -> None:
        """Active mode -- full conversation with intent dispatch."""
        # Fresh session each time we wake up.
        self._session_id = None
        print("Awake -- listening for conversation...", flush=True)
        self.write_status("awake")
        self.speaker.speak(self._wake_msg)

        while self._state == State.AWAKE:
            # Check silence timeout.
            if time.time() - self._last_speech_time > self._silence_timeout:
                print("Silence timeout -- going back to sleep", flush=True)
                self.speaker.speak(self._timeout_msg)
                self.state = State.SLEEPING
                return

            # Record speech (blocks until utterance or timeout).
            audio = self.mic.record_speech(
                stop_flag_fn=lambda: self._state != State.AWAKE,
            )
            if audio is None:
                continue

            # Identify who is speaking (run on same audio).
            speaker = self._identify_speaker(audio)
            if speaker != self._current_speaker:
                self._current_speaker = speaker
                if speaker:
                    print(f"  Speaker: {speaker}", flush=True)

            # Transcribe.
            text = self.transcriber.transcribe(audio)
            if not text or len(text.strip()) < 2:
                continue

            self._last_speech_time = time.time()
            speaker_label = self._current_speaker or "Someone"
            print(f"  {speaker_label}: {text}", flush=True)

            # Detect intent and dispatch.
            result = detect_intent(text)
            intent = result["intent"]

            if intent == "sleep":
                self._handle_sleep()
                return

            elif intent == "camera_direction":
                self._handle_camera_direction(result["direction"])

            elif intent == "find_person":
                self._handle_find_person(text)

            elif intent == "vision":
                self._handle_vision(text)

            else:
                # Default: conversation
                self._handle_conversation(text)

    # ------------------------------------------------------------------ #
    # Speaker identification
    # ------------------------------------------------------------------ #

    def _identify_speaker(self, audio_16k) -> str | None:
        """Identify who is speaking from audio. Returns name or None."""
        if self.voice_id is None:
            return None

        try:
            import numpy as np

            # Strip silence — only keep chunks above noise threshold
            chunk_size = int(16000 * 0.1)  # 100ms chunks at 16kHz
            speech_chunks = []
            for i in range(0, len(audio_16k) - chunk_size, chunk_size):
                chunk = audio_16k[i:i + chunk_size]
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                if rms > 0.006:  # ~200 in int16 scale / 32768
                    speech_chunks.append(chunk)

            if not speech_chunks or len(speech_chunks) < 3:
                return self._current_speaker  # not enough speech, keep last

            speech_only = np.concatenate(speech_chunks)
            name, confidence = self.voice_id.identify(speech_only)
            return name
        except Exception as exc:
            log.debug("Voice ID error: %s", exc)
            return self._current_speaker

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #

    def _handle_sleep(self) -> None:
        """Transition to sleep on user request."""
        print("Sleep phrase detected", flush=True)
        self.speaker.speak(self._sleep_msg)
        self.state = State.SLEEPING

    def _handle_camera_direction(self, direction: str) -> None:
        """Send a direction command to the camera daemon."""
        print(f"Moving camera: {direction}", flush=True)
        cmd_map = {
            "left": "pan_left 300",
            "right": "pan_right 300",
            "up": "tilt_up 200",
            "down": "tilt_down 200",
            "home": "home",
            "center": "home",
        }
        cmd = cmd_map.get(direction, "home")
        try:
            self.camera.send(cmd)
        except RuntimeError as exc:
            print(f"  Camera command failed: {exc}", flush=True)
        self.speaker.speak(f"Looking {direction}.")

    def _handle_find_person(self, text: str) -> None:
        """Scan for a person, then describe what was found via Claude."""
        print("Starting person scan...", flush=True)
        self.speaker.speak("Let me look around for you.")

        result = self.scanner.find_person()
        if result:
            pan, tilt, desc = result
            # Capture a final image at the person's position.
            image_path = capture_and_resize()
            if image_path:
                print("Describing what I found...", flush=True)
                reply, sid = self.claude.ask(
                    f"I just scanned around with my camera and found someone. "
                    f"The user asked me to find them. Describe who you see in "
                    f"the image and greet them naturally. The user said: {text}",
                    image_path=image_path,
                )
                self._session_id = sid
                print(f"  Ari: {reply}", flush=True)
                self.speaker.speak_streaming(reply)
            else:
                self.speaker.speak("I found someone but couldn't capture an image.")
        else:
            self.speaker.speak("I couldn't find anyone. Try waving at me!")

    def _handle_vision(self, text: str) -> None:
        """Capture an image and send it to Claude along with the user's text."""
        print("Capturing for vision...", flush=True)
        speaker_prefix = f"[{self._current_speaker} says]: " if self._current_speaker else ""
        image_path = capture_and_resize()
        if image_path:
            reply, sid = self.claude.ask(
                f"{speaker_prefix}{text}",
                image_path=image_path, session_id=self._session_id,
            )
        else:
            reply, sid = self.claude.ask(
                f"{speaker_prefix}(I tried to take a photo but the camera failed.) {text}",
                session_id=self._session_id,
            )
        self._session_id = sid
        print(f"  Ari: {reply}", flush=True)
        self.speaker.speak_streaming(reply)

    def _handle_conversation(self, text: str) -> None:
        """General conversation -- text only, with session continuity."""
        print("Thinking...", flush=True)
        # Include speaker name so Claude knows who it's talking to
        if self._current_speaker:
            prompt = f"[{self._current_speaker} says]: {text}"
        else:
            prompt = text
        reply, sid = self.claude.ask(prompt, session_id=self._session_id)
        self._session_id = sid
        if sid:
            print(f"  [session: {sid[:8]}...]", flush=True)
        print(f"  Ari: {reply}", flush=True)
        self.speaker.speak_streaming(reply)

    # ------------------------------------------------------------------ #
    # FIFO listener
    # ------------------------------------------------------------------ #

    def fifo_listener(self) -> None:
        """Background thread: read commands from the voice FIFO.

        Supports: wake, sleep, status, quit.
        """
        fifo = Path(self._voice_fifo)
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(fifo)

        while self._state != State.SHUTTING_DOWN:
            try:
                with open(fifo, "r") as fh:
                    for line in fh:
                        cmd = line.strip().lower()
                        if cmd == "wake":
                            self.state = State.AWAKE
                            self._last_speech_time = time.time()
                            self.write_status("awake (forced)")
                            print("Forced wake via FIFO", flush=True)
                        elif cmd == "sleep":
                            self.state = State.SLEEPING
                            self.write_status("sleeping (forced)")
                            print("Forced sleep via FIFO", flush=True)
                        elif cmd == "status":
                            self.write_status(self._state.value)
                        elif cmd == "quit":
                            self.state = State.SHUTTING_DOWN
                            self.write_status("shutting_down")
                            print("Quit command received via FIFO", flush=True)
                            return
            except Exception:
                if self._state == State.SHUTTING_DOWN:
                    return
                time.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def write_status(self, msg: str) -> None:
        """Write a status string to the voice status file."""
        try:
            with open(self._status_path, "w") as fh:
                fh.write(msg + "\n")
        except OSError as exc:
            log.warning("Could not write status: %s", exc)

    def shutdown(self) -> None:
        """Clean up resources and say goodbye."""
        self.state = State.SHUTTING_DOWN
        self.write_status("stopped")
        self.speaker.speak(self._goodbye_msg)

        # Remove the FIFO.
        fifo = Path(self._voice_fifo)
        if fifo.exists():
            try:
                fifo.unlink()
            except OSError:
                pass

        print("Ari daemon stopped.", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python -m ari.daemon``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    daemon = AriDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
