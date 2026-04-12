#!/usr/bin/env python3 -u
"""Ari Voice Daemon -- the main orchestrator for the Ari robot assistant.

States::

    SLEEPING       -- passively listening for wake phrases
    AWAKE          -- actively listening, transcribing, and responding
    SHUTTING_DOWN  -- cleanup and exit

Usage::

    from ari.daemon import AriDaemon
    daemon = AriDaemon()
    daemon.run()
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from enum import Enum
from pathlib import Path

from ari.config import cfg
from ari.audio.microphone import Microphone
from ari.audio.transcriber import Transcriber
from ari.brain import create_brain
from ari.brain.intent import detect_intent, contains_phrase
from ari.vision.scanner import PersonScanner
from ari.vision.camera import capture_and_resize, image_to_base64
from ari.ipc.fifo import FifoClient

log = logging.getLogger(__name__)


class State(Enum):
    SLEEPING = "sleeping"
    AWAKE = "awake"
    SHUTTING_DOWN = "shutting_down"


class AriDaemon:
    """Main Ari voice daemon."""

    def __init__(self) -> None:
        ipc_cfg = cfg["ipc"]
        self._voice_fifo: str = ipc_cfg["voice_fifo"]
        self._status_path: str = ipc_cfg["voice_status"]
        self._camera_fifo: str = ipc_cfg["camera_fifo"]
        self._temp_dir: str = ipc_cfg.get("temp_dir", "/tmp/ari")
        os.makedirs(self._temp_dir, exist_ok=True)

        # Engine mode
        self._engine: str = cfg["brain"].get("engine", "claude")
        self._is_gemini: bool = self._engine == "gemini"

        # Microphone (always needed)
        print("Loading microphone...", flush=True)
        self.mic = Microphone()

        # Brain
        self.brain = create_brain()

        # Whisper (needed for wake word in all modes)
        print("Loading transcriber (Whisper)...", flush=True)
        self.transcriber = Transcriber()

        # Speaker + VoiceID (only for text pipeline modes)
        if self._is_gemini:
            print("Gemini Live mode", flush=True)
            self.speaker = None
            self.voice_id = None
        else:
            from ari.audio.speaker import Speaker
            from ari.audio.voice_id import VoiceID
            self.speaker = Speaker(mic=self.mic)
            print("Loading voice ID...", flush=True)
            try:
                self.voice_id = VoiceID()
                print(f"  Registered voices: {self.voice_id.known_voices}", flush=True)
            except Exception as exc:
                log.warning("Voice ID failed to load: %s", exc)
                self.voice_id = None

        # Camera/scanner
        self.camera = FifoClient(self._camera_fifo)
        self.scanner = PersonScanner(self.camera)

        # Wake / sleep config
        wake_cfg = cfg["wake"]
        self._wake_phrases: list[str] = wake_cfg["phrases"]
        self._sleep_phrases: list[str] = wake_cfg["sleep_phrases"]
        self._wake_check_duration: float = float(wake_cfg.get("check_duration", 3))
        self._silence_timeout: float = float(wake_cfg.get("silence_timeout", 15))
        self._startup_msg: str = wake_cfg.get("startup_message", "Ari is ready.")
        self._wake_msg: str = wake_cfg.get("wake_message", "I'm listening!")
        self._sleep_msg: str = wake_cfg.get("sleep_message", "Going back to sleep.")
        self._timeout_msg: str = wake_cfg.get("timeout_message", "Going back to sleep.")
        self._goodbye_msg: str = wake_cfg.get("goodbye_message", "Goodbye!")

        # Runtime state
        self._state = State.SLEEPING
        self._last_speech_time: float = 0.0
        self._session_id: str | None = None
        self._current_speaker: str | None = None
        self._shutting_down = False

        # Handle Ctrl+C cleanly
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\nShutting down...", flush=True)
        self._shutting_down = True
        self._state = State.SHUTTING_DOWN
        os._exit(0)

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, new: State) -> None:
        old = self._state
        self._state = new
        if old != new:
            log.info("State: %s -> %s", old.value, new.value)

    # -- Main entry point -----------------------------------------------------

    def run(self) -> None:
        print("=" * 50, flush=True)
        print("  Ari Voice Daemon", flush=True)
        print(f"  Engine: {self._engine}", flush=True)
        print(f"  Wake: say 'hello'", flush=True)
        print(f"  Sleep: {self._silence_timeout}s silence timeout", flush=True)
        print("=" * 50, flush=True)

        if not self._is_gemini:
            self.speaker.speak(self._startup_msg)

        while self._state != State.SHUTTING_DOWN:
            if self._state == State.SLEEPING:
                self._sleep_loop()
            elif self._state == State.AWAKE:
                if self._is_gemini:
                    self._awake_gemini()
                else:
                    self._awake_text()

    # -- Sleep loop (Whisper-based, all modes) --------------------------------

    def _sleep_loop(self) -> None:
        print("Sleeping -- say 'hello' to wake up...", flush=True)
        self.write_status("sleeping")

        while self._state == State.SLEEPING:
            if self.mic.is_muted:
                time.sleep(0.5)
                continue

            try:
                audio = self.mic.record_chunk(self._wake_check_duration)
            except Exception as exc:
                print(f"  Mic error: {exc}", flush=True)
                time.sleep(1)
                continue

            if not self.mic._has_speech_threshold(audio):
                continue

            audio_16k = Microphone.resample(
                audio, self.mic._sample_rate, self.mic._whisper_rate,
            )
            text = self.transcriber.transcribe(audio_16k)
            if text:
                print(f"  [heard]: \"{text}\"", flush=True)

            if text and contains_phrase(text, self._wake_phrases):
                print(f"Wake word detected: \"{text}\"", flush=True)
                self.state = State.AWAKE
                self._last_speech_time = time.time()
                return

    # -- Awake: Gemini mode ---------------------------------------------------

    def _awake_gemini(self) -> None:
        aplay_device = cfg["audio"]["aplay_device"]
        self._last_speech_time = time.time()
        print("Awake -- listening (Gemini Live)...", flush=True)
        self.write_status("awake")

        while self._state == State.AWAKE:
            if time.time() - self._last_speech_time > self._silence_timeout:
                print("Silence timeout -- going back to sleep", flush=True)
                self.state = State.SLEEPING
                return

            audio = self.mic.record_speech(
                stop_flag_fn=lambda: self._state != State.AWAKE,
            )
            if audio is None:
                continue

            self._last_speech_time = time.time()
            print("Sending to Gemini...", flush=True)
            self.mic.mute()
            try:
                self.brain.run_live_turn(audio, aplay_device)
                print("  Response played", flush=True)
            except Exception as e:
                print(f"  Gemini error: {e}", flush=True)
            finally:
                time.sleep(0.2)
                self.mic.unmute()

    # -- Awake: Text pipeline mode --------------------------------------------

    def _awake_text(self) -> None:
        self._session_id = None
        self._last_speech_time = time.time()
        print("Awake -- listening for conversation...", flush=True)
        self.write_status("awake")
        self.speaker.speak(self._wake_msg)

        while self._state == State.AWAKE:
            if time.time() - self._last_speech_time > self._silence_timeout:
                print("Silence timeout -- going back to sleep", flush=True)
                self.speaker.speak(self._timeout_msg)
                self.state = State.SLEEPING
                return

            audio = self.mic.record_speech(
                stop_flag_fn=lambda: self._state != State.AWAKE,
            )
            if audio is None:
                continue

            # Identify speaker
            if self.voice_id:
                speaker = self._identify_speaker(audio)
                if speaker != self._current_speaker:
                    self._current_speaker = speaker
                    if speaker:
                        print(f"  Speaker: {speaker}", flush=True)

            # Transcribe
            text = self.transcriber.transcribe(audio)
            if not text or len(text.strip()) < 2:
                continue

            self._last_speech_time = time.time()
            speaker_label = self._current_speaker or "Someone"
            print(f"  {speaker_label}: {text}", flush=True)

            # Intent dispatch
            result = detect_intent(text)
            intent = result["intent"]

            if intent == "sleep":
                self.speaker.speak(self._sleep_msg)
                self.state = State.SLEEPING
                return
            elif intent == "camera_direction":
                self._handle_camera_direction(result["direction"])
            elif intent == "find_person":
                self._handle_find_person(text)
            elif intent == "vision":
                self._handle_vision(text)
            else:
                self._handle_conversation(text)

    # -- Handlers -------------------------------------------------------------

    def _identify_speaker(self, audio_16k):
        if self.voice_id is None:
            return None
        try:
            import numpy as np
            chunk_size = int(16000 * 0.1)
            speech_chunks = []
            for i in range(0, len(audio_16k) - chunk_size, chunk_size):
                chunk = audio_16k[i:i + chunk_size]
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                if rms > 0.006:
                    speech_chunks.append(chunk)
            if not speech_chunks or len(speech_chunks) < 3:
                return self._current_speaker
            speech_only = np.concatenate(speech_chunks)
            name, confidence = self.voice_id.identify(speech_only)
            return name
        except Exception:
            return self._current_speaker

    def _handle_camera_direction(self, direction):
        print(f"Moving camera: {direction}", flush=True)
        cmd_map = {"left": "pan_left 300", "right": "pan_right 300",
                    "up": "tilt_up 200", "down": "tilt_down 200",
                    "home": "home", "center": "home"}
        try:
            self.camera.send(cmd_map.get(direction, "home"))
        except RuntimeError as exc:
            print(f"  Camera command failed: {exc}", flush=True)
        self.speaker.speak(f"Looking {direction}.")

    def _handle_find_person(self, text):
        print("Starting person scan...", flush=True)
        self.speaker.speak("Let me look around for you.")
        result = self.scanner.find_person()
        if result:
            pan, tilt, desc = result
            image_path = capture_and_resize()
            if image_path:
                reply, sid = self.brain.ask(
                    f"I found someone. Describe who you see and greet them. User said: {text}",
                    image_path=image_path)
                self._session_id = sid
                print(f"  Ari: {reply}", flush=True)
                self.speaker.speak_streaming(reply)
            else:
                self.speaker.speak("I found someone but couldn't capture an image.")
        else:
            self.speaker.speak("I couldn't find anyone.")

    def _handle_vision(self, text):
        print("Capturing for vision...", flush=True)
        prefix = f"[{self._current_speaker} says]: " if self._current_speaker else ""
        image_path = capture_and_resize()
        if image_path:
            reply, sid = self.brain.ask(f"{prefix}{text}", image_path=image_path,
                                        session_id=self._session_id)
        else:
            reply, sid = self.brain.ask(f"{prefix}(camera failed) {text}",
                                        session_id=self._session_id)
        self._session_id = sid
        print(f"  Ari: {reply}", flush=True)
        self.speaker.speak_streaming(reply)

    def _handle_conversation(self, text):
        print("Thinking...", flush=True)
        prompt = f"[{self._current_speaker} says]: {text}" if self._current_speaker else text
        self.mic.mute()
        try:
            reply = self.brain.ask_and_speak(prompt, self.speaker,
                                             session_id=self._session_id)
            print(f"  Ari: {reply}", flush=True)
        finally:
            time.sleep(0.2)
            self.mic.unmute()

    # -- Helpers --------------------------------------------------------------

    def write_status(self, msg: str) -> None:
        try:
            with open(self._status_path, "w") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    daemon = AriDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
