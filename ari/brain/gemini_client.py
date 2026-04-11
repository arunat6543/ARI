"""Gemini Live audio brain for Ari -- speech-to-speech via WebSocket.

Bypasses Whisper and Piper entirely. Sends raw mic audio to Gemini,
receives audio back, plays it directly.

Also implements the Brain text interface as a fallback for non-audio use.

Usage::

    client = GeminiClient()

    # Speech-to-speech (used by daemon in gemini mode)
    client.run_live(mic, speaker_device)

    # Text fallback (Brain interface)
    reply, sid = client.ask("Hello")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import subprocess
import threading
from typing import Generator, Optional

from ari.brain.base import Brain
from ari.config import cfg

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class GeminiClient(Brain):
    """Gemini Live API client -- supports both text and speech-to-speech."""

    def __init__(self) -> None:
        brain_cfg = cfg["brain"]
        gemini_cfg = brain_cfg.get("gemini", {})

        self._model: str = gemini_cfg.get("model", "gemini-2.5-flash-preview-native-audio-dialog")
        self._api_key: str = os.environ.get("GEMINI_API_KEY", gemini_cfg.get("api_key", ""))
        self._system_prompt: str = brain_cfg["system_prompt"]
        self._voice: str = gemini_cfg.get("voice", "Orus")
        self._sample_rate_in: int = 16000   # mic input rate
        self._sample_rate_out: int = 24000  # Gemini output rate

        if not self._api_key:
            log.warning("GEMINI_API_KEY not set -- Gemini brain will fail")

    # -- Speech-to-speech (main mode) -----------------------------------------

    def run_live_turn(self, audio_16k, aplay_device: str) -> None:
        """Send recorded audio to Gemini Live, play audio response.

        This is a single-turn exchange:
        1. Send the recorded audio chunk
        2. Receive and play audio response
        3. Return when done

        Args:
            audio_16k: numpy int16 array at 16kHz (from Microphone)
            aplay_device: ALSA device string for playback
        """
        asyncio.run(self._live_turn_async(audio_16k, aplay_device))

    async def _live_turn_async(self, audio_16k, aplay_device: str) -> None:
        """Async implementation of a single live turn."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice,
                    )
                ),
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self._system_prompt)]
            ),
        )

        try:
            async with client.aio.live.connect(
                model=self._model, config=config
            ) as session:
                # Convert int16 numpy to bytes
                audio_bytes = audio_16k.tobytes()

                # Send audio in chunks (large blobs can fail)
                chunk_size = 32000  # 1 second of 16kHz int16
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )

                # Signal end of audio input
                await session.send_realtime_input(audio_stream_end=True)

                # Stream audio response directly to aplay
                aplay = subprocess.Popen(
                    ["aplay", "-r", "24000", "-f", "S16_LE",
                     "-t", "raw", "-c", "1", "-D", aplay_device],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )

                async for response in session.receive():
                    sc = response.server_content
                    if sc and sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                try:
                                    aplay.stdin.write(part.inline_data.data)
                                except BrokenPipeError:
                                    break
                    if sc and sc.turn_complete:
                        break

                aplay.stdin.close()
                aplay.wait(timeout=30)

        except Exception as e:
            log.error("Gemini Live error: %s", e)

    # -- Brain text interface (fallback) --------------------------------------

    def ask(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Text-based ask using Gemini generate (non-live)."""
        try:
            from google import genai
            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model=self._model.replace("native-audio-dialog", "preview"),
                contents=text,
                config={"system_instruction": self._system_prompt},
            )
            reply = response.text.strip() if response.text else "(empty response)"
            return (reply, None)
        except Exception as e:
            log.error("Gemini ask error: %s", e)
            return ("Sorry, something went wrong.", None)

    def ask_streaming(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Text streaming -- yields sentences."""
        reply, _ = self.ask(text, image_path, session_id)
        parts = _SENTENCE_RE.split(reply)
        for p in parts:
            p = p.strip()
            if p:
                yield p

    def ask_and_speak(
        self,
        text: str,
        speaker,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Text ask + TTS speak (fallback when not using live audio)."""
        full_reply = []
        for sentence in self.ask_streaming(text, image_path, session_id):
            full_reply.append(sentence)
            speaker._speak_stream(sentence)
        reply = " ".join(full_reply)
        return reply if reply else "Sorry, I didn't get a response."
