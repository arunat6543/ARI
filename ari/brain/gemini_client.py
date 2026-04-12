"""Gemini Live audio brain for Ari -- speech-to-speech via WebSocket.

Sends raw mic audio to Gemini, receives audio back, plays it directly.
No Whisper, no Piper needed in the conversation loop.

Usage::

    client = GeminiClient()
    client.run_live_turn(audio_16k, "plughw:3,0")
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Generator, Optional

from ari.brain.base import Brain
from ari.config import cfg

log = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class GeminiClient(Brain):
    """Gemini Live API client -- speech-to-speech."""

    def __init__(self) -> None:
        brain_cfg = cfg["brain"]
        gemini_cfg = brain_cfg.get("gemini", {})

        self._model: str = gemini_cfg.get("model", "gemini-3.1-flash-live-preview")
        self._api_key: str = os.environ.get("GEMINI_API_KEY", gemini_cfg.get("api_key", ""))
        self._system_prompt: str = brain_cfg["system_prompt"]
        self._voice: str = gemini_cfg.get("voice", "Orus")

        if not self._api_key:
            log.warning("GEMINI_API_KEY not set -- Gemini brain will fail")

    # -- Speech-to-speech -----------------------------------------------------

    def run_live_turn(self, audio_16k, aplay_device: str) -> None:
        """Send recorded audio to Gemini Live, play audio response.

        Uses subprocess to run the async code in a clean process,
        avoiding asyncio event loop conflicts with threads.
        """
        import tempfile
        import numpy as np

        # Save audio to temp file
        audio_path = "/tmp/ari_gemini_input.raw"
        audio_16k.tofile(audio_path)

        # Run the async Gemini call in a subprocess to avoid asyncio conflicts
        script = f'''
import asyncio
import subprocess
import numpy as np
import sys

async def live_turn():
    from google import genai
    from google.genai import types

    client = genai.Client(api_key="{self._api_key}")
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="{self._voice}",
                )
            ),
        ),
        system_instruction=types.Content(
            parts=[types.Part(text="""{self._system_prompt.replace('"', '\\"')}""")]
        ),
    )

    audio_16k = np.fromfile("{audio_path}", dtype=np.int16)
    audio_bytes = audio_16k.tobytes()

    async with client.aio.live.connect(
        model="{self._model}", config=config
    ) as session:
        chunk_size = 32000
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )
        await session.send_realtime_input(audio_stream_end=True)

        aplay = subprocess.Popen(
            ["aplay", "-r", "24000", "-f", "S16_LE", "-t", "raw", "-c", "1", "-D", "{aplay_device}"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        async for response in session.receive():
            sc = response.server_content
            if sc and sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        aplay.stdin.write(part.inline_data.data)
            if sc and sc.turn_complete:
                break

        aplay.stdin.close()
        aplay.wait(timeout=30)

asyncio.run(live_turn())
'''

        try:
            result = subprocess.run(
                ["/home/arun/ari-assistant/bin/python", "-c", script],
                timeout=30,
                capture_output=True,
                text=True,
                env={**os.environ, "GEMINI_API_KEY": self._api_key},
            )
            if result.returncode != 0:
                error = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
                print(f"Gemini error: {error}", flush=True)
        except subprocess.TimeoutExpired:
            print("Gemini timed out", flush=True)
        except Exception as e:
            print(f"Gemini error: {e}", flush=True)

    # -- Brain text interface (fallback) --------------------------------------

    def ask(
        self,
        text: str,
        image_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Text-based ask using Gemini generate."""
        try:
            from google import genai
            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
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
        """Text ask + TTS speak (fallback)."""
        full_reply = []
        for sentence in self.ask_streaming(text, image_path, session_id):
            full_reply.append(sentence)
            speaker._speak_stream(sentence)
        reply = " ".join(full_reply)
        return reply if reply else "Sorry, I didn't get a response."
