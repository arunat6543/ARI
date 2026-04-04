#!/usr/bin/env python3
"""
Ari Continuous Listening Loop — listen, think, speak, repeat.
Uses Claude Code CLI with Max subscription — no separate API key needed.
"""

import os
import sys
import base64
import tempfile
import subprocess
import time
import json

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ── Config ─────────────────────────────────────────────────────────────────────

MIC_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000
MIC_DEVICE = 0
APLAY_DEVICE = "plughw:3,0"
SILENCE_THRESHOLD = 200
SILENCE_DURATION = 1.5
MAX_RECORD_SECS = 30

PIPER_BIN = os.path.expanduser("~/ari-assistant/bin/piper")
PIPER_MODEL = os.path.expanduser("~/ari-assistant/models/en_US-amy-medium.onnx")
CLAUDE_CLI = os.path.expanduser("~/.claude/remote/ccd-cli/2.1.87")

VISION_KEYWORDS = [
    "look", "see", "camera", "show", "what is this", "what's this",
    "what do you see", "picture", "photo", "image", "watch", "observe",
    "read this", "scan", "what am i holding", "what color", "identify",
    "looking at", "in front", "describe what"
]

SYSTEM_PROMPT = """You are Ari, a helpful voice assistant running on a Raspberry Pi 5.
You speak conversationally and keep responses concise (1-3 sentences) since your answers
will be spoken aloud via text-to-speech. Be natural and friendly.
When an image is included, describe what you see naturally as part of the conversation.
You can help with general questions and discuss what you see through the camera.
Do NOT use emojis or special characters — your output will be spoken aloud."""


def resample(audio, orig_rate, target_rate):
    if orig_rate == target_rate:
        return audio
    n = int(len(audio) * target_rate / orig_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, n),
        np.arange(len(audio)),
        audio.astype(np.float64)
    ).astype(np.int16)


def record_speech():
    """Record until speech + silence detected."""
    audio_buffer = []
    silence_start = None
    speech_detected = False
    start = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal silence_start, speech_detected
        chunk = indata[:, 0].copy()
        audio_buffer.append(chunk)
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        if rms > SILENCE_THRESHOLD:
            speech_detected = True
            silence_start = None
        elif speech_detected and silence_start is None:
            silence_start = time.time()

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=int(MIC_SAMPLE_RATE * 0.1), device=MIC_DEVICE,
                        callback=callback):
        while True:
            time.sleep(0.05)
            elapsed = time.time() - start
            if speech_detected and silence_start and (time.time() - silence_start) > SILENCE_DURATION:
                break
            if elapsed > MAX_RECORD_SECS:
                break
            if not speech_detected and elapsed > 15:
                return None

    if not speech_detected or not audio_buffer:
        return None

    audio = np.concatenate(audio_buffer)
    if len(audio) / MIC_SAMPLE_RATE < 0.3:
        return None

    return resample(audio, MIC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)


def transcribe(model, audio):
    audio_f32 = audio.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_f32, beam_size=1, language="en", vad_filter=True)
    return " ".join(s.text for s in segments).strip()


def capture_camera():
    """Capture a frame from the Pi camera and return as temp file path."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name
        subprocess.run(
            ["rpicam-still", "-o", tmp, "--timeout", "1000",
             "--width", "1280", "--height", "960", "--nopreview"],
            capture_output=True, timeout=10
        )
        print("📷 Camera captured")
        return tmp
    except Exception as e:
        print(f"⚠️  Camera error: {e}")
        return None


def should_use_vision(text):
    t = text.lower()
    return any(kw in t for kw in VISION_KEYWORDS)


def ask_claude(user_text, image_path=None, session_id=None):
    """Use Claude Code CLI with Max subscription."""
    # Build the prompt — if there's an image, encode it inline
    prompt = user_text
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()
        # Pass image as a data URL in the prompt for vision
        prompt = f"[Image attached as base64 JPEG: data:image/jpeg;base64,{b64}]\n\n{user_text}"
        os.unlink(image_path)

    cmd = [
        CLAUDE_CLI, "-p",
        "--bare",
        "--model", "sonnet",
        "--tools", "",
        "--system-prompt", SYSTEM_PROMPT,
    ]

    # Continue conversation if we have a session
    if session_id:
        cmd.extend(["--resume", session_id])

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True, text=True, timeout=60
        )
        reply = result.stdout.strip()
        if not reply:
            reply = "Sorry, I didn't get a response. Could you try again?"
        return reply
    except subprocess.TimeoutExpired:
        return "Sorry, that took too long. Could you try again?"
    except Exception as e:
        return f"Sorry, something went wrong. {e}"


def speak(text):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "-f", tmp],
            input=text.encode(), capture_output=True, timeout=30
        )
        subprocess.run(["aplay", "-D", APLAY_DEVICE, tmp],
                       capture_output=True, timeout=30)
        os.unlink(tmp)
    except Exception as e:
        print(f"⚠️  TTS error: {e}")


def main():
    print("=" * 50)
    print("  🤖 Ari — Continuous Voice + Vision Assistant")
    print("  Powered by Claude (Max) on Raspberry Pi 5")
    print("=" * 50)
    print()

    print("🔧 Loading Whisper model...")
    whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("✅ Whisper ready")

    # Verify Claude CLI works
    print("🔧 Testing Claude CLI...")
    test = subprocess.run(
        [CLAUDE_CLI, "-p", "--bare", "--model", "haiku", "--tools", "",
         "--system-prompt", "Reply with only: OK"],
        input="test", capture_output=True, text=True, timeout=30
    )
    if test.returncode != 0:
        print(f"❌ Claude CLI error: {test.stderr}")
        print("Make sure you're logged in: run 'claude /login'")
        sys.exit(1)
    print("✅ Claude CLI ready (using Max subscription)")

    print("🔊 Saying hello...")
    speak("Hello Arun! I'm Ari, your voice assistant. I'm listening.")
    print("✅ Ready! I'm continuously listening now.\n")

    try:
        while True:
            print("🎤 Listening...")
            audio = record_speech()
            if audio is None:
                continue

            text = transcribe(whisper, audio)
            if not text or len(text.strip()) < 2:
                continue

            print(f"🗣️  You: {text}")

            # Check for exit commands
            if text.lower().strip() in ("goodbye", "bye", "stop", "exit", "quit"):
                speak("Goodbye Arun! Talk to you later.")
                print("👋 Goodbye!")
                break

            # Vision check
            image_path = None
            if should_use_vision(text):
                image_path = capture_camera()

            # Ask Claude via CLI (uses Max subscription)
            print("🤔 Thinking...")
            reply = ask_claude(text, image_path)
            print(f"🤖 Ari: {reply}")

            # Speak
            speak(reply)
            print()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        speak("Goodbye!")


if __name__ == "__main__":
    main()
