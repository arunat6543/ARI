#!/usr/bin/env python3
"""
Ari Voice + Vision Assistant — powered by Claude on Raspberry Pi 5.

Listens via microphone, responds via speaker, can see via camera.
"""

import os
import sys
import io
import wave
import base64
import tempfile
import subprocess
import threading
import queue
import time
import signal

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import anthropic

# ── Configuration ──────────────────────────────────────────────────────────────

WHISPER_MODEL = "tiny"           # tiny/base/small — tiny is fastest on Pi
MIC_SAMPLE_RATE = 44100          # Mic native rate (USB PnP Sound Device)
WHISPER_SAMPLE_RATE = 16000      # Whisper expects 16kHz
CHANNELS = 1
SILENCE_THRESHOLD = 200          # RMS threshold to detect speech
SILENCE_DURATION = 1.5           # Seconds of silence to stop recording
MIN_RECORDING_DURATION = 0.5     # Minimum seconds to count as speech
MAX_RECORDING_DURATION = 30      # Maximum seconds per utterance

PIPER_BIN = os.path.expanduser("~/ari-assistant/bin/piper")
PIPER_MODEL = os.path.expanduser("~/ari-assistant/models/en_US-amy-medium.onnx")

# Audio devices
MIC_DEVICE = 0          # USB PnP Sound Device (hw:2,0)
APLAY_DEVICE = "plughw:3,0"  # USB PnP Audio Device (handles mono→stereo)

# Vision keywords — if the user says any of these, capture camera
VISION_KEYWORDS = [
    "look", "see", "camera", "show", "what is this", "what's this",
    "what do you see", "picture", "photo", "image", "watch", "observe",
    "read this", "scan", "visual", "in front", "looking at", "describe what",
    "what am i holding", "what color", "identify"
]

SYSTEM_PROMPT = """You are Ari, a helpful voice assistant running on a Raspberry Pi 5.
You speak conversationally and keep responses concise (1-3 sentences) since your answers
will be spoken aloud via text-to-speech. Be natural and friendly.

When an image is included, describe what you see naturally as part of the conversation.
If the user asks you to look at something, comment on what you observe.

You can help with general questions, control the Pi, and discuss what you see through the camera."""


# ── Globals ────────────────────────────────────────────────────────────────────

conversation_history = []
whisper_model = None
client = None


def init():
    """Initialize models and API client."""
    global whisper_model, client

    print("🔧 Loading Whisper model (this may take a moment)...")
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("✅ Whisper ready")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Set ANTHROPIC_API_KEY environment variable first!")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)
    print("✅ Claude API ready")

    # Quick TTS test
    print("🔊 Testing speaker...")
    speak("Ari is ready.")
    print("✅ All systems go!\n")


def get_rms(audio_chunk):
    """Calculate RMS volume of audio chunk."""
    return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))


def resample(audio, orig_rate, target_rate):
    """Simple linear resampling from orig_rate to target_rate."""
    if orig_rate == target_rate:
        return audio
    ratio = target_rate / orig_rate
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples)
    return np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(audio.dtype)


def record_speech():
    """Record speech from microphone, stop after silence."""
    print("🎤 Listening... (speak now)")

    audio_buffer = []
    silence_start = None
    speech_detected = False
    recording_start = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal silence_start, speech_detected
        if status:
            pass  # Ignore minor audio glitches

        chunk = indata[:, 0].copy()
        audio_buffer.append(chunk)
        rms = get_rms(chunk)

        if rms > SILENCE_THRESHOLD:
            speech_detected = True
            silence_start = None
        elif speech_detected and silence_start is None:
            silence_start = time.time()

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", blocksize=int(MIC_SAMPLE_RATE * 0.1),
                        device=MIC_DEVICE, callback=callback):
        while True:
            time.sleep(0.05)
            elapsed = time.time() - recording_start

            if speech_detected and silence_start and \
               (time.time() - silence_start) > SILENCE_DURATION:
                break
            if elapsed > MAX_RECORDING_DURATION:
                print("⏱️  Max recording time reached")
                break
            # Timeout if no speech detected after 10 seconds
            if not speech_detected and elapsed > 10:
                return None

    if not speech_detected or not audio_buffer:
        return None

    audio = np.concatenate(audio_buffer)
    duration = len(audio) / MIC_SAMPLE_RATE
    if duration < MIN_RECORDING_DURATION:
        return None

    # Resample from mic rate to Whisper's expected 16kHz
    audio = resample(audio, MIC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

    print(f"📝 Recorded {duration:.1f}s of audio")
    return audio


def transcribe(audio):
    """Transcribe audio using faster-whisper."""
    # Convert int16 to float32 normalized
    audio_float = audio.astype(np.float32) / 32768.0

    segments, info = whisper_model.transcribe(audio_float, beam_size=1,
                                               language="en",
                                               vad_filter=True)
    text = " ".join(seg.text for seg in segments).strip()
    return text


def capture_camera():
    """Capture a frame from the Pi camera and return as base64 JPEG."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name

        subprocess.run(
            ["rpicam-still", "-o", tmp_path, "--timeout", "1000",
             "--width", "1280", "--height", "960", "--nopreview"],
            capture_output=True, timeout=10
        )

        with open(tmp_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        os.unlink(tmp_path)
        print("📷 Camera image captured")
        return image_data
    except Exception as e:
        print(f"⚠️  Camera error: {e}")
        return None


def should_use_vision(text):
    """Check if the user's message suggests using the camera."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in VISION_KEYWORDS)


def ask_claude(user_text, image_b64=None):
    """Send message to Claude and get response."""
    content = []

    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64
            }
        })

    content.append({"type": "text", "text": user_text})

    conversation_history.append({"role": "user", "content": content})

    # Keep conversation history manageable (last 20 exchanges)
    trimmed = conversation_history[-40:]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=trimmed,
    )

    reply = response.content[0].text
    conversation_history.append({"role": "assistant", "content": reply})

    return reply


def speak(text):
    """Convert text to speech using piper and play via speaker."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name

        # Generate WAV file with piper
        proc = subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "-f", tmp_wav],
            input=text.encode("utf-8"),
            capture_output=True, timeout=30
        )

        # Play WAV via aplay with plughw (handles channel conversion)
        subprocess.run(
            ["aplay", "-D", APLAY_DEVICE, tmp_wav],
            capture_output=True, timeout=30
        )

        os.unlink(tmp_wav)
    except Exception as e:
        print(f"⚠️  TTS error: {e}")


def main():
    print("=" * 50)
    print("  🤖 Ari — Voice + Vision Assistant")
    print("  Powered by Claude on Raspberry Pi 5")
    print("=" * 50)
    print()

    init()

    print("Say something to start a conversation.")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            # 1. Listen
            audio = record_speech()
            if audio is None:
                continue

            # 2. Transcribe
            text = transcribe(audio)
            if not text or len(text.strip()) < 2:
                print("(couldn't understand, try again)")
                continue
            print(f"🗣️  You: {text}")

            # 3. Check if vision is needed
            image_b64 = None
            if should_use_vision(text):
                image_b64 = capture_camera()

            # 4. Ask Claude
            print("🤔 Thinking...")
            reply = ask_claude(text, image_b64)
            print(f"🤖 Ari: {reply}")

            # 5. Speak response
            speak(reply)
            print()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
