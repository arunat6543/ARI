#!/usr/bin/env python3 -u
"""
Ari Voice Daemon — wake word activated, runs independently.

States:
  SLEEPING  — passively listening for "listen to me ari"
  AWAKE     — actively listening, transcribing, responding via Claude

Transitions:
  SLEEPING → AWAKE:  wake phrase detected ("listen to me ari")
  AWAKE → SLEEPING:  sleep phrase ("go back to sleep") or silence timeout

Control via FIFO /tmp/ari_voice_cmd:
  wake      — force wake up
  sleep     — force sleep
  status    — print current state
  quit      — exit daemon
"""

import os
import sys
import subprocess
import time
import threading
import re
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Add ari-assistant to path for imports
sys.path.insert(0, os.path.expanduser("~/ari-assistant"))
import camera_control

# ── Config ─────────────────────────────────────────────────────────────────────

MIC_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000
MIC_DEVICE = 0       # first available input device
MIC_CHANNELS = None  # auto-detect at startup
APLAY_DEVICE = "plughw:3,0"

PIPER_BIN = os.path.expanduser("~/ari-assistant/bin/piper")
PIPER_MODEL = os.path.expanduser("~/ari-assistant/models/en_US-libritts_r-medium.onnx")
PIPER_SPEAKER = "10"
PIPER_LENGTH_SCALE = "1.4"
CLAUDE_CLI = os.path.expanduser("~/.claude/remote/ccd-cli/2.1.87")

FIFO_PATH = "/tmp/ari_voice_cmd"
STATUS_PATH = "/tmp/ari_voice_status"
CAMERA_FIFO = "/tmp/ari_camera_cmd"

# Wake/sleep phrases
WAKE_PHRASES = ["listen to me ari", "listen to me, ari", "hey ari", "ari listen",
                 "listen ari", "are you listening ari", "look at me ari",
                 "look at me, ari", "can you hear me ari", "ready ari",
                 "are you ready ari", "ari are you", "ari can you",
                 "lister", "listen"]
SLEEP_PHRASES = ["go back to sleep", "stop listening", "bye ari", "goodbye ari"]

# Timing
SILENCE_THRESHOLD = 400
WAKE_CHECK_DURATION = 3        # seconds of audio to check for wake word
ACTIVE_SILENCE_TIMEOUT = 60    # seconds of silence before auto-sleep in active mode
CONVERSATION_SILENCE = 1.5     # seconds of silence to end an utterance

SYSTEM_PROMPT = """You are Ari, a friendly voice assistant running on a Raspberry Pi 5.
You speak conversationally and keep responses concise (1-2 sentences) since your answers
will be spoken aloud via text-to-speech. Be natural, warm, and friendly.
When talking to Aadi (a young boy), be playful and encouraging.
Do NOT use emojis or special characters — your output will be spoken aloud."""

# Vision keywords — triggers camera capture + Claude Vision
VISION_KEYWORDS = [
    "look", "see", "camera", "what is this", "what do you see",
    "picture", "photo", "watch", "in front", "looking at",
    "what am i holding", "what color", "identify", "show me"
]

# Find keywords — triggers person scanning
FIND_KEYWORDS = ["find me", "where am i", "find aadi", "find him", "find her",
                  "look for me", "can you see me", "look at me"]

# Camera direction keywords
CAMERA_DIRECTION_KEYWORDS = {
    "look left": "left", "look right": "right",
    "look up": "up", "look down": "down",
    "look straight": "home", "look center": "home",
    "turn left": "left", "turn right": "right",
}


# ── State ──────────────────────────────────────────────────────────────────────

class AriState:
    SLEEPING = "sleeping"
    AWAKE = "awake"
    SHUTTING_DOWN = "shutting_down"


state = AriState.SLEEPING
whisper_model = None
last_speech_time = 0
is_speaking = False  # mute mic while Ari is talking
session_id = None    # Claude CLI session ID for conversation continuity


# ── Audio Helpers ──────────────────────────────────────────────────────────────

def resample(audio, orig_rate, target_rate):
    if orig_rate == target_rate:
        return audio
    n = int(len(audio) * target_rate / orig_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, n),
        np.arange(len(audio)),
        audio.astype(np.float64)
    ).astype(np.int16)


def record_chunk(duration):
    """Record a fixed-duration chunk of audio."""
    samples = int(duration * MIC_SAMPLE_RATE)
    audio = sd.rec(samples, samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS,
                   dtype="int16", device=MIC_DEVICE)
    sd.wait()
    return audio[:, 0]


def record_speech():
    """Record until speech + silence detected (for active conversation)."""
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

    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS, dtype="int16",
                        blocksize=int(MIC_SAMPLE_RATE * 0.1), device=MIC_DEVICE,
                        callback=callback):
        while state == AriState.AWAKE:
            time.sleep(0.05)
            elapsed = time.time() - start
            if speech_detected and silence_start and \
               (time.time() - silence_start) > CONVERSATION_SILENCE:
                break
            if elapsed > 30:
                break
            if not speech_detected and elapsed > 8:
                return None

    if not speech_detected or not audio_buffer:
        return None

    audio = np.concatenate(audio_buffer)
    if len(audio) / MIC_SAMPLE_RATE < 0.3:
        return None

    return resample(audio, MIC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)


def transcribe(audio):
    """Transcribe audio using faster-whisper."""
    audio_f32 = audio.astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(audio_f32, beam_size=1,
                                            language="en", vad_filter=True)
    return " ".join(s.text for s in segments).strip()


def has_speech(audio):
    """Check if audio chunk contains speech (above silence threshold)."""
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    return rms > SILENCE_THRESHOLD


# ── Actions ────────────────────────────────────────────────────────────────────

def speak(text):
    """Convert text to speech and play. Mutes mic during playback."""
    global is_speaking
    try:
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--speaker", PIPER_SPEAKER,
             "--length-scale", PIPER_LENGTH_SCALE, "-f", "/tmp/ari_speak.wav"],
            input=text.encode(), capture_output=True, timeout=30
        )
        is_speaking = True
        subprocess.run(
            ["aplay", "-D", APLAY_DEVICE, "/tmp/ari_speak.wav"],
            capture_output=True, timeout=30
        )
        time.sleep(0.3)  # small buffer after speaking
        is_speaking = False
    except Exception as e:
        is_speaking = False
        print(f"  TTS error: {e}", flush=True)


def speak_streaming(text):
    """Pre-generate TTS for all sentences in parallel, then play sequentially."""
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    if len(sentences) <= 1:
        speak(text)
        return

    # Generate all WAV files in parallel using threads
    wav_files = []
    threads = []

    def gen_wav(sentence, idx):
        wav_path = f"/tmp/ari_s{idx}.wav"
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--speaker", PIPER_SPEAKER,
             "--length-scale", PIPER_LENGTH_SCALE, "-f", wav_path],
            input=sentence.encode(), capture_output=True, timeout=30
        )
        wav_files.append((idx, wav_path))

    for i, sentence in enumerate(sentences):
        t = threading.Thread(target=gen_wav, args=(sentence, i))
        threads.append(t)
        t.start()

    # Wait for first sentence to be ready and play it immediately
    threads[0].join()

    # Play in order, waiting for each to be generated
    for i in range(len(sentences)):
        if i > 0:
            threads[i].join()
        wav_path = f"/tmp/ari_s{i}.wav"
        subprocess.run(["aplay", "-D", APLAY_DEVICE, wav_path],
                       capture_output=True, timeout=30)


def ask_claude(text):
    """Ask Claude via CLI using Haiku for speed."""
    try:
        env = os.environ.copy()
        result = subprocess.run(
            [CLAUDE_CLI, "-p", "--bare", "--model", "haiku", "--tools", "",
             "--system-prompt", SYSTEM_PROMPT],
            input=text, capture_output=True, text=True, timeout=30, env=env
        )
        return result.stdout.strip() or "Sorry, I didn't catch that."
    except Exception as e:
        return f"Sorry, something went wrong."


def ask_and_speak(text, image_path=None):
    """Ask Claude (with optional image) and speak the response. Maintains session."""
    global session_id

    # Build the prompt — attach image if provided
    prompt = text
    if image_path:
        b64 = camera_control.image_to_base64(image_path)
        if b64:
            prompt = f"[Image from my camera as base64 JPEG: data:image/jpeg;base64,{b64}]\n\n{text}"

    # Build CLI command with session continuity
    cmd = [CLAUDE_CLI, "-p", "--bare", "--model", "haiku", "--tools", "",
           "--system-prompt", SYSTEM_PROMPT, "--output-format", "json"]
    if session_id:
        cmd.extend(["--resume", session_id])

    # Start Claude
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=env
    )
    proc.stdin.write(prompt)
    proc.stdin.close()

    # Read output
    raw_output = proc.stdout.read().strip()
    proc.wait(timeout=30)

    # Try to parse JSON to get session_id and result
    reply = raw_output
    try:
        import json
        data = json.loads(raw_output)
        reply = data.get("result", raw_output)
        new_session = data.get("session_id")
        if new_session:
            session_id = new_session
            print(f"  [session: {session_id[:8]}...]", flush=True)
    except (json.JSONDecodeError, Exception):
        # Not JSON — just use raw text
        pass

    if not reply:
        reply = "Sorry, I didn't catch that."

    print(f"🤖 Ari: {reply}", flush=True)

    # Split into sentences and generate TTS in parallel
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reply) if s.strip()]
    if not sentences:
        sentences = [reply]

    # Generate all WAVs in parallel
    wav_threads = []
    for i, sentence in enumerate(sentences):
        def gen(s, idx):
            subprocess.run(
                [PIPER_BIN, "--model", PIPER_MODEL, "--speaker", PIPER_SPEAKER,
                 "--length-scale", PIPER_LENGTH_SCALE, "-f", f"/tmp/ari_s{idx}.wav"],
                input=s.encode(), capture_output=True, timeout=30
            )
        t = threading.Thread(target=gen, args=(sentence, i))
        wav_threads.append(t)
        t.start()

    # Play each sentence as soon as its WAV is ready
    for i, t in enumerate(wav_threads):
        t.join()
        subprocess.run(["aplay", "-D", APLAY_DEVICE, f"/tmp/ari_s{i}.wav"],
                       capture_output=True, timeout=30)

    return reply


def should_use_vision(text):
    t = text.lower()
    return any(kw in t for kw in VISION_KEYWORDS)


def should_find_person(text):
    t = text.lower()
    return any(kw in t for kw in FIND_KEYWORDS)


def get_camera_direction(text):
    """Check if user wants to move camera in a direction."""
    t = text.lower()
    for phrase, direction in CAMERA_DIRECTION_KEYWORDS.items():
        if phrase in t:
            return direction
    return None


def contains_phrase(text, phrases):
    """Check if text contains any of the given phrases."""
    t = text.lower().strip()
    return any(p in t for p in phrases)


def write_status(msg):
    with open(STATUS_PATH, "w") as f:
        f.write(msg + "\n")


# ── Main Loops ─────────────────────────────────────────────────────────────────

def sleep_loop():
    """Passive mode — listen for wake word only."""
    global state, last_speech_time
    print("💤 Sleeping — waiting for wake word...", flush=True)
    write_status("sleeping")

    while state == AriState.SLEEPING:
        # Skip if Ari is currently speaking (avoid picking up own voice)
        if is_speaking:
            time.sleep(0.5)
            continue

        # Record a short chunk using callback (works in background)
        audio_buffer = []

        def callback(indata, frames, time_info, status):
            if not is_speaking:
                audio_buffer.append(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=MIC_CHANNELS,
                                dtype="int16", blocksize=int(MIC_SAMPLE_RATE * 0.1),
                                device=MIC_DEVICE, callback=callback):
                time.sleep(WAKE_CHECK_DURATION)
        except Exception as e:
            print(f"  Mic error: {e}", flush=True)
            time.sleep(1)
            continue

        if not audio_buffer:
            continue

        audio = np.concatenate(audio_buffer)

        # Only transcribe if there's speech above threshold
        if not has_speech(audio):
            continue

        # Transcribe and check for wake phrase
        audio_16k = resample(audio, MIC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        text = transcribe(audio_16k)
        if text:
            print(f"  [sleep heard]: \"{text}\"", flush=True)

        if text and contains_phrase(text, WAKE_PHRASES):
            print(f"🔔 Wake word detected: \"{text}\"", flush=True)
            state = AriState.AWAKE
            last_speech_time = time.time()
            return


def awake_loop():
    """Active mode — full conversation."""
    global state, last_speech_time, session_id
    session_id = None  # fresh conversation each wake-up
    print("👂 Awake — listening for conversation...", flush=True)
    write_status("awake")
    speak("I'm listening!")

    while state == AriState.AWAKE:
        # Check for silence timeout
        if time.time() - last_speech_time > ACTIVE_SILENCE_TIMEOUT:
            print("😴 Silence timeout — going back to sleep")
            speak("I'll go back to sleep now. Call me if you need me!")
            state = AriState.SLEEPING
            return

        # Record speech
        audio = record_speech()
        if audio is None:
            continue

        # Transcribe
        text = transcribe(audio)
        if not text or len(text.strip()) < 2:
            continue

        last_speech_time = time.time()
        print(f"🗣️  User: {text}")

        # Check for sleep phrase
        if contains_phrase(text, SLEEP_PHRASES):
            print("😴 Sleep phrase detected", flush=True)
            speak("Okay, going back to sleep. Call me when you need me!")
            state = AriState.SLEEPING
            return

        # Check for camera direction command
        direction = get_camera_direction(text)
        if direction:
            print(f"📷 Moving camera: {direction}", flush=True)
            camera_control.look_direction(direction)
            speak(f"Looking {direction}.")
            continue

        # Check for "find me" request — scan, then describe with image
        if should_find_person(text):
            print("🔍 Starting person scan...", flush=True)
            speak("Let me look around for you.")
            result = camera_control.find_person()
            if result:
                pan, tilt, desc = result
                # Capture final image and send to Claude with context
                image_path = camera_control.capture()
                print("🤔 Describing what I found...", flush=True)
                ask_and_speak(
                    f"I just scanned around with my camera and found someone. "
                    f"The user asked me to find them. Describe who you see in the image "
                    f"and greet them naturally. The user said: {text}",
                    image_path=image_path
                )
            else:
                speak("I couldn't find anyone. Try waving at me!")
            continue

        # Check for vision request — capture image and send to Claude
        image_path = None
        if should_use_vision(text):
            print("📷 Capturing for vision...", flush=True)
            image_path = camera_control.capture()

        # All requests go through ask_and_speak — with image if captured
        print("🤔 Thinking...", flush=True)
        ask_and_speak(text, image_path=image_path)


def fifo_listener():
    """Listen for control commands via FIFO pipe."""
    global state

    if os.path.exists(FIFO_PATH):
        os.unlink(FIFO_PATH)
    os.mkfifo(FIFO_PATH)

    while state != AriState.SHUTTING_DOWN:
        try:
            with open(FIFO_PATH, "r") as fifo:
                for line in fifo:
                    cmd = line.strip().lower()
                    if cmd == "wake":
                        state = AriState.AWAKE
                        write_status("awake (forced)")
                        print("🔔 Forced wake via command")
                    elif cmd == "sleep":
                        state = AriState.SLEEPING
                        write_status("sleeping (forced)")
                        print("😴 Forced sleep via command")
                    elif cmd == "status":
                        write_status(state)
                    elif cmd == "quit":
                        state = AriState.SHUTTING_DOWN
                        write_status("shutting_down")
                        print("👋 Quit command received")
                        return
        except Exception:
            if state == AriState.SHUTTING_DOWN:
                return
            time.sleep(0.1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global whisper_model, state

    print("=" * 50, flush=True)
    print("  🤖 Ari Voice Daemon", flush=True)
    print("  Wake: \"Listen to me, Ari\"", flush=True)
    print("  Sleep: \"Go back to sleep\"", flush=True)
    print("=" * 50, flush=True)
    print(flush=True)

    # Auto-detect mic channels
    global MIC_CHANNELS
    try:
        dev_info = sd.query_devices(MIC_DEVICE)
        MIC_CHANNELS = dev_info['max_input_channels']
        print(f"🎤 Mic: {dev_info['name']} ({MIC_CHANNELS}ch)", flush=True)
    except Exception:
        MIC_CHANNELS = 1
        print("🎤 Mic: using default (1ch)", flush=True)

    print("🔧 Loading Whisper model...", flush=True)
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("✅ Whisper ready", flush=True)
    print(f"📡 Control pipe: {FIFO_PATH}", flush=True)
    print(flush=True)

    # Start FIFO listener in background thread
    fifo_thread = threading.Thread(target=fifo_listener, daemon=True)
    fifo_thread.start()

    # Announce startup
    speak("Ari is ready. Say, listen to me Ari, when you want to talk.")

    try:
        while state != AriState.SHUTTING_DOWN:
            if state == AriState.SLEEPING:
                sleep_loop()
            elif state == AriState.AWAKE:
                awake_loop()
    except KeyboardInterrupt:
        print("\n👋 Shutting down")
    finally:
        state = AriState.SHUTTING_DOWN
        write_status("stopped")
        speak("Goodbye!")
        if os.path.exists(FIFO_PATH):
            os.unlink(FIFO_PATH)


if __name__ == "__main__":
    main()
