# Ari Robot Assistant

Ari is a voice + vision chess-playing robot running on a Raspberry Pi 5. It listens via a USB microphone, speaks through a speaker, sees through an IMX708 camera on a pan-tilt mount, and will eventually play chess with a robotic arm.

## Directory Structure

```
ari-assistant/
  ari/                     # Main Python package
    audio/                 # Microphone input, speaker output, Whisper transcription
      microphone.py
      speaker.py
      transcriber.py
      voice_id.py          # Speaker identification via resemblyzer
      wakeword.py          # openWakeWord-based wake detection (lightweight)
    brain/                 # LLM backends with pluggable engine support
      base.py              # Abstract Brain base class
      claude_client.py     # Claude CLI backend (vision, complex reasoning)
      gemma_client.py      # Gemma/Ollama backend (local, offline conversation)
      gemini_client.py     # Gemini Live backend (speech-to-speech via WebSocket)
      intent.py            # Keyword-based intent detection
    hardware/              # Hardware abstraction (servos, future arm)
      servo.py             # PanTilt class -- single source of truth for servo control
    ipc/                   # Inter-process communication
      fifo.py              # FifoServer (daemon side) and FifoClient (caller side)
    skills/                # Pluggable skill modules
      base.py              # Abstract Skill base class
    vision/                # Camera capture and person scanning
      camera.py
      scanner.py           # Person scanning (YOLO + Claude Vision)
      detector.py          # YOLO and Haar cascade detectors
      recognizer.py        # Person recognition via Claude Vision
    config.py              # YAML config loader, exposes global `cfg` singleton
  config/
    default.yaml           # All configuration values (the single source of truth)
    local.yaml             # Machine-specific overrides (git-ignored)
  models/                  # Piper TTS model files
  faces/                   # Reference photos for person recognition
  voices/                  # Voiceprint embeddings for speaker ID
  scripts/                 # Start/stop scripts
  camera_daemon.py         # Camera + pan-tilt daemon using ari modules
```

## Brain Engine System

Ari supports three LLM backends, switchable via `config/default.yaml`:

```yaml
brain:
  engine: "gemini"   # "gemma", "claude", or "gemini"
```

| Engine   | How it works                        | Use case                          |
|----------|-------------------------------------|-----------------------------------|
| `gemma`  | Local Ollama (Gemma 3 1B on Pi)     | Offline conversation, no internet |
| `claude` | Claude CLI (Whisper -> text -> TTS) | Vision tasks, complex reasoning   |
| `gemini` | Gemini Live API (speech-to-speech)  | Low-latency voice conversation    |

### Gemma (local)
- Runs via Ollama on localhost:11434
- Uses `urllib` for HTTP API (no extra pip packages)
- Model: `gemma3:1b` (~815MB, fits in 4GB RAM)
- Flow: Whisper STT -> Gemma text -> Piper TTS

### Claude
- Uses Claude CLI (`ccd-cli`) for text generation
- Supports streaming (sentence-by-sentence TTS)
- Used by vision/scanner modules for image analysis
- Flow: Whisper STT -> Claude text -> Piper TTS

### Gemini (speech-to-speech)
- Uses Gemini Live API via WebSocket
- Sends raw mic audio, receives audio response -- no Whisper or Piper needed
- Requires `GEMINI_API_KEY` environment variable
- Free tier: ~1000 requests/day
- Voice set via `brain.gemini.voice` config (e.g., "Orus")
- Flow: Raw audio -> Gemini -> Raw audio playback

### Adding a new engine
1. Create `ari/brain/my_engine.py` implementing the `Brain` base class
2. Add it to the factory in `ari/brain/__init__.py`
3. Add config section under `brain:` in `config/default.yaml`

## How to Run

Start Ari:
```bash
scripts/start_ari.sh
```

Stop Ari:
```bash
scripts/stop_ari.sh
```

Manual start (voice daemon only, without camera):
```bash
cd ~/ari-assistant
PYTHONPATH=. GEMINI_API_KEY="your-key" python -u -m ari.daemon
```

Important: always kill old daemon processes before starting new ones.

## Configuration

All configurable values live in `config/default.yaml`. Never hardcode values in Python code -- read them from `ari.config.cfg` instead.

```python
from ari.config import cfg
sample_rate = cfg["audio"]["mic_sample_rate"]
```

To add a new config value:
1. Add the key with a sensible default to `config/default.yaml` under the appropriate section.
2. Read it via `cfg["section"]["key"]` in your code.
3. If you need a machine-specific override, put it in `config/local.yaml` (same structure, only the keys you want to change).

## Speaker (TTS)

The speaker module (`ari/audio/speaker.py`) uses Piper TTS piped to aplay.

Key optimization: `_speak_multi()` keeps a single aplay process alive across multiple sentences, eliminating the gap between sentences that occurs when spawning separate aplay processes.

- `speak()` -- single sentence, mutes mic during playback
- `speak_streaming()` -- multi-sentence, gapless playback via single aplay
- `_speak_stream()` -- low-level single sentence (also used by brain clients)

## Wake Word Detection

Two modes available, configured automatically:

1. **openWakeWord** (preferred): Uses pre-trained models, <6ms per check, ~50MB RAM
   - Config: `wake.oww_model` and `wake.oww_threshold`
   - Currently uses `hey_jarvis` (custom "ok ari" model needs training)
2. **Whisper** (fallback): Full STT, checks for wake phrases in transcribed text
   - Config: `wake.phrases` list
   - Works with "ok ari", "hey ari", etc.

## How to Add a New Skill

1. Create a new file in `ari/skills/`, e.g. `ari/skills/chess.py`.
2. Import and implement the `Skill` base class.
3. Register the skill in the main assistant loop.

The `context` dict passed to `handle` contains shared resources:
- `speaker` -- `ari.audio.speaker.Speaker`
- `brain` -- current Brain instance (GemmaClient, ClaudeClient, or GeminiClient)
- `camera_client` -- `ari.ipc.fifo.FifoClient`
- `scanner` -- `ari.vision.scanner.PersonScanner`
- `session_id` -- current conversation session ID (or None)

## IPC Protocol

Daemons communicate via named pipes (FIFOs). Each daemon has a command FIFO and a status file:

- Camera daemon: `/tmp/ari_camera_cmd` and `/tmp/ari_camera_status`
- Voice daemon: `/tmp/ari_voice_cmd` and `/tmp/ari_voice_status`

## Daemon Code Rules

- Use `flush=True` on all `print()` statements in daemon code.
- Always write a status reply after handling each command so clients do not hang.
- Handle `KeyboardInterrupt` and clean up (close hardware, remove FIFO).

## Infrastructure on ari (Raspberry Pi 5)

- **Ollama** v0.20.5 installed at `~/bin/ollama`, runs as user systemd service
  - Config: `~/.config/systemd/user/ollama.service`
  - Models stored in `~/.ollama/models/`
  - Libs at `~/lib/ollama/`
  - Manage: `systemctl --user start/stop/status ollama`
- **Gemini API key**: set via `GEMINI_API_KEY` env var (free tier from Google AI Studio)
- **Piper TTS**: binary at `~/ari-assistant/bin/piper`

## Hardware Details

- **Board:** Raspberry Pi 5 (4GB RAM)
- **Camera:** IMX708 (captured via `rpicam-still`)
- **Microphone:** USB mic (1 or 2 channels, auto-detected)
- **Speaker:** ALSA output on `plughw:3,0`
- **Pan-tilt mount:** Two servos on PCA9685 PWM driver over I2C
  - Pan = channel 0, Tilt = channel 1, Frequency: 50 Hz

## Servo Direction Mapping

From the camera's perspective:

| Direction  | Pulse change         |
|------------|----------------------|
| Tilt UP    | LOWER pulse (toward tilt_min) |
| Tilt DOWN  | HIGHER pulse (toward tilt_max) |
| Pan LEFT   | HIGHER pulse (toward pan_max) |
| Pan RIGHT  | LOWER pulse (toward pan_min) |
