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
  models/                  # Piper TTS model files, wake word models
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
- Uses Gemini Live API via WebSocket (`gemini-3.1-flash-live-preview`)
- Sends raw mic audio, receives audio response -- no Whisper or Piper in conversation
- Whisper still used for wake word detection in sleep mode
- Runs Gemini call in a subprocess to avoid asyncio conflicts
- Requires `GEMINI_API_KEY` environment variable
- Free tier: ~1000 requests/day, 15 min audio-only sessions
- Voice set via `brain.gemini.voice` config (default: "Orus")
- Available voices: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr, and more
- Flow: Wake word (Whisper) -> Raw audio -> Gemini -> Raw audio playback -> Silence timeout -> Sleep

### Adding a new engine
1. Create `ari/brain/my_engine.py` implementing the `Brain` base class
2. Add it to the factory in `ari/brain/__init__.py`
3. Add config section under `brain:` in `config/default.yaml`

## Daemon Modes

The daemon (`ari/daemon.py`) operates differently based on the engine:

### Gemini mode
- Sleep: Whisper listens for wake phrase (e.g., "hello")
- Wake: sends raw audio to Gemini via subprocess, plays audio response
- Sleep trigger: silence timeout (configurable, default 15s)
- No Piper TTS, no VoiceID loaded -- minimal memory footprint
- Ctrl+C exits cleanly via signal handler (os._exit)

### Text pipeline mode (Gemma/Claude)
- Sleep: Whisper listens for wake phrase
- Wake: Whisper transcribes -> intent detection -> brain generates text -> Piper speaks
- Sleep trigger: silence timeout or "go back to sleep" phrase
- Full feature set: VoiceID, intent detection, camera commands, vision

## How to Run

Start Ari (Gemini mode):
```bash
cd ~/ari-assistant
PYTHONPATH=. GEMINI_API_KEY="your-key" ~/ari-assistant/bin/python -u -m ari.daemon
```

Start Ari (with camera daemon):
```bash
scripts/start_ari.sh
```

Stop Ari:
```bash
scripts/stop_ari.sh
```

Important: always kill old daemon processes before starting new ones.

## Configuration

All configurable values live in `config/default.yaml`. Never hardcode values in Python code -- read them from `ari.config.cfg` instead.

```python
from ari.config import cfg
sample_rate = cfg["audio"]["mic_sample_rate"]
```

### Key audio settings
- `silence_threshold: 1000` -- RMS threshold to detect speech (raise if background noise triggers false detection)
- `conversation_silence: 1.5` -- seconds of silence to end an utterance
- `max_recording_duration: 30` -- max seconds per utterance
- `silence_timeout: 15` -- seconds before auto-sleep in awake mode

## Speaker (TTS)

The speaker module (`ari/audio/speaker.py`) uses Piper TTS piped to aplay. Only used in Gemma/Claude modes, not Gemini.

Key optimization: `_speak_multi()` keeps a single aplay process alive across multiple sentences, eliminating the gap between sentences that occurs when spawning separate aplay processes.

## Wake Word Detection

Current setup uses **Whisper** for wake word detection:
- Config: `wake.phrases` list (default: "hello")
- Whisper transcribes 3-second audio chunks and checks for phrase match
- Works reliably but uses ~500MB RAM

openWakeWord is also available but needs a custom trained model for "hey ari":
- Pre-trained models available: alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer, weather
- Config: `wake.oww_model` and `wake.oww_threshold`
- Much lighter (~50MB RAM, <6ms per check) but no "hey ari" model yet

## Freenove Robot Arm

The robot arm is a **Freenove Robot Arm Kit (FNK0036)** for Raspberry Pi.

### Hardware
- **3 stepper motors** driven by A4988 modules via GPIO
- **5 servo channels** on GPIO pins 13, 16, 19, 20, 26 (via piolib on Pi 5)
- **3 infrared sensors** (TCRT5000) on GPIO 7, 8, 11 for homing
- **WS2812 RGB LEDs** and buzzer
- Requires **9-12.6V** external power (flat-top 18650 batteries or 12V adapter)
- Connected to Pi via **40-pin stacking header** (Geekworm) due to cooling board

### GPIO Pin Map
| Function | GPIO Pins |
|----------|-----------|
| A4988 Enable | 9 |
| A4988 Microstepping | 10, 24, 23 |
| A4988 Direction | 14, 15, 27 |
| A4988 Step | 4, 17, 22 |

### Confirmed Motor Pin Mapping
| Motor | Step GPIO | Dir GPIO | DIR.on() | DIR.off() |
|-------|-----------|----------|----------|-----------|
| Base | 22 | 27 | Right | Left |
| Shoulder | 17 | 15 | TBD | TBD |
| Elbow | 4 | 14 | TBD | TBD |
| Servo PWM | 13, 16, 19, 20, 26 |
| IR Sensors | 8, 11, 7 |

### Software
- Server code: `~/Freenove_Robot_Arm_Kit_for_Raspberry_Pi/Server/Code/`
- Client app: `~/Desktop/Freenove_Client/Arm_Software_Codes/` (PyQt5, runs on Mac)
- `libfreenove_pwm_lib.so` installed at `/usr/local/lib/` (for Pi 5 servo PWM)
- Start server: `cd ~/Freenove_Robot_Arm_Kit_for_Raspberry_Pi/Server/Code && sudo python3 main.py`
- Server listens on port 5000

### Important notes
- Button-top 18650 batteries don't provide enough current -- use flat-top or 12V 3A adapter
- A4988 Vref should be 0.7-0.8V (adjustable potentiometer, turn clockwise)
- Do NOT plug/unplug A4988 modules while powered
- Calibration requires manually positioning arm near sensor points first
- SPI must be disabled on Pi 5 (`dtparam=spi=off` in config.txt) to free GPIO 8 for sensors

## Infrastructure on ari (Raspberry Pi 5)

- **Ollama** v0.20.5 installed at `~/bin/ollama`, runs as user systemd service
  - Config: `~/.config/systemd/user/ollama.service`
  - Models stored in `~/.ollama/models/`
  - Libs at `~/lib/ollama/`
  - Manage: `systemctl --user start/stop/status ollama`
- **Gemini API key**: set via `GEMINI_API_KEY` env var (free tier from Google AI Studio)
- **Piper TTS**: binary at `~/ari-assistant/bin/piper`
- **Freenove piolib**: `libfreenove_pwm_lib.so` at `/usr/local/lib/`
- **UFW firewall**: ports 22 (SSH), 5000 (Freenove), 5900 (VNC) allowed

## Hardware Details

- **Board:** Raspberry Pi 5 (4GB RAM)
- **Cooling:** Cooling board on Pi (requires 40-pin stacking header for robot board)
- **Camera 1:** IMX708 ArduCam (CSI ribbon cable, captured via `rpicam-still`)
- **Camera 2:** Innomaker U20CAM 1080p USB webcam (on pan-tilt mount)
- **Microphone:** USB mic (1 or 2 channels, auto-detected)
- **Speaker:** ALSA output on `plughw:3,0`
- **Pan-tilt mount:** Two servos on robot board GPIO 13, 16
- **Robot arm:** Freenove FNK0036 (3 steppers + 2 servos + gripper)

## Servo Direction Mapping

From the camera's perspective:

| Direction  | Pulse change         |
|------------|----------------------|
| Tilt UP    | LOWER pulse (toward tilt_min) |
| Tilt DOWN  | HIGHER pulse (toward tilt_max) |
| Pan LEFT   | HIGHER pulse (toward pan_max) |
| Pan RIGHT  | LOWER pulse (toward pan_min) |

## Next Steps

- **Function calling**: Add Gemini Live API tool use for camera control and robot arm movement
- **Camera vision**: Send ArduCam frames to Gemini (1 per 5-10 seconds) for visual awareness
- **Custom wake word**: Train "hey ari" model for openWakeWord
- **Robot arm integration**: Map camera coordinates to arm X,Y,Z for pick-and-place
