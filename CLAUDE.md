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
    brain/                 # Claude LLM integration and intent detection
      claude_client.py
      intent.py
    hardware/              # Hardware abstraction (servos, future arm)
      servo.py             # PanTilt class — single source of truth for servo control
    ipc/                   # Inter-process communication
      fifo.py              # FifoServer (daemon side) and FifoClient (caller side)
    skills/                # Pluggable skill modules
      base.py              # Abstract Skill base class
    vision/                # Camera capture and person scanning
      camera.py
      scanner.py
    config.py              # YAML config loader, exposes global `cfg` singleton
  config/
    default.yaml           # All configuration values (the single source of truth)
    local.yaml             # Machine-specific overrides (git-ignored)
  models/                  # Piper TTS model files
  scripts/                 # Start/stop scripts
  camera_daemon_new.py     # Refactored camera daemon using ari modules
```

## How to Run

Start Ari:
```bash
scripts/start_ari.sh
```

Stop Ari:
```bash
scripts/stop_ari.sh
```

Important: always kill old daemon processes before starting new ones. The start script should handle this, but if debugging manually, check for leftover processes with `ps aux | grep ari` and kill them.

## Configuration

All configurable values live in `config/default.yaml`. Never hardcode values in Python code — read them from `ari.config.cfg` instead.

```python
from ari.config import cfg
sample_rate = cfg["audio"]["mic_sample_rate"]
```

To add a new config value:
1. Add the key with a sensible default to `config/default.yaml` under the appropriate section.
2. Read it via `cfg["section"]["key"]` in your code.
3. If you need a machine-specific override, put it in `config/local.yaml` (same structure, only the keys you want to change).

## How to Add a New Skill

1. Create a new file in `ari/skills/`, e.g. `ari/skills/chess.py`.
2. Import and implement the `Skill` base class:

```python
from ari.skills.base import Skill

class ChessSkill(Skill):
    name = "chess"

    def can_handle(self, intent: dict) -> bool:
        return intent.get("type") == "chess"

    def handle(self, text: str, intent: dict, context: dict) -> None:
        speaker = context["speaker"]
        # ... skill logic here ...
        speaker.say("Your move!")
```

3. Register the skill in the main assistant loop so it gets checked via `can_handle`.

The `context` dict passed to `handle` contains shared resources:
- `speaker` — `ari.audio.speaker.Speaker`
- `claude` — `ari.brain.claude_client.ClaudeClient`
- `camera_client` — `ari.ipc.fifo.FifoClient`
- `scanner` — `ari.vision.scanner.PersonScanner`
- `session_id` — current Claude conversation session ID (or None)

## How to Add New Hardware

1. Create a new file in `ari/hardware/`, e.g. `ari/hardware/arm.py`.
2. Read all calibration constants from `cfg` (add them to `config/default.yaml` first).
3. Provide a `close()` method that releases resources (I2C bus, GPIO, etc.).
4. Import and use from daemons or skills as needed.

## IPC Protocol

Daemons communicate via named pipes (FIFOs). Each daemon has a command FIFO and a status file:

- Camera daemon: `/tmp/ari_camera_cmd` (commands) and `/tmp/ari_camera_status` (status)
- Voice daemon: `/tmp/ari_voice_cmd` (commands) and `/tmp/ari_voice_status` (status)

Daemon side uses `FifoServer`, client side uses `FifoClient`:

```python
# Daemon
from ari.ipc.fifo import FifoServer
srv = FifoServer("/tmp/ari_camera_cmd")
for cmd in srv:
    handle(cmd)
    srv.reply("ok pan=1600 tilt=2200")

# Client
from ari.ipc.fifo import FifoClient
cli = FifoClient("/tmp/ari_camera_cmd")
status = cli.send("pan_left 200")
```

The status file path is derived automatically by replacing `_cmd` with `_status` in the FIFO path.

## Daemon Code Rules

- Use `flush=True` on all `print()` statements in daemon code. Daemons run with buffered stdout and logs will not appear in real time without explicit flushing.
- Always write a status reply after handling each command so clients do not hang.
- Handle `KeyboardInterrupt` and clean up (close hardware, remove FIFO).

## Hardware Details

- **Board:** Raspberry Pi 5
- **Camera:** IMX708 (captured via `rpicam-still`)
- **Microphone:** USB mic (may present as 1-channel or 2-channel depending on the device — code must handle both)
- **Speaker:** ALSA output on `plughw:3,0`
- **Pan-tilt mount:** Two servos on a PCA9685 PWM driver over I2C
  - Pan = channel 0, Tilt = channel 1
  - PCA9685 frequency: 50 Hz

## Servo Direction Mapping

From the camera's perspective:

| Direction  | Pulse change         |
|------------|----------------------|
| Tilt UP    | LOWER pulse (toward tilt_min) |
| Tilt DOWN  | HIGHER pulse (toward tilt_max) |
| Pan LEFT   | HIGHER pulse (toward pan_max) |
| Pan RIGHT  | LOWER pulse (toward pan_min) |

This is counterintuitive — double-check the direction mapping in `ari/hardware/servo.py` if servo movement seems reversed.
