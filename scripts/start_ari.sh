#!/bin/bash
# Start Ari - kills any existing processes first
#
# Starts the camera daemon and the voice daemon in order.
# Both run as background processes with output logged to /tmp.

set -euo pipefail

ARI_DIR="$HOME/ari-assistant"
CAMERA_DAEMON="$ARI_DIR/camera_daemon.py"
ARI_DAEMON_MODULE="ari.daemon"
PYTHON="$ARI_DIR/bin/python"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ARI_DIR"

# Use Claude Code OAuth token if ANTHROPIC_API_KEY not set
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    export ANTHROPIC_API_KEY="$CLAUDE_CODE_OAUTH_TOKEN"
fi

CAMERA_LOG="/tmp/ari_camera.log"
VOICE_LOG="/tmp/ari_voice.log"

CAMERA_FIFO="/tmp/ari_camera_cmd"
VOICE_FIFO="/tmp/ari_voice_cmd"

# ── Kill existing processes ────────────────────────────────────────────────

echo "Stopping any existing Ari processes..."

# Send quit to FIFOs first (graceful shutdown).
for fifo in "$VOICE_FIFO" "$CAMERA_FIFO"; do
    if [ -p "$fifo" ]; then
        echo "quit" > "$fifo" 2>/dev/null || true
    fi
done
sleep 1

# Fall back to pkill if they're still running.
pkill -f "python.*ari.daemon" 2>/dev/null || true
pkill -f "python.*camera_daemon" 2>/dev/null || true
sleep 1

# Free the mic device in case something is holding it.
fuser -k /dev/snd/pcmC2D0c 2>/dev/null || true

# Clean up stale FIFOs.
rm -f "$VOICE_FIFO" "$CAMERA_FIFO"

echo "Existing processes stopped."

# ── Start camera daemon ───────────────────────────────────────────────────

echo "Starting camera daemon..."
cd "$ARI_DIR"
$PYTHON "$CAMERA_DAEMON" > "$CAMERA_LOG" 2>&1 &
CAMERA_PID=$!
echo "  Camera daemon PID: $CAMERA_PID"
echo "  Log: $CAMERA_LOG"

# Wait for camera daemon to create its FIFO.
echo "  Waiting for camera daemon to be ready..."
sleep 3

if ! kill -0 "$CAMERA_PID" 2>/dev/null; then
    echo "ERROR: Camera daemon failed to start. Check $CAMERA_LOG"
    exit 1
fi

# ── Start voice daemon ───────────────────────────────────────────────────

echo "Starting Ari voice daemon..."
cd "$ARI_DIR"
$PYTHON -u -m "$ARI_DAEMON_MODULE" > "$VOICE_LOG" 2>&1 &
VOICE_PID=$!
echo "  Voice daemon PID: $VOICE_PID"
echo "  Log: $VOICE_LOG"

sleep 2

if ! kill -0 "$VOICE_PID" 2>/dev/null; then
    echo "ERROR: Voice daemon failed to start. Check $VOICE_LOG"
    exit 1
fi

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Ari is running!"
echo "  Camera daemon:  PID $CAMERA_PID"
echo "  Voice daemon:   PID $VOICE_PID"
echo "  Camera log:     $CAMERA_LOG"
echo "  Voice log:      $VOICE_LOG"
echo "========================================"
echo ""
echo "Commands:"
echo "  echo wake  > $VOICE_FIFO"
echo "  echo sleep > $VOICE_FIFO"
echo "  echo quit  > $VOICE_FIFO"
echo "  tail -f $VOICE_LOG"
