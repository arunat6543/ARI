#!/bin/bash
# Stop Ari - sends quit commands, falls back to pkill
#
# Attempts a graceful shutdown via FIFO commands first.
# If processes are still running after 3 seconds, uses pkill.

set -uo pipefail

VOICE_FIFO="/tmp/ari_voice_cmd"
CAMERA_FIFO="/tmp/ari_camera_cmd"

echo "Stopping Ari..."

# ── Graceful shutdown via FIFOs ──────────────────────────────────────────

for fifo in "$VOICE_FIFO" "$CAMERA_FIFO"; do
    if [ -p "$fifo" ]; then
        echo "  Sending quit to $fifo"
        echo "quit" > "$fifo" 2>/dev/null || true
    else
        echo "  FIFO not found: $fifo"
    fi
done

echo "  Waiting for graceful shutdown..."
sleep 3

# ── Check if processes are still running ─────────────────────────────────

VOICE_ALIVE=false
CAMERA_ALIVE=false

if pgrep -f "python.*ari.daemon" > /dev/null 2>&1; then
    VOICE_ALIVE=true
fi
if pgrep -f "python.*camera_daemon" > /dev/null 2>&1; then
    CAMERA_ALIVE=true
fi

# ── Force-kill any remaining processes ───────────────────────────────────

if $VOICE_ALIVE; then
    echo "  Voice daemon still running, sending pkill..."
    pkill -f "python.*ari.daemon" 2>/dev/null || true
fi

if $CAMERA_ALIVE; then
    echo "  Camera daemon still running, sending pkill..."
    pkill -f "python.*camera_daemon" 2>/dev/null || true
fi

if $VOICE_ALIVE || $CAMERA_ALIVE; then
    sleep 1
fi

# ── Clean up stale FIFOs ────────────────────────────────────────────────

rm -f "$VOICE_FIFO" "$CAMERA_FIFO"

# ── Verify ───────────────────────────────────────────────────────────────

if pgrep -f "python.*(ari.daemon|camera_daemon)" > /dev/null 2>&1; then
    echo ""
    echo "WARNING: Some processes may still be running:"
    pgrep -af "python.*(ari.daemon|camera_daemon)" 2>/dev/null || true
    echo "Use 'kill -9 <pid>' to force-stop them."
else
    echo ""
    echo "Ari stopped."
fi
