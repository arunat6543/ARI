#!/usr/bin/env python3
"""
Persistent camera + pan-tilt daemon for Ari.
Keeps PCA9685 connection alive. Accepts commands via a named pipe (FIFO).

Commands:
  pan_left [amount]     Pan camera left (default 200us)
  pan_right [amount]    Pan camera right
  tilt_up [amount]      Tilt camera up
  tilt_down [amount]    Tilt camera down
  home                  Go to home position
  set <pan_us> <tilt_us>  Set absolute position
  capture [filename]    Capture image (default /tmp/cam_live.jpg)
  position              Print current position
  quit                  Exit daemon
"""

import os
import sys
import subprocess
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# ── Config ─────────────────────────────────────────────────────────────────────

FIFO_PATH = "/tmp/ari_camera_cmd"
STATUS_PATH = "/tmp/ari_camera_status"

PAN_CHANNEL = 0
TILT_CHANNEL = 1

PAN_HOME = 1600
TILT_HOME = 2200

PAN_MIN = 500
PAN_MAX = 2500
TILT_MIN = 1500
TILT_MAX = 2800

STEP_US = 5
STEP_DELAY = 0.012


# ── Servo Control ──────────────────────────────────────────────────────────────

class PanTiltDaemon:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        self.pan_us = PAN_HOME
        self.tilt_us = TILT_HOME
        # Set home position immediately
        self._set_direct(PAN_HOME, TILT_HOME)

    def _us_to_duty(self, us):
        return int(us / 20000 * 65535)

    def _set_direct(self, pan_us, tilt_us):
        self.pca.channels[PAN_CHANNEL].duty_cycle = self._us_to_duty(pan_us)
        self.pca.channels[TILT_CHANNEL].duty_cycle = self._us_to_duty(tilt_us)
        self.pan_us = pan_us
        self.tilt_us = tilt_us

    def _move_channel(self, channel, from_us, to_us):
        step = STEP_US if to_us > from_us else -STEP_US
        for us in range(from_us, to_us, step):
            self.pca.channels[channel].duty_cycle = self._us_to_duty(us)
            time.sleep(STEP_DELAY)
        self.pca.channels[channel].duty_cycle = self._us_to_duty(to_us)

    def pan_left(self, amount=200):
        target = min(PAN_MAX, self.pan_us + amount)
        self._move_channel(PAN_CHANNEL, self.pan_us, target)
        self.pan_us = target

    def pan_right(self, amount=200):
        target = max(PAN_MIN, self.pan_us - amount)
        self._move_channel(PAN_CHANNEL, self.pan_us, target)
        self.pan_us = target

    def tilt_up(self, amount=200):
        target = max(TILT_MIN, self.tilt_us - amount)
        self._move_channel(TILT_CHANNEL, self.tilt_us, target)
        self.tilt_us = target

    def tilt_down(self, amount=200):
        target = min(TILT_MAX, self.tilt_us + amount)
        self._move_channel(TILT_CHANNEL, self.tilt_us, target)
        self.tilt_us = target

    def home(self):
        self._move_channel(PAN_CHANNEL, self.pan_us, PAN_HOME)
        self.pan_us = PAN_HOME
        self._move_channel(TILT_CHANNEL, self.tilt_us, TILT_HOME)
        self.tilt_us = TILT_HOME

    def set_position(self, pan_us, tilt_us):
        pan_us = max(PAN_MIN, min(PAN_MAX, pan_us))
        tilt_us = max(TILT_MIN, min(TILT_MAX, tilt_us))
        self._move_channel(PAN_CHANNEL, self.pan_us, pan_us)
        self.pan_us = pan_us
        self._move_channel(TILT_CHANNEL, self.tilt_us, tilt_us)
        self.tilt_us = tilt_us

    def capture(self, filename="/tmp/cam_live.jpg"):
        subprocess.run(
            ["rpicam-still", "-o", filename, "--timeout", "1500",
             "--width", "1280", "--height", "960", "--nopreview"],
            capture_output=True, timeout=10
        )
        return filename

    @property
    def position(self):
        return f"pan={self.pan_us} tilt={self.tilt_us}"

    def close(self):
        self.pca.channels[PAN_CHANNEL].duty_cycle = 0
        self.pca.channels[TILT_CHANNEL].duty_cycle = 0
        self.pca.deinit()


def write_status(msg):
    with open(STATUS_PATH, "w") as f:
        f.write(msg + "\n")


def main():
    # Create FIFO if it doesn't exist
    if os.path.exists(FIFO_PATH):
        os.unlink(FIFO_PATH)
    os.mkfifo(FIFO_PATH)

    print(f"🎥 Camera daemon starting...")
    print(f"   Command pipe: {FIFO_PATH}")
    print(f"   Status file:  {STATUS_PATH}")

    daemon = PanTiltDaemon()
    print(f"✅ Ready at home position ({daemon.position})")
    write_status(f"ready {daemon.position}")

    try:
        while True:
            # Open FIFO for reading (blocks until a writer connects)
            with open(FIFO_PATH, "r") as fifo:
                for line in fifo:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    cmd = parts[0].lower()
                    args = parts[1:]

                    try:
                        if cmd == "pan_left":
                            amt = int(args[0]) if args else 200
                            daemon.pan_left(amt)
                            write_status(f"ok {daemon.position}")

                        elif cmd == "pan_right":
                            amt = int(args[0]) if args else 200
                            daemon.pan_right(amt)
                            write_status(f"ok {daemon.position}")

                        elif cmd == "tilt_up":
                            amt = int(args[0]) if args else 200
                            daemon.tilt_up(amt)
                            write_status(f"ok {daemon.position}")

                        elif cmd == "tilt_down":
                            amt = int(args[0]) if args else 200
                            daemon.tilt_down(amt)
                            write_status(f"ok {daemon.position}")

                        elif cmd == "home":
                            daemon.home()
                            write_status(f"ok {daemon.position}")

                        elif cmd == "set":
                            pan = int(args[0])
                            tilt = int(args[1])
                            daemon.set_position(pan, tilt)
                            write_status(f"ok {daemon.position}")

                        elif cmd == "capture":
                            fname = args[0] if args else "/tmp/cam_live.jpg"
                            daemon.capture(fname)
                            write_status(f"captured {fname} {daemon.position}")

                        elif cmd == "position":
                            write_status(f"ok {daemon.position}")

                        elif cmd == "quit":
                            write_status("bye")
                            daemon.close()
                            print("👋 Shutting down")
                            return

                        else:
                            write_status(f"error unknown command: {cmd}")

                        print(f"  [{cmd}] → {daemon.position}")

                    except Exception as e:
                        write_status(f"error {e}")
                        print(f"  [{cmd}] ERROR: {e}")

    except KeyboardInterrupt:
        print("\n👋 Shutting down")
        daemon.close()
    finally:
        if os.path.exists(FIFO_PATH):
            os.unlink(FIFO_PATH)


if __name__ == "__main__":
    main()
