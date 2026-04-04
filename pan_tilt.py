#!/usr/bin/env python3
"""
Pan-Tilt Camera Control for Ari — calibrated for PCA9685 on RPi 5.

Direction mapping (from camera's perspective):
  - Tilt UP:    LOWER pulse values (toward 500us)
  - Tilt DOWN:  HIGHER pulse values (toward 2800us)
  - Pan LEFT:   HIGHER pulse values (toward 2500us)
  - Pan RIGHT:  LOWER pulse values (toward 500us)

Home position (facing straight forward):
  - Pan:  1600us
  - Tilt: 2200us

Safety limits:
  - Tilt: max ±45° from center (camera wire constraint)
  - Pan:  full range OK
"""

import board
import busio
from adafruit_pca9685 import PCA9685
import time

# ── Calibration ────────────────────────────────────────────────────────────────

PAN_CHANNEL = 0
TILT_CHANNEL = 1

# Home position (facing forward)
PAN_HOME = 1600    # microseconds
TILT_HOME = 2200   # microseconds

# Limits (microseconds)
PAN_MIN = 500
PAN_MAX = 2500
TILT_MIN = 1500    # max UP (limited by cable)
TILT_MAX = 2800    # max DOWN (limited by cable)

# Movement speed
STEP_US = 5        # microseconds per step
STEP_DELAY = 0.015 # seconds between steps


class PanTilt:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        self.pan_us = PAN_HOME
        self.tilt_us = TILT_HOME

    def _us_to_duty(self, us):
        return int(us / 20000 * 65535)

    def _move_smooth(self, channel, from_us, to_us):
        """Move servo smoothly from current to target position."""
        step = STEP_US if to_us > from_us else -STEP_US
        for us in range(from_us, to_us, step):
            self.pca.channels[channel].duty_cycle = self._us_to_duty(us)
            time.sleep(STEP_DELAY)
        self.pca.channels[channel].duty_cycle = self._us_to_duty(to_us)

    def home(self):
        """Move to home position (facing forward)."""
        self.move_to(PAN_HOME, TILT_HOME)

    def move_to(self, pan_us=None, tilt_us=None):
        """Move to absolute position in microseconds."""
        if pan_us is not None:
            pan_us = max(PAN_MIN, min(PAN_MAX, pan_us))
            self._move_smooth(PAN_CHANNEL, self.pan_us, pan_us)
            self.pan_us = pan_us

        if tilt_us is not None:
            tilt_us = max(TILT_MIN, min(TILT_MAX, tilt_us))
            self._move_smooth(TILT_CHANNEL, self.tilt_us, tilt_us)
            self.tilt_us = tilt_us

    def pan_left(self, amount_us=200):
        """Pan camera left."""
        self.move_to(pan_us=self.pan_us + amount_us)

    def pan_right(self, amount_us=200):
        """Pan camera right."""
        self.move_to(pan_us=self.pan_us - amount_us)

    def tilt_up(self, amount_us=200):
        """Tilt camera up."""
        self.move_to(tilt_us=self.tilt_us - amount_us)

    def tilt_down(self, amount_us=200):
        """Tilt camera down."""
        self.move_to(tilt_us=self.tilt_us + amount_us)

    def look_at(self, direction):
        """Move camera by direction name."""
        directions = {
            "left": self.pan_left,
            "right": self.pan_right,
            "up": self.tilt_up,
            "down": self.tilt_down,
            "home": lambda: self.home(),
            "center": lambda: self.home(),
        }
        func = directions.get(direction.lower())
        if func:
            func()
        else:
            print(f"Unknown direction: {direction}")

    def set_position(self, pan_us, tilt_us):
        """Set position directly without smooth movement."""
        pan_us = max(PAN_MIN, min(PAN_MAX, pan_us))
        tilt_us = max(TILT_MIN, min(TILT_MAX, tilt_us))
        self.pca.channels[PAN_CHANNEL].duty_cycle = self._us_to_duty(pan_us)
        self.pca.channels[TILT_CHANNEL].duty_cycle = self._us_to_duty(tilt_us)
        self.pan_us = pan_us
        self.tilt_us = tilt_us

    def relax(self):
        """Stop sending PWM — servos go limp."""
        self.pca.channels[PAN_CHANNEL].duty_cycle = 0
        self.pca.channels[TILT_CHANNEL].duty_cycle = 0

    def close(self):
        """Clean up."""
        self.relax()
        self.pca.deinit()

    @property
    def position(self):
        return {"pan_us": self.pan_us, "tilt_us": self.tilt_us}


if __name__ == "__main__":
    import sys

    pt = PanTilt()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        pt.set_position(PAN_HOME, TILT_HOME)
        time.sleep(0.3)
        pt.look_at(cmd)
        print(f"Moved {cmd}: {pt.position}")
    else:
        print("Going to home position...")
        pt.set_position(PAN_HOME, TILT_HOME)
        print(f"Home: {pt.position}")

    time.sleep(1)
    pt.close()
