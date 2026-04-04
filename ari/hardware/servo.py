"""Pan-Tilt servo control for Ari — single source of truth.

Wraps the PCA9685 servo driver on I2C bus 1.  All calibration constants
come from ``ari.config.cfg`` so nothing is hardcoded here.

Direction mapping (from the camera's perspective):
  - Tilt UP   = LOWER pulse values (toward tilt_min)
  - Tilt DOWN = HIGHER pulse values (toward tilt_max)
  - Pan LEFT  = HIGHER pulse values (toward pan_max)
  - Pan RIGHT = LOWER pulse values (toward pan_min)

Usage::

    from ari.hardware.servo import PanTilt
    pt = PanTilt()
    pt.pan_left(300)
    pt.capture("/tmp/snapshot.jpg")
    pt.home()
    pt.close()
"""

from __future__ import annotations

import subprocess
import time

import board
import busio
from adafruit_pca9685 import PCA9685

from ari.config import cfg


class PanTilt:
    """Controls pan/tilt camera mount via PCA9685 PWM driver."""

    def __init__(self) -> None:
        scfg = cfg["servo"]

        # Channel assignments
        self._pan_ch: int = scfg["pan_channel"]
        self._tilt_ch: int = scfg["tilt_channel"]

        # Home positions (microseconds)
        self._pan_home: int = scfg["pan_home"]
        self._tilt_home: int = scfg["tilt_home"]

        # Limits (microseconds)
        self._pan_min: int = scfg["pan_min"]
        self._pan_max: int = scfg["pan_max"]
        self._tilt_min: int = scfg["tilt_min"]
        self._tilt_max: int = scfg["tilt_max"]

        # Movement tuning
        self._step_us: int = scfg["step_us"]
        self._step_delay: float = scfg["step_delay"]

        # Initialise PCA9685
        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(self._i2c)
        self._pca.frequency = scfg["pca9685_frequency"]

        # Track current position
        self._pan_us: int = self._pan_home
        self._tilt_us: int = self._tilt_home

        # Move to home on startup
        self._pca.channels[self._pan_ch].duty_cycle = self._us_to_duty(self._pan_home)
        self._pca.channels[self._tilt_ch].duty_cycle = self._us_to_duty(self._tilt_home)

    # -- Low-level helpers ----------------------------------------------------

    @staticmethod
    def _us_to_duty(us: int) -> int:
        """Convert microseconds to a 16-bit duty-cycle value (0-65535).

        The PCA9685 period is 20 000 us at 50 Hz, so::

            duty = us / 20000 * 65535
        """
        return int(us / 20_000 * 65_535)

    def _move_channel(self, channel: int, from_us: int, to_us: int) -> None:
        """Smoothly sweep *channel* from *from_us* to *to_us* in small steps."""
        step = self._step_us if to_us > from_us else -self._step_us
        for us in range(from_us, to_us, step):
            self._pca.channels[channel].duty_cycle = self._us_to_duty(us)
            time.sleep(self._step_delay)
        # Ensure we land exactly on the target
        self._pca.channels[channel].duty_cycle = self._us_to_duty(to_us)

    def _clamp_pan(self, us: int) -> int:
        return max(self._pan_min, min(self._pan_max, us))

    def _clamp_tilt(self, us: int) -> int:
        return max(self._tilt_min, min(self._tilt_max, us))

    # -- Directional movement -------------------------------------------------

    def pan_left(self, amount: int = 200) -> None:
        """Pan camera left (higher pulse)."""
        target = self._clamp_pan(self._pan_us + amount)
        self._move_channel(self._pan_ch, self._pan_us, target)
        self._pan_us = target

    def pan_right(self, amount: int = 200) -> None:
        """Pan camera right (lower pulse)."""
        target = self._clamp_pan(self._pan_us - amount)
        self._move_channel(self._pan_ch, self._pan_us, target)
        self._pan_us = target

    def tilt_up(self, amount: int = 200) -> None:
        """Tilt camera up (lower pulse)."""
        target = self._clamp_tilt(self._tilt_us - amount)
        self._move_channel(self._tilt_ch, self._tilt_us, target)
        self._tilt_us = target

    def tilt_down(self, amount: int = 200) -> None:
        """Tilt camera down (higher pulse)."""
        target = self._clamp_tilt(self._tilt_us + amount)
        self._move_channel(self._tilt_ch, self._tilt_us, target)
        self._tilt_us = target

    # -- Absolute positioning -------------------------------------------------

    def home(self) -> None:
        """Smoothly return to home (forward-facing) position."""
        self.set_position(self._pan_home, self._tilt_home)

    def set_position(self, pan_us: int, tilt_us: int) -> None:
        """Move smoothly to an absolute position, clamped to safe limits."""
        pan_us = self._clamp_pan(pan_us)
        tilt_us = self._clamp_tilt(tilt_us)
        self._move_channel(self._pan_ch, self._pan_us, pan_us)
        self._pan_us = pan_us
        self._move_channel(self._tilt_ch, self._tilt_us, tilt_us)
        self._tilt_us = tilt_us

    # -- Camera capture -------------------------------------------------------

    def capture(self, filename: str = "/tmp/cam_live.jpg") -> str:
        """Capture a still image via ``rpicam-still``.

        Parameters
        ----------
        filename:
            Output path for the JPEG image.

        Returns
        -------
        str
            The *filename* that was written.
        """
        ccfg = cfg["camera"]
        subprocess.run(
            [
                "rpicam-still",
                "-o", filename,
                "--timeout", str(ccfg["capture_timeout"]),
                "--width", str(ccfg["capture_width"]),
                "--height", str(ccfg["capture_height"]),
                "--nopreview",
            ],
            capture_output=True,
            timeout=15,
        )
        return filename

    # -- Properties and cleanup -----------------------------------------------

    @property
    def position(self) -> dict[str, int]:
        """Current position as ``{"pan_us": ..., "tilt_us": ...}``."""
        return {"pan_us": self._pan_us, "tilt_us": self._tilt_us}

    def close(self) -> None:
        """Relax servos (stop PWM) and release the I2C bus."""
        self._pca.channels[self._pan_ch].duty_cycle = 0
        self._pca.channels[self._tilt_ch].duty_cycle = 0
        self._pca.deinit()
