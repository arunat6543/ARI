"""Pan-Tilt servo control for Ari — single source of truth.

SAFETY RULES (learned the hard way):
  - NEVER jump to a position — always move slowly
  - NEVER assume the servo's physical position matches tracked position
  - NEVER sweep more than MAX_MOVE_US (200us) in a single command
  - ALWAYS release PWM after movement to prevent buzzing
  - ALWAYS respect pan_min/pan_max limits — camera wire is fragile

Direction mapping (from the camera's perspective):
  - Tilt UP   = LOWER pulse values (toward tilt_min)
  - Tilt DOWN = HIGHER pulse values (toward tilt_max)
  - Pan LEFT  = HIGHER pulse values (toward pan_max)
  - Pan RIGHT = LOWER pulse values (toward pan_min)
"""

from __future__ import annotations

import subprocess
import time

import board
import busio
from adafruit_pca9685 import PCA9685

from ari.config import cfg

# Maximum microseconds to move in a single command.
# Anything larger is rejected to prevent violent swings.
MAX_MOVE_US = 300


class PanTilt:
    """Controls pan/tilt camera mount via PCA9685 PWM driver.

    On init, servos are OFF (no PWM). The first call to any move method
    will gently nudge from the commanded position. If the servo was moved
    manually, call ``sync_position(pan_us, tilt_us)`` to tell the code
    where the servo actually is before moving.
    """

    def __init__(self) -> None:
        scfg = cfg["servo"]

        self._pan_ch: int = scfg["pan_channel"]
        self._tilt_ch: int = scfg["tilt_channel"]

        self._pan_home: int = scfg["pan_home"]
        self._tilt_home: int = scfg["tilt_home"]

        self._pan_min: int = scfg["pan_min"]
        self._pan_max: int = scfg["pan_max"]
        self._tilt_min: int = scfg["tilt_min"]
        self._tilt_max: int = scfg["tilt_max"]

        self._step_us: int = scfg["step_us"]
        self._step_delay: float = scfg["step_delay"]

        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(self._i2c)
        self._pca.frequency = scfg["pca9685_frequency"]

        # Position is UNKNOWN on startup — servos are OFF
        self._pan_us: int | None = None
        self._tilt_us: int | None = None
        self._holding: bool = False

        # Do NOT send any PWM signal on init.
        # Servos stay wherever they physically are.

    # -- Low-level helpers ----------------------------------------------------

    @staticmethod
    def _us_to_duty(us: int) -> int:
        return int(us / 20_000 * 65_535)

    def _move_channel_safe(self, channel: int, target_us: int) -> None:
        """Move a channel to target, slowly and safely.

        If position is unknown (first move after init or manual adjustment),
        starts PWM at the target directly — this causes a small jump but
        only to the nearby target, not across full range.
        """
        # Get current tracked position for this channel
        if channel == self._pan_ch:
            current = self._pan_us
        else:
            current = self._tilt_us

        if current is None:
            # Position unknown — set directly (small jump to target)
            self._pca.channels[channel].duty_cycle = self._us_to_duty(target_us)
            time.sleep(0.3)
        else:
            # Move smoothly from known position
            step = self._step_us if target_us > current else -self._step_us
            for us in range(current, target_us, step):
                self._pca.channels[channel].duty_cycle = self._us_to_duty(us)
                time.sleep(self._step_delay)
            self._pca.channels[channel].duty_cycle = self._us_to_duty(target_us)

        # Release PWM unless holding
        if not self._holding:
            time.sleep(0.3)
            self._pca.channels[channel].duty_cycle = 0

    def _clamp_pan(self, us: int) -> int:
        return max(self._pan_min, min(self._pan_max, us))

    def _clamp_tilt(self, us: int) -> int:
        return max(self._tilt_min, min(self._tilt_max, us))

    def _check_move_distance(self, current: int | None, target: int) -> None:
        """Raise an error if the move distance exceeds MAX_MOVE_US.

        This prevents dangerous full-range sweeps.
        """
        if current is not None:
            distance = abs(target - current)
            if distance > MAX_MOVE_US:
                raise ValueError(
                    f"Move too large: {distance}us (max {MAX_MOVE_US}us). "
                    f"Break into smaller steps or call sync_position() first."
                )

    # -- Position sync --------------------------------------------------------

    def sync_position(self, pan_us: int, tilt_us: int) -> None:
        """Tell the code where the servos physically are.

        Call this after manually moving the servos, or on startup when
        you know the physical position. Does NOT move the servos.
        """
        self._pan_us = pan_us
        self._tilt_us = tilt_us

    def forget_position(self) -> None:
        """Mark position as unknown. Next move will set directly."""
        self._pan_us = None
        self._tilt_us = None

    # -- Hold / Release -------------------------------------------------------

    def hold(self) -> None:
        """Keep PWM active after moves (for scanning). Call release() when done."""
        self._holding = True

    def release(self) -> None:
        """Release PWM on both channels to stop buzzing."""
        self._holding = False
        self._pca.channels[self._pan_ch].duty_cycle = 0
        self._pca.channels[self._tilt_ch].duty_cycle = 0

    # -- Directional movement (relative) --------------------------------------

    def pan_left(self, amount: int = 200) -> None:
        """Pan camera left (higher pulse). Max 300us per call."""
        amount = min(amount, MAX_MOVE_US)
        current = self._pan_us if self._pan_us is not None else self._pan_home
        target = self._clamp_pan(current + amount)
        self._move_channel_safe(self._pan_ch, target)
        self._pan_us = target

    def pan_right(self, amount: int = 200) -> None:
        """Pan camera right (lower pulse). Max 300us per call."""
        amount = min(amount, MAX_MOVE_US)
        current = self._pan_us if self._pan_us is not None else self._pan_home
        target = self._clamp_pan(current - amount)
        self._move_channel_safe(self._pan_ch, target)
        self._pan_us = target

    def tilt_up(self, amount: int = 200) -> None:
        """Tilt camera up (lower pulse). Max 300us per call."""
        amount = min(amount, MAX_MOVE_US)
        current = self._tilt_us if self._tilt_us is not None else self._tilt_home
        target = self._clamp_tilt(current - amount)
        self._move_channel_safe(self._tilt_ch, target)
        self._tilt_us = target

    def tilt_down(self, amount: int = 200) -> None:
        """Tilt camera down (higher pulse). Max 300us per call."""
        amount = min(amount, MAX_MOVE_US)
        current = self._tilt_us if self._tilt_us is not None else self._tilt_home
        target = self._clamp_tilt(current + amount)
        self._move_channel_safe(self._tilt_ch, target)
        self._tilt_us = target

    # -- Absolute positioning -------------------------------------------------

    def home(self) -> None:
        """Move to home position in safe steps (multiple moves if needed)."""
        self.set_position(self._pan_home, self._tilt_home)

    def set_position(self, pan_us: int, tilt_us: int) -> None:
        """Move to an absolute position, in safe steps.

        If the move is larger than MAX_MOVE_US, it's broken into
        multiple smaller steps automatically.
        """
        pan_us = self._clamp_pan(pan_us)
        tilt_us = self._clamp_tilt(tilt_us)

        # Move pan in safe chunks
        current_pan = self._pan_us
        if current_pan is None:
            # Unknown position — jump directly to target (small risk)
            self._move_channel_safe(self._pan_ch, pan_us)
            self._pan_us = pan_us
        else:
            while current_pan != pan_us:
                diff = pan_us - current_pan
                step = max(-MAX_MOVE_US, min(MAX_MOVE_US, diff))
                next_pos = current_pan + step
                self._move_channel_safe(self._pan_ch, next_pos)
                self._pan_us = next_pos
                current_pan = next_pos

        # Move tilt in safe chunks
        current_tilt = self._tilt_us
        if current_tilt is None:
            self._move_channel_safe(self._tilt_ch, tilt_us)
            self._tilt_us = tilt_us
        else:
            while current_tilt != tilt_us:
                diff = tilt_us - current_tilt
                step = max(-MAX_MOVE_US, min(MAX_MOVE_US, diff))
                next_pos = current_tilt + step
                self._move_channel_safe(self._tilt_ch, next_pos)
                self._tilt_us = next_pos
                current_tilt = next_pos

    # -- Camera capture -------------------------------------------------------

    def capture(self, filename: str = "/tmp/cam_live.jpg") -> str:
        ccfg = cfg["camera"]
        subprocess.run(
            ["rpicam-still", "-o", filename,
             "--timeout", str(ccfg["capture_timeout"]),
             "--width", str(ccfg["capture_width"]),
             "--height", str(ccfg["capture_height"]),
             "--nopreview"],
            capture_output=True, timeout=15,
        )
        return filename

    # -- Properties and cleanup -----------------------------------------------

    @property
    def position(self) -> dict[str, int | None]:
        return {"pan_us": self._pan_us, "tilt_us": self._tilt_us}

    def close(self) -> None:
        """Relax servos and release I2C."""
        self._pca.channels[self._pan_ch].duty_cycle = 0
        self._pca.channels[self._tilt_ch].duty_cycle = 0
        self._pca.deinit()
