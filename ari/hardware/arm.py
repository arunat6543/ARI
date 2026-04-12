"""Robot arm control for Ari.

Simple motor commands for the Freenove Robot Arm (FNK0036).
Uses gpiozero for GPIO control on Raspberry Pi 5.

Motor mapping (confirmed by hardware testing):
    Base:     Step=GPIO22, Dir=GPIO27, on=Right, off=Left
    Shoulder: Step=GPIO17, Dir=GPIO15, on=Forward, off=Backward
    Elbow:    Step=GPIO4,  Dir=GPIO14, on=Up, off=Down

Usage::

    from ari.hardware.arm import RobotArm
    arm = RobotArm()
    arm.move_front()
    arm.move_back()
    arm.base_left()
    arm.close()
"""

from __future__ import annotations

import logging
import time

from gpiozero import OutputDevice

log = logging.getLogger(__name__)

# Motor pin mapping
MOTORS = {
    "base":     {"step": 22, "dir": 27},
    "shoulder": {"step": 17, "dir": 15},
    "elbow":    {"step": 4,  "dir": 14},
}

# Direction mapping
DIRECTIONS = {
    "base":     {"right": True, "left": False},
    "shoulder": {"forward": True, "backward": False},
    "elbow":    {"up": True, "down": False},
}

# Shared pins
A4988_EN = 9
A4988_MS = [10, 24, 23]

DEFAULT_DURATION = 0.75  # seconds per movement


class RobotArm:
    """Control the Freenove robot arm stepper motors."""

    def __init__(self) -> None:
        self._en = OutputDevice(A4988_EN, initial_value=True)
        self._ms = [OutputDevice(pin, initial_value=False) for pin in A4988_MS]
        self._enabled = False

    def _enable(self) -> None:
        if not self._enabled:
            self._en.off()  # active LOW
            self._enabled = True

    def _disable(self) -> None:
        if self._enabled:
            self._en.on()
            self._enabled = False

    def _move_motor(self, motor: str, direction: str, duration: float = DEFAULT_DURATION) -> None:
        """Move a single motor in a direction for a duration."""
        pins = MOTORS[motor]
        dir_value = DIRECTIONS[motor][direction]

        dir_pin = OutputDevice(pins["dir"], initial_value=False)
        step_pin = OutputDevice(pins["step"], initial_value=False)

        self._enable()

        if dir_value:
            dir_pin.on()
        else:
            dir_pin.off()

        log.info("%s %s %.2fs", motor, direction, duration)
        end_time = time.time() + duration
        while time.time() < end_time:
            step_pin.on()
            time.sleep(0.002)
            step_pin.off()
            time.sleep(0.002)

        dir_pin.close()
        step_pin.close()

    # -- Named actions --------------------------------------------------------

    def move_front(self, duration: float = DEFAULT_DURATION) -> None:
        """Move arm front: elbow up, shoulder forward, elbow down."""
        self._move_motor("elbow", "up", duration)
        self._move_motor("shoulder", "forward", duration)
        self._move_motor("elbow", "down", duration)
        self._disable()

    def move_back(self, duration: float = DEFAULT_DURATION) -> None:
        """Move arm back: elbow up, shoulder backward, elbow down."""
        self._move_motor("elbow", "up", duration)
        self._move_motor("shoulder", "backward", duration)
        self._move_motor("elbow", "down", duration)
        self._disable()

    # -- Single motor movements -----------------------------------------------

    def base_left(self, duration: float = DEFAULT_DURATION) -> None:
        self._move_motor("base", "left", duration)
        self._disable()

    def base_right(self, duration: float = DEFAULT_DURATION) -> None:
        self._move_motor("base", "right", duration)
        self._disable()

    def shoulder_forward(self, duration: float = DEFAULT_DURATION) -> None:
        self._move_motor("shoulder", "forward", duration)
        self._disable()

    def shoulder_backward(self, duration: float = DEFAULT_DURATION) -> None:
        self._move_motor("shoulder", "backward", duration)
        self._disable()

    def elbow_up(self, duration: float = DEFAULT_DURATION) -> None:
        self._move_motor("elbow", "up", duration)
        self._disable()

    def elbow_down(self, duration: float = DEFAULT_DURATION) -> None:
        self._move_motor("elbow", "down", duration)
        self._disable()

    # -- Cleanup --------------------------------------------------------------

    def close(self) -> None:
        self._disable()
        self._en.close()
        for ms in self._ms:
            ms.close()
