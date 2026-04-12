"""Robot arm control for Ari.

Simple motor commands for the Freenove Robot Arm (FNK0036).
Uses gpiozero for GPIO control on Raspberry Pi 5.

Motor mapping (confirmed by hardware testing):
    Base:     Step=GPIO22, Dir=GPIO27, on=Right, off=Left
    Shoulder: Step=GPIO17, Dir=GPIO15, on=Forward, off=Backward
    Elbow:    Step=GPIO4,  Dir=GPIO14, on=Up, off=Down

Gripper: Servo on GPIO 13 via piolib
    Home=10 degrees, Open=35 degrees

Usage::

    from ari.hardware.arm import RobotArm
    arm = RobotArm()
    arm.move_front()
    arm.move_back()
    arm.move_left()
    arm.move_right()
    arm.pick_up()
    arm.close()
"""

from __future__ import annotations

import ctypes
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

# Gripper servo
GRIPPER_GPIO = 13
GRIPPER_HOME = 10   # degrees - small gap
GRIPPER_OPEN = 35   # degrees - teeth apart


class RobotArm:
    """Control the Freenove robot arm stepper motors and gripper."""

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

    # -- Named movement actions -----------------------------------------------

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

    def move_left(self, duration: float = DEFAULT_DURATION) -> None:
        """Move arm left (base rotates left)."""
        self._move_motor("base", "left", duration)
        self._disable()

    def move_right(self, duration: float = DEFAULT_DURATION) -> None:
        """Move arm right (base rotates right)."""
        self._move_motor("base", "right", duration)
        self._disable()

    # -- Single motor movements -----------------------------------------------

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

    # -- Gripper (servo on GPIO 13 via piolib) --------------------------------

    def _servo_angle(self, angle: int) -> None:
        """Set gripper servo to angle (0-180) using Freenove piolib."""
        lib = ctypes.CDLL("/usr/local/lib/libfreenove_pwm_lib.so")
        lib.pwm_init.argtypes = [ctypes.c_int]
        lib.pwm_init.restype = ctypes.c_void_p
        lib.pwm_deinit.argtypes = [ctypes.c_void_p]
        lib.pwm_deinit.restype = None
        lib.pwm_set_frequency.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.pwm_set_frequency.restype = ctypes.c_int
        lib.pwm_set_duty_cycle.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
        lib.pwm_set_duty_cycle.restype = ctypes.c_int
        lib.pwm_start.argtypes = [ctypes.c_void_p]
        lib.pwm_start.restype = None
        lib.pwm_stop.argtypes = [ctypes.c_void_p]
        lib.pwm_stop.restype = None

        inst = lib.pwm_init(GRIPPER_GPIO)
        lib.pwm_set_frequency(inst, 50)
        lib.pwm_start(inst)

        servo_duty = 500 + (2000 / 180) * angle
        duty_cycle = int((servo_duty / 20000.0) * 255)
        duty_cycle = max(0, min(255, duty_cycle))
        lib.pwm_set_duty_cycle(inst, duty_cycle)
        time.sleep(0.5)
        lib.pwm_stop(inst)
        lib.pwm_deinit(inst)

    def pick_up(self) -> None:
        """Pick up: open gripper, wait 1 second, close gripper."""
        log.info("pick_up")
        self._servo_angle(GRIPPER_OPEN)
        time.sleep(1)
        self._servo_angle(GRIPPER_HOME)

    def gripper_open(self) -> None:
        """Open gripper."""
        log.info("gripper_open")
        self._servo_angle(GRIPPER_OPEN)

    def gripper_home(self) -> None:
        """Close gripper to home position (small gap)."""
        log.info("gripper_home")
        self._servo_angle(GRIPPER_OPEN)
        time.sleep(0.3)
        self._servo_angle(GRIPPER_HOME)

    # -- Cleanup --------------------------------------------------------------

    def close(self) -> None:
        self._disable()
        self._en.close()
        for ms in self._ms:
            ms.close()
