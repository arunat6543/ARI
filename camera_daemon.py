#!/usr/bin/env python3
"""Refactored camera + pan-tilt daemon for Ari.

Uses the shared ari modules instead of inline constants:
  - ari.config.cfg        for all configuration
  - ari.hardware.servo    for PanTilt control (single source of truth)
  - ari.ipc.fifo          for FIFO command handling

Commands:
  pan_left [amount]      Pan camera left (default 200us)
  pan_right [amount]     Pan camera right
  tilt_up [amount]       Tilt camera up
  tilt_down [amount]     Tilt camera down
  home                   Go to home position
  set <pan_us> <tilt_us> Set absolute position
  capture [filename]     Capture image (default /tmp/cam_live.jpg)
  position               Print current position
  quit                   Exit daemon
"""

import sys

from ari.config import cfg
from ari.hardware.servo import PanTilt
from ari.ipc.fifo import FifoServer


def _pos_str(pt: PanTilt) -> str:
    """Format current position as a compact string."""
    pos = pt.position
    return f"pan={pos['pan_us']} tilt={pos['tilt_us']}"


def handle_command(line: str, pt: PanTilt, srv: FifoServer) -> bool:
    """Parse and execute a single command.

    Returns False if the daemon should exit, True otherwise.
    """
    parts = line.strip().split()
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    try:
        if cmd == "pan_left":
            amt = int(args[0]) if args else 200
            pt.pan_left(amt)
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "pan_right":
            amt = int(args[0]) if args else 200
            pt.pan_right(amt)
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "tilt_up":
            amt = int(args[0]) if args else 200
            pt.tilt_up(amt)
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "tilt_down":
            amt = int(args[0]) if args else 200
            pt.tilt_down(amt)
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "home":
            pt.home()
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "set":
            pan = int(args[0])
            tilt = int(args[1])
            pt.set_position(pan, tilt)
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "capture":
            fname = args[0] if args else "/tmp/cam_live.jpg"
            pt.capture(fname)
            srv.reply(f"captured {fname} {_pos_str(pt)}")

        elif cmd == "position":
            srv.reply(f"ok {_pos_str(pt)}")

        elif cmd == "quit":
            srv.reply("bye")
            print("Shutting down", flush=True)
            return False

        else:
            srv.reply(f"error unknown command: {cmd}")

        print(f"  [{cmd}] -> {_pos_str(pt)}", flush=True)

    except Exception as e:
        srv.reply(f"error {e}")
        print(f"  [{cmd}] ERROR: {e}", flush=True)

    return True


def main() -> None:
    fifo_path = cfg["ipc"]["camera_fifo"]

    print(f"Camera daemon starting...", flush=True)
    print(f"  Command pipe: {fifo_path}", flush=True)

    pt = PanTilt()
    srv = FifoServer(fifo_path)

    print(f"Ready at home position ({_pos_str(pt)})", flush=True)
    srv.reply(f"ready {_pos_str(pt)}")

    try:
        for line in srv:
            if not handle_command(line, pt, srv):
                break
    except KeyboardInterrupt:
        print("\nShutting down", flush=True)
    finally:
        pt.close()
        srv.cleanup()


if __name__ == "__main__":
    main()
