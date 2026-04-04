"""Camera helper functions for Ari.

Handles image capture (via IPC to the camera daemon), Base64 encoding,
and resizing of images that exceed the API size limit.

Usage::

    from ari.vision.camera import image_to_base64, capture_and_resize
    b64 = image_to_base64("/tmp/snapshot.jpg")
    path = capture_and_resize("/tmp/ari_vision.jpg")
"""

from __future__ import annotations

import base64
import os
import subprocess
import time

from ari.config import cfg


def image_to_base64(filepath: str) -> str | None:
    """Read an image file and return its Base64-encoded content.

    If the file exceeds ``cfg['camera']['image_max_bytes']`` it is resized
    to ``cfg['camera']['image_resize_width']`` pixels wide via ``ffmpeg``
    before encoding.

    Returns ``None`` if the file cannot be read.
    """
    max_bytes: int = cfg["camera"]["image_max_bytes"]
    resize_width: int = cfg["camera"]["image_resize_width"]

    try:
        with open(filepath, "rb") as fh:
            data = fh.read()
    except OSError:
        return None

    if len(data) > max_bytes:
        resized_path = filepath + ".small.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", filepath,
                    "-vf", f"scale={resize_width}:-1",
                    "-q:v", "8",
                    resized_path,
                ],
                capture_output=True,
                timeout=10,
            )
            with open(resized_path, "rb") as fh:
                data = fh.read()
        except (subprocess.TimeoutExpired, OSError):
            # Fall back to the original (possibly large) data
            pass

    return base64.standard_b64encode(data).decode("utf-8")


def capture_and_resize(
    filename: str = "/tmp/ari_vision.jpg",
    camera_fifo: str | None = None,
    camera_status: str | None = None,
) -> str | None:
    """Capture an image via IPC to the camera daemon, resize if needed.

    Sends a ``capture`` command to the camera daemon FIFO, waits for the
    image to be written, and shrinks it if it exceeds the API size limit.

    Parameters
    ----------
    filename:
        Path where the daemon should write the JPEG.
    camera_fifo:
        Path to the camera daemon command FIFO.  Defaults to
        ``cfg['ipc']['camera_fifo']``.
    camera_status:
        Path to the camera daemon status file.  Defaults to
        ``cfg['ipc']['camera_status']``.

    Returns
    -------
    str or None
        The path to the (possibly resized) image, or ``None`` on failure.
    """
    fifo = camera_fifo or cfg["ipc"]["camera_fifo"]
    status_path = camera_status or cfg["ipc"]["camera_status"]
    max_bytes: int = cfg["camera"]["image_max_bytes"]
    resize_width: int = cfg["camera"]["image_resize_width"]

    if not os.path.exists(fifo):
        return None

    # Ask the daemon to capture
    try:
        with open(fifo, "w") as fh:
            fh.write(f"capture {filename}\n")
    except OSError:
        return None

    # Wait for the daemon to write the image
    time.sleep(cfg["vision"].get("scan_capture_wait", 2.5))

    if not os.path.exists(filename):
        return None

    # Resize if too large for the API
    try:
        if os.path.getsize(filename) > max_bytes:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", filename,
                    "-vf", f"scale={resize_width}:-1",
                    "-q:v", "8",
                    filename,
                ],
                capture_output=True,
                timeout=10,
            )
    except (subprocess.TimeoutExpired, OSError):
        pass  # keep the original file as-is

    return filename
