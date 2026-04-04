#!/usr/bin/env python3
"""
Camera control module for Ari — integrates with camera daemon.
Provides scanning, person detection, and tracking via Claude Vision.
"""

import os
import time
import base64
import subprocess

CAMERA_FIFO = "/tmp/ari_camera_cmd"
CAMERA_STATUS = "/tmp/ari_camera_status"
CLAUDE_CLI = os.path.expanduser("~/.claude/remote/ccd-cli/2.1.87")

# Scan positions (pan_us values) — covers ~180° field of view
SCAN_POSITIONS = [
    (800, 2200),   # far right
    (1200, 2200),  # right
    (1600, 2200),  # center (home)
    (2000, 2200),  # left
    (2400, 2200),  # far left
]


def send_camera_cmd(cmd, retries=3):
    """Send a command to the camera daemon with retry."""
    if not os.path.exists(CAMERA_FIFO):
        print("  ⚠️ Camera FIFO not found", flush=True)
        return None
    for attempt in range(retries):
        try:
            with open(CAMERA_FIFO, "w") as f:
                f.write(cmd + "\n")
            time.sleep(0.5)
            with open(CAMERA_STATUS, "r") as f:
                result = f.read().strip()
            if result and "error" not in result.lower():
                return result
        except Exception as e:
            print(f"  Camera cmd error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(0.5)
    return None


def move_camera(pan_us, tilt_us):
    """Move camera to absolute position and wait."""
    result = send_camera_cmd(f"set {pan_us} {tilt_us}")
    time.sleep(1)  # extra wait for servo to physically move
    return result


def capture(filename="/tmp/ari_vision.jpg"):
    """Capture image via camera daemon."""
    send_camera_cmd(f"capture {filename}")
    time.sleep(2)  # wait for capture
    # Resize for API if too large
    try:
        file_size = os.path.getsize(filename)
        if file_size > 100000:
            subprocess.run(
                ["ffmpeg", "-y", "-i", filename, "-vf", "scale=480:-1",
                 "-q:v", "8", filename],
                capture_output=True, timeout=10
            )
    except Exception:
        pass
    return filename


def image_to_base64(filepath):
    """Read image file and return base64 encoded string. Resize if too large."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        # If image is over 100KB, resize it down
        if len(data) > 100000:
            subprocess.run(
                ["ffmpeg", "-y", "-i", filepath, "-vf", "scale=480:-1",
                 "-q:v", "8", "/tmp/ari_vision_small.jpg"],
                capture_output=True, timeout=10
            )
            with open("/tmp/ari_vision_small.jpg", "rb") as f:
                data = f.read()
        return base64.standard_b64encode(data).decode("utf-8")
    except:
        return None


def ask_claude_vision(prompt, image_b64):
    """Ask Claude with an image via CLI."""
    try:
        result = subprocess.run(
            [CLAUDE_CLI, "-p", "--bare", "--model", "haiku", "--tools", "",
             "--system-prompt",
             "You are analyzing a camera image for person detection. "
             "Reply with ONLY 'YES' or 'NO' on the first line. "
             "If YES, on the second line say where in frame: LEFT, CENTER, or RIGHT. "
             "On the third line, briefly describe the person. Nothing else."],
            input=f"[Image as base64 JPEG: data:image/jpeg;base64,{image_b64}]\n\n{prompt}",
            capture_output=True, text=True, timeout=20,
            env=os.environ
        )
        return result.stdout.strip()
    except Exception as e:
        return "NO"


def capture_and_describe(prompt="What do you see?"):
    """Capture image at current position and get Claude's description."""
    filepath = capture()
    b64 = image_to_base64(filepath)
    if not b64:
        return None, "Couldn't capture an image."

    try:
        result = subprocess.run(
            [CLAUDE_CLI, "-p", "--bare", "--model", "haiku", "--tools", "",
             "--system-prompt",
             "You are Ari, a friendly robot. Describe what you see in the image "
             "conversationally in 1-2 sentences. No emojis or markdown."],
            input=f"[Image as base64 JPEG: data:image/jpeg;base64,{b64}]\n\n{prompt}",
            capture_output=True, text=True, timeout=20,
            env=os.environ
        )
        return filepath, result.stdout.strip()
    except Exception as e:
        return filepath, f"I tried to look but something went wrong."


def find_person():
    """Scan around to find a person. Returns (pan_us, tilt_us, description) or None."""
    print("  📷 Scanning for person...", flush=True)

    for pan_us, tilt_us in SCAN_POSITIONS:
        # Move camera and wait for it to settle
        print(f"  📷 Moving to pan={pan_us}...", flush=True)
        send_camera_cmd(f"set {pan_us} {tilt_us}")
        time.sleep(3)  # wait for servo to move AND settle

        # Capture via camera daemon
        filepath = f"/tmp/ari_scan_{pan_us}.jpg"
        send_camera_cmd(f"capture {filepath}")
        time.sleep(2.5)  # wait for capture to complete

        b64 = image_to_base64(filepath)
        if not b64:
            print(f"  ⚠️ No image at pan={pan_us}", flush=True)
            continue

        print(f"  📷 Checking position pan={pan_us}...", flush=True)

        response = ask_claude_vision(
            "Is there a person or human visible in this image?",
            b64
        )

        print(f"  📷 Response: {response[:80]}", flush=True)

        # Parse simple YES/NO response
        lines = response.strip().split("\n")
        first_line = lines[0].strip().upper() if lines else ""

        if first_line.startswith("YES"):
            print(f"  ✅ Person found at pan={pan_us}!", flush=True)

            # Fine-tune position based on where in frame
            position = lines[1].strip().upper() if len(lines) > 1 else "CENTER"
            description = lines[2].strip() if len(lines) > 2 else "I found someone!"

            if "LEFT" in position:
                pan_us += 150
            elif "RIGHT" in position:
                pan_us -= 150

            send_camera_cmd(f"set {pan_us} {tilt_us}")
            time.sleep(1)

            return pan_us, tilt_us, description

    # Not found — return to home
    print("  ❌ No person found, returning home", flush=True)
    send_camera_cmd("home")
    time.sleep(1)
    return None


def look_direction(direction):
    """Move camera in a direction."""
    direction = direction.lower().strip()
    cmd_map = {
        "left": "pan_left 300",
        "right": "pan_right 300",
        "up": "tilt_up 200",
        "down": "tilt_down 200",
        "home": "home",
        "center": "home",
    }
    cmd = cmd_map.get(direction)
    if cmd:
        send_camera_cmd(cmd)
        return True
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "find":
            result = find_person()
            if result:
                print(f"Found: {result}")
            else:
                print("No person found")
        elif cmd == "describe":
            _, desc = capture_and_describe()
            print(f"Description: {desc}")
        elif cmd in ("left", "right", "up", "down", "home"):
            look_direction(cmd)
            print(f"Moved {cmd}")
    else:
        print("Usage: camera_control.py [find|describe|left|right|up|down|home]")
