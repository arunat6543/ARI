"""Real-time object and person detection using OpenCV.

Uses Haar cascades for face detection (built-in, zero downloads).
Designed to run at 30+ FPS on Pi 5 with picamera2 continuous capture.

Future: add YOLOv8-nano ONNX for full 80-class object detection.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class names for future YOLO integration
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class Detection:
    """A single detection result."""

    def __init__(self, label: str, confidence: float, x: int, y: int, w: int, h: int,
                 frame_width: int, frame_height: int):
        self.label = label
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_width = frame_width
        self.frame_height = frame_height

    @property
    def center_x(self) -> int:
        return self.x + self.w // 2

    @property
    def center_y(self) -> int:
        return self.y + self.h // 2

    @property
    def position_in_frame(self) -> str:
        """Where in the frame: LEFT, CENTER, or RIGHT."""
        cx = self.center_x / self.frame_width
        if cx < 0.33:
            return "LEFT"
        elif cx > 0.66:
            return "RIGHT"
        return "CENTER"

    @property
    def area(self) -> int:
        return self.w * self.h

    def __repr__(self) -> str:
        return (f"Detection({self.label}, {self.confidence:.0%}, "
                f"pos={self.position_in_frame}, "
                f"bbox=({self.x},{self.y},{self.w},{self.h}))")


class FaceDetector:
    """Fast face detection using OpenCV Haar cascades.

    Runs at ~30 FPS on Pi 5 at 640x480. No model download required.
    Limited: only works with frontal upright faces in good lighting.
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
        logger.info("Face detector loaded (Haar cascade)")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
        )

        return [
            Detection("person", 0.9, x, y, fw, fh, w, h)
            for (x, y, fw, fh) in faces
        ]


class YoloDetector:
    """Object detection using YOLOv8-nano.

    Detects 80 object types (person, chair, cup, book, etc.).
    Runs at ~3-5 FPS on Pi 5 CPU. Much more robust than Haar cascades —
    handles angles, low light, partial occlusion.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.25):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._confidence = confidence
        logger.info("YOLO detector loaded: %s (conf=%.0f%%)", model_path, confidence * 100)

    def detect(self, frame: np.ndarray, classes: list[str] | None = None) -> list[Detection]:
        """Detect objects in an RGB frame.

        Args:
            frame: numpy array (H, W, 3) — RGB image
            classes: optional filter — only return these class names (e.g., ["person"])

        Returns:
            List of Detection objects.
        """
        h, w = frame.shape[:2]
        # Convert RGB to BGR for YOLO
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self._model(bgr, verbose=False, conf=self._confidence)

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self._model.names[cls_id]

                if classes and name not in classes:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                det = Detection(
                    label=name,
                    confidence=conf,
                    x=x1, y=y1,
                    w=x2 - x1, h=y2 - y1,
                    frame_width=w, frame_height=h,
                )
                detections.append(det)

        return detections


class LiveScanner:
    """Continuous video capture + real-time detection for seamless scanning.

    Instead of stop-capture-analyze-move, this runs detection on every frame
    while the camera pans smoothly. When a target is found, it stops.

    Usage:
        scanner = LiveScanner(detector)
        scanner.start()

        # Pan camera while scanner runs
        result = scanner.wait_for_detection(timeout=10)

        scanner.stop()
    """

    def __init__(self, detector: FaceDetector, resolution: tuple = (640, 480)):
        self.detector = detector
        self.resolution = resolution
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: list[Detection] = []
        self._lock = threading.Lock()
        self._detection_event = threading.Event()
        self._cam = None

    def start(self):
        """Start continuous capture and detection."""
        if self._running:
            return

        from picamera2 import Picamera2

        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.5)

        self._running = True
        self._detection_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Live scanner started (%dx%d)", *self.resolution)

    def _capture_loop(self):
        """Continuously capture frames and run detection."""
        while self._running:
            try:
                frame = self._cam.capture_array()
                detections = self.detector.detect(frame)

                with self._lock:
                    self._latest_frame = frame
                    self._latest_detections = detections

                if detections:
                    self._detection_event.set()

            except Exception as e:
                logger.error("Capture error: %s", e)
                time.sleep(0.1)

    def wait_for_detection(self, timeout: float = 10.0,
                           label: str = "person") -> Optional[Detection]:
        """Block until a detection matching label is found, or timeout.

        Args:
            timeout: seconds to wait
            label: object label to look for (default: "person")

        Returns:
            Detection if found, None if timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self._detection_event.wait(timeout=0.1):
                with self._lock:
                    for det in self._latest_detections:
                        if det.label == label:
                            return det
                    self._detection_event.clear()
        return None

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def latest_detections(self) -> list[Detection]:
        with self._lock:
            return list(self._latest_detections)

    def capture_frame_as_jpeg(self, path: str) -> str:
        """Save the latest frame as JPEG."""
        frame = self.latest_frame
        if frame is not None:
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
        return path

    def stop(self):
        """Stop capture and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._cam:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        logger.info("Live scanner stopped")
