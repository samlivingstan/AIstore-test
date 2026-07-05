"""
face_box_overlay.py
Draws a live bounding box around the detected face in the camera
preview, so you can see what the camera is picking up before you press
SPACE to capture.

This is a separate, lightweight detector from face_utils.py's
DeepFace.represent() call. Running the full embedding pipeline on every
single video frame would make the preview laggy — this just runs MTCNN's
detector (fast, box-only, no embedding) periodically and reuses the last
known box in between frames, which keeps the preview smooth.
"""

import cv2
from mtcnn import MTCNN

# Run real detection every N frames; reuse the last box on frames in
# between. Lower = more responsive box, higher = smoother/faster video.
DETECT_EVERY_N_FRAMES = 5

BOX_COLOR = (0, 255, 0)      # green
BOX_THICKNESS = 2


class LiveFaceBoxDetector:
    def __init__(self):
        self._detector = MTCNN()
        self._frame_count = 0
        self._last_box = None  # (x, y, w, h) or None

    def update_and_draw(self, frame_bgr):
        """Call once per video frame. Detects a face every few frames
        (for speed) and draws the most recent known box on every frame
        (for smoothness). Returns the frame with the box drawn on it."""
        self._frame_count += 1

        if self._frame_count % DETECT_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                detections = self._detector.detect_faces(rgb)
            except Exception:
                detections = []

            if detections:
                # if more than one face is in frame, use the largest —
                # almost always the person closest to the camera
                best = max(detections, key=lambda d: d["box"][2] * d["box"][3])
                self._last_box = tuple(best["box"])  # (x, y, w, h)
            else:
                self._last_box = None

        if self._last_box is not None:
            x, y, w, h = self._last_box
            # MTCNN boxes can occasionally report small negative x/y
            # near frame edges — clamp so cv2.rectangle doesn't error
            x, y = max(0, x), max(0, y)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

        return frame_bgr

    def has_face(self) -> bool:
        """Whether a face is currently being tracked in the box."""
        return self._last_box is not None