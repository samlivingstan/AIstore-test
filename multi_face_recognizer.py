"""
multi_face_recognizer.py
Continuously detects EVERY face currently in the camera frame and shows
each person's name + pending balance as a labeled box, all at once —
unlike recognize_live.py, which handles one person per SPACE press.

Performance design:
Running the embedding model (the slow part) on every single video frame
for every face would make the preview freeze. Instead, recognition runs
periodically (every RECOGNITION_EVERY_N_FRAMES frames) and the results
are cached and redrawn on every frame in between, so the video stays
smooth with a brief refresh every second or two rather than a constant
stutter.

Expect a small freeze each time recognition runs — that's the model
doing real work, not a bug. If it feels too laggy on your machine,
increase RECOGNITION_EVERY_N_FRAMES below.
"""

import cv2
from mtcnn import MTCNN

import store
import face_utils

# How often (in frames) to re-run full recognition on all faces.
# Lower = more up-to-date labels but more frequent freezes.
# Higher = smoother video but labels update less often.
RECOGNITION_EVERY_N_FRAMES = 45

# MTCNN sometimes reports low-confidence false-positive boxes (e.g. on
# textured backgrounds). Ignore anything below this.
MIN_DETECTION_CONFIDENCE = 0.90

# Extra margin added around each detected face box before cropping for
# recognition — a too-tight crop can make the embedding step's own
# internal face detection fail on what should be an easy face.
CROP_PADDING_RATIO = 0.25

TEMP_CROP_PATH = "_multi_face_crop_temp.jpg"

COLOR_DUE = (0, 0, 255)        # red — has a pending balance
COLOR_NO_DUE = (0, 200, 0)     # green — no dues
COLOR_UNKNOWN = (0, 200, 255)  # orange — face detected but not enrolled
COLOR_UNCLEAR = (150, 150, 150)  # gray — box found but couldn't get a clean embedding


class MultiFaceRecognizer:
    def __init__(self):
        self._detector = MTCNN()
        self._frame_count = 0
        # cached results, redrawn every frame: list of dicts with
        # 'box' (x,y,w,h), 'label' (str), 'color' (BGR tuple)
        self._cached_results = []

    def update_and_draw(self, frame_bgr):
        self._frame_count += 1

        if self._frame_count % RECOGNITION_EVERY_N_FRAMES == 0:
            self._cached_results = self._recognize_all_faces(frame_bgr)

        for r in self._cached_results:
            x, y, w, h = r["box"]
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), r["color"], 2)
            # label background for readability
            label = r["label"]
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(0, y - 10)
            cv2.rectangle(frame_bgr, (x, label_y - text_h - 6), (x + text_w + 6, label_y + 4), r["color"], -1)
            cv2.putText(frame_bgr, label, (x + 3, label_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame_bgr

    def _recognize_all_faces(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            detections = self._detector.detect_faces(rgb)
        except Exception:
            detections = []

        enrolled = store.get_enrolled_customers()  # fresh each cycle, catches new enrollments
        frame_h, frame_w = frame_bgr.shape[:2]

        results = []
        for det in detections:
            if det.get("confidence", 0) < MIN_DETECTION_CONFIDENCE:
                continue

            x, y, w, h = det["box"]
            x, y = max(0, x), max(0, y)

            # pad the crop so the embedding step's own internal
            # detection has enough context to work with
            pad_w, pad_h = int(w * CROP_PADDING_RATIO), int(h * CROP_PADDING_RATIO)
            x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
            x1, y1 = min(frame_w, x + w + pad_w), min(frame_h, y + h + pad_h)
            crop = frame_bgr[y0:y1, x0:x1]

            if crop.size == 0:
                continue

            cv2.imwrite(TEMP_CROP_PATH, crop)
            try:
                embedding = face_utils.get_embedding(TEMP_CROP_PATH)
            except ValueError:
                embedding = None

            if embedding is None:
                results.append({"box": (x, y, w, h), "label": "?", "color": COLOR_UNCLEAR})
                continue

            match, distance = face_utils.find_best_match(embedding, enrolled)

            if match is None:
                results.append({"box": (x, y, w, h), "label": "Unknown", "color": COLOR_UNKNOWN})
            else:
                balance = store.get_balance(match["id"])
                if balance > 0:
                    label = f"{match['name']}: Rs.{balance:.0f} due"
                    color = COLOR_DUE
                else:
                    label = f"{match['name']}: no due"
                    color = COLOR_NO_DUE
                results.append({"box": (x, y, w, h), "label": label, "color": color})

        return results