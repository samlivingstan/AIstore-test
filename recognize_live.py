"""
recognize_live.py
Open the webcam, and whenever a face is captured (press SPACE), match it
against enrolled customers and show their name + pending balance.

Capture-on-keypress rather than continuous recognition every frame,
since running a deep learning model on every frame is slow and
unnecessary for a checkout-counter use case.

Run:  python recognize_live.py
"""

import os
import cv2
import store
import face_utils
from db import init_db
from face_box_overlay import LiveFaceBoxDetector

TEMP_PHOTO_PATH = "_recognize_temp.jpg"
WARMUP_FRAMES = 20


def main():
    init_db()

    enrolled = store.get_enrolled_customers()
    if not enrolled:
        print("No customers are enrolled yet. Run enroll_customer.py first.")
        return

    print(f"{len(enrolled)} customer(s) enrolled.")
    print("Press SPACE to capture and identify a face. Press ESC to quit.\n")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check it's connected and not in use "
            "by another application."
        )

    # Request a larger capture resolution. If the webcam doesn't support
    # exactly this size it'll pick its closest match — either way this
    # gives a bigger, sharper feed than the default (often 640x480).
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(WARMUP_FRAMES):
        cam.read()

    window_name = "Recognize Customer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print(">>> CLICK ON THE VIDEO WINDOW FIRST <<<")
    last_result_text = ""
    box_detector = LiveFaceBoxDetector()

    while True:
        ok, frame = cam.read()
        if not ok:
            raise RuntimeError("Failed to read from webcam.")

        display = frame.copy()
        display = box_detector.update_and_draw(display)

        cv2.putText(display, "CLICK THIS WINDOW: SPACE=identify ESC=quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if last_result_text:
            cv2.putText(display, last_result_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            cv2.imwrite(TEMP_PHOTO_PATH, frame)
            try:
                embedding = face_utils.get_embedding(TEMP_PHOTO_PATH)
            except ValueError as e:
                last_result_text = f"Error: {e}"
                print(last_result_text)
                continue

            if embedding is None:
                last_result_text = "No face detected — try again."
                print(last_result_text)
                continue

            match, distance = face_utils.find_best_match(embedding, enrolled)

            if match is None:
                last_result_text = f"Not recognized (closest distance {distance:.3f})"
                print(last_result_text)
            else:
                balance = store.get_balance(match["id"])
                if balance > 0:
                    last_result_text = f"{match['name']} - PENDING DUE: Rs.{balance:.2f}"
                else:
                    last_result_text = f"{match['name']} - no dues"
                print(f"Matched: {match['name']} (distance={distance:.3f}) | Balance: Rs.{balance:.2f}")

    if os.path.exists(TEMP_PHOTO_PATH):
        os.remove(TEMP_PHOTO_PATH)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()