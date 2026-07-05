"""
recognize_multi_live.py
Opens the webcam and continuously identifies EVERY person currently in
frame, showing each one's name and pending balance as a labeled box —
no need to press SPACE per person. Good for a counter where more than
one customer might be in view at once.

Colors:
  red    = enrolled customer with a pending balance
  green  = enrolled customer, no dues
  orange = a face was detected but doesn't match any enrolled customer
  gray   = a box was found but a clean face reading wasn't possible

Labels refresh every second or two (not every frame) — see
multi_face_recognizer.py if you want to tune that speed/smoothness
tradeoff.

Run:  python recognize_multi_live.py
Press ESC to quit.
"""

import os
import cv2
import store
from db import init_db
from multi_face_recognizer import MultiFaceRecognizer, TEMP_CROP_PATH

WARMUP_FRAMES = 20


def main():
    init_db()

    enrolled = store.get_enrolled_customers()
    if not enrolled:
        print("No customers are enrolled yet. Run enroll_customer.py first.")
        return

    print(f"{len(enrolled)} customer(s) enrolled.")
    print("Opening camera — recognition refreshes every ~1-2 seconds.")
    print("Press ESC to quit.\n")

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

    window_name = "Recognize Multiple Customers"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    recognizer = MultiFaceRecognizer()

    while True:
        ok, frame = cam.read()
        if not ok:
            raise RuntimeError("Failed to read from webcam.")

        frame = recognizer.update_and_draw(frame)
        cv2.putText(frame, "ESC = quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    if os.path.exists(TEMP_CROP_PATH):
        os.remove(TEMP_CROP_PATH)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()