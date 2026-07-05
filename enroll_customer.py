"""
enroll_customer.py
Capture a customer's face via webcam and store it against their record
so recognize_live.py can identify them later.

Run:  python enroll_customer.py
"""

import os
import cv2
import store
import face_utils
from db import init_db
from face_box_overlay import LiveFaceBoxDetector

NUM_PHOTOS = 3           # how many photos to average into one encoding
TEMP_PHOTO_PATH = "_enroll_temp.jpg"
WARMUP_FRAMES = 20        # let the webcam auto-adjust exposure before use


def capture_photo(window_title: str, box_detector: LiveFaceBoxDetector):
    """Open the webcam, show a live preview with a box around any
    detected face, and capture one frame when the user presses SPACE.
    Press ESC to cancel."""
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

    # Many webcams need a few frames to auto-adjust exposure/white
    # balance. Skipping straight to capture can give a dark, unusable
    # frame even though the live preview looks fine a second later.
    for _ in range(WARMUP_FRAMES):
        cam.read()

    # Force the window to a known position and keep it on top. OpenCV's
    # imshow window only receives SPACE/ESC key presses while it is the
    # ACTIVE/FOCUSED window — if it opens behind VS Code or the terminal
    # still has focus, key presses silently go nowhere. Making it
    # topmost and clearly positioned makes it obvious where to click.
    window_name = "Enroll Customer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print(f"[{window_title}] >>> CLICK ON THE VIDEO WINDOW FIRST <<<")
    print(f"[{window_title}] Then press SPACE to capture, ESC to cancel.")
    frame = None
    while True:
        ok, preview = cam.read()
        if not ok:
            raise RuntimeError("Failed to read from webcam.")

        preview = box_detector.update_and_draw(preview)

        cv2.putText(preview, window_title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(preview, "CLICK THIS WINDOW, then SPACE=capture ESC=cancel", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            frame = preview
            break
        elif key == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    return frame


def enroll(customer_id: int):
    customer = store.get_customer(customer_id)
    if not customer:
        print(f"No customer with id {customer_id}.")
        return

    print(f"Enrolling face for: {customer['name']}")
    print(f"We'll capture {NUM_PHOTOS} photos — move your head slightly "
          f"between each one (straight, slight left, slight right).")

    box_detector = LiveFaceBoxDetector()
    embeddings = []
    attempt = 0
    while len(embeddings) < NUM_PHOTOS and attempt < NUM_PHOTOS + 3:
        attempt += 1
        frame = capture_photo(f"Photo {len(embeddings) + 1} of {NUM_PHOTOS}", box_detector)
        if frame is None:
            print("Cancelled.")
            return

        cv2.imwrite(TEMP_PHOTO_PATH, frame)
        try:
            embedding = face_utils.get_embedding(TEMP_PHOTO_PATH)
        except ValueError as e:
            print(f"  Problem with that photo: {e}. Let's retry this shot.")
            continue

        if embedding is None:
            print("  No face detected in that photo — let's retry this shot.")
            continue

        embeddings.append(embedding)
        print(f"  Photo {len(embeddings)} captured OK.")

    if os.path.exists(TEMP_PHOTO_PATH):
        os.remove(TEMP_PHOTO_PATH)

    if len(embeddings) < NUM_PHOTOS:
        print("Didn't get enough usable photos after several tries. "
              "Check lighting and that your face is centered, then run "
              "enrollment again.")
        return

    final_encoding = face_utils.average_embeddings(embeddings)

    # Check this face isn't already registered to a DIFFERENT customer.
    # Re-enrolling the same customer (e.g. re-doing their photos) is
    # fine and expected to match themselves — only block if it matches
    # someone else.
    other_enrolled = [c for c in store.get_enrolled_customers() if c["id"] != customer_id]
    match, distance = face_utils.find_best_match(final_encoding, other_enrolled)
    if match is not None:
        print(f"\nThis face is already registered to '{match['name']}' "
              f"(customer id {match['id']}, distance={distance:.3f}).")
        print("Enrollment cancelled — each face can only be linked to one "
              "customer record.")
        return

    store.save_face_encoding(customer_id, face_utils.encoding_to_bytes(final_encoding))
    print(f"Done — {customer['name']} is now enrolled for face recognition.")


def main():
    init_db()
    print("--- Enroll a customer's face ---")
    print("(Customer must already exist — add them via cli.py option 1 first)\n")

    customers = store.list_all_customers()
    if not customers:
        print("No customers exist yet. Run cli.py and add one first.")
        return

    for c in customers:
        print(f"  [{c['id']}] {c['name']}")

    try:
        customer_id = int(input("\nEnter customer id to enroll: ").strip())
    except ValueError:
        print("That's not a valid id.")
        return

    enroll(customer_id)


if __name__ == "__main__":
    main()