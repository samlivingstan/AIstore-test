"""
face_utils.py
Everything face-recognition-related lives here, separate from the camera
UI (enroll_customer.py, recognize_live.py) and separate from the database
(store.py). This keeps the AI piece swappable — if you ever want to try
a different model or library, this is the only file that should need to
change.

Model: DeepFace with the 'Facenet' model (128-number embedding per face).
We chose DeepFace over the more commonly-tutorialed `face_recognition`
library because `face_recognition` depends on `dlib`, which has to
compile from C++ source and is a common source of install failures,
especially on Windows. DeepFace uses TensorFlow, which ships as a
pre-built wheel — pip install just works.

How matching works:
Each face becomes a 128-number vector (an "embedding"). Two photos of
the same person produce vectors that are close together; different
people produce vectors that are far apart. We measure "close" using
cosine distance, and anything under FACE_MATCH_THRESHOLD counts as a
match. This threshold is a real tuning knob — see the note below.
"""

import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet"

# Which face detector DeepFace uses before generating the embedding.
# DeepFace's own default ('opencv') relies on a bundled Haar-cascade
# file that is sometimes missing from certain opencv-python installs,
# causing "no face detected" even on a perfectly good photo. 'mtcnn' is
# more reliable and doesn't have that dependency issue.
DETECTOR_BACKEND = "mtcnn"

# Cosine distance threshold below which two faces are considered the
# same person. Lower = stricter (fewer false matches, more missed
# matches). Higher = looser (more false matches, fewer missed matches).
# 0.40 is DeepFace's own published default for Facenet + cosine, and is
# a reasonable starting point. Expect to tune this once you test with
# real store lighting and real customer faces.
FACE_MATCH_THRESHOLD = 0.40


def get_embedding(image_path: str):
    """Given a path to a photo, return a 128-number embedding for the
    face in it, or None if no face was found.

    Raises ValueError if more than one face is detected — during
    enrollment and recognition we want exactly one person in frame.
    """
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,   # raises if no face found
        )
    except ValueError:
        # DeepFace raises ValueError when it can't detect any face
        return None

    if len(results) > 1:
        raise ValueError(
            f"Found {len(results)} faces in the image — only one person "
            "should be in frame."
        )

    return np.array(results[0]["embedding"], dtype=np.float32)


def average_embeddings(embeddings: list) -> np.ndarray:
    """Enrollment captures several photos of the same person (different
    angles/expressions). Averaging their embeddings gives a more robust
    reference than relying on a single photo."""
    stacked = np.stack(embeddings)
    return np.mean(stacked, axis=0).astype(np.float32)


def encoding_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert a numpy embedding to raw bytes for storing in the MySQL
    BLOB column."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_encoding(data: bytes) -> np.ndarray:
    """Reverse of encoding_to_bytes — read a BLOB back out as a numpy
    array."""
    return np.frombuffer(data, dtype=np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = identical direction (same face), 1 = unrelated, 2 = opposite.
    In practice same-person scores land well under 0.40, different-person
    scores land well above it."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(1 - np.dot(a_norm, b_norm))


def find_best_match(live_embedding: np.ndarray, enrolled_customers: list):
    """Compare one live face against every enrolled customer.

    enrolled_customers: list of dicts with at least 'id', 'name', and
    'face_encoding' (raw bytes from the DB) — i.e. exactly what
    store.get_enrolled_customers() returns.

    Returns (customer_dict, distance) for the closest match if it's
    under FACE_MATCH_THRESHOLD, otherwise (None, best_distance_found)
    so the caller can still see how close the nearest miss was.
    """
    best_customer = None
    best_distance = float("inf")

    for customer in enrolled_customers:
        stored_embedding = bytes_to_encoding(customer["face_encoding"])
        distance = cosine_distance(live_embedding, stored_embedding)
        if distance < best_distance:
            best_distance = distance
            best_customer = customer

    if best_customer is not None and best_distance <= FACE_MATCH_THRESHOLD:
        return best_customer, best_distance
    return None, best_distance