"""
Milestone 5: Track ID → person crop → Staff / Beneficiary / Unknown.

This is only a classification *interface*. Every person is Unknown until a
real trained model is plugged into classify_person().
"""

import argparse
import os
import sys

import cv2

_INGESTION_DIR = os.path.join(os.path.dirname(__file__), "..", "video_ingestion")
_DETECTION_DIR = os.path.join(os.path.dirname(__file__), "..", "person_detection")
_TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "person_tracking")
sys.path.insert(0, os.path.abspath(_INGESTION_DIR))
sys.path.insert(0, os.path.abspath(_DETECTION_DIR))
sys.path.insert(0, os.path.abspath(_TRACKING_DIR))

from person_detector import MODEL_FILENAME, MODEL_PATH, load_model  # noqa: E402
from person_tracker import TRACKER_CONFIG, TRACKER_NAME, track_people  # noqa: E402
from video_capture import frame_delay_ms, open_source  # noqa: E402

STAFF = "Staff"
BENEFICIARY = "Beneficiary"
UNKNOWN = "Unknown"
LABELS = (STAFF, BENEFICIARY, UNKNOWN)

DEFAULT_CONFIDENCE = 0.5
WINDOW_TITLE = "Sentinal AI - Person Classification"
LOG_EVERY_N_FRAMES = 30
BOX_COLOR = (180, 80, 200)
TEXT_COLOR = (255, 255, 255)


def crop_person(frame, x1, y1, x2, y2):
    """
    A person crop is the pixels inside the tracking box: one image of one
    person, cut out of the full frame. Classification looks at this crop,
    not at the whole camera view.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def classify_person(person_crop):
    """
    Baseline classifier. Always returns Unknown.

    Detection (YOLO) only says "this is a person." It cannot say Staff vs
    Beneficiary. A future trained model should replace the body of this
    function, for example:

        return trained_model.predict(person_crop)

    until then we do not guess from clothing, color, or random rules.
    """
    if person_crop is None or person_crop.size == 0:
        return UNKNOWN
    # TODO: connect a trained Staff / Beneficiary classifier here.
    return UNKNOWN


def draw_classified(frame, tracks, labels):
    for (x1, y1, x2, y2, track_id, _conf), label in zip(tracks, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(
            frame,
            f"Person ID {track_id}",
            (x1, max(18, y1 - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (x1, max(36, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (8, 8), (240, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        f"Active persons: {len(tracks)}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def run(cap, model, confidence, fps):
    delay_ms = frame_delay_ms(fps)
    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if frame_number == 0:
                    print("Error: a frame could not be read from the source.")
                    sys.exit(1)
                print(f"Video ended after {frame_number} frame(s).")
                break

            frame_number += 1
            tracks = track_people(model, frame, confidence)
            labels = []
            for x1, y1, x2, y2, _tid, _conf in tracks:
                crop = crop_person(frame, x1, y1, x2, y2)
                labels.append(classify_person(crop))

            cv2.imshow(WINDOW_TITLE, draw_classified(frame, tracks, labels))
            if frame_number == 1 or frame_number % LOG_EVERY_N_FRAMES == 0:
                print(f"Frame {frame_number} | active persons {len(tracks)}")

            if (cv2.waitKey(delay_ms) & 0xFF) in (ord("q"), ord("Q")):
                print(f"Quit requested after {frame_number} frame(s).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Staff / Beneficiary / Unknown interface (always Unknown)."
    )
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        print("Error: --confidence must be between 0 and 1.")
        sys.exit(1)
    if args.source is not None and not os.path.isfile(args.source):
        print(f"Error: video file does not exist: {args.source}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    cap, source_type, source_label = open_source(args.source)
    print("Person classification (baseline) started.")
    print(f"  Model    : {MODEL_FILENAME}")
    print(f"  Tracker  : {TRACKER_NAME} ({TRACKER_CONFIG})")
    print(f"  Source   : {source_type} / {source_label}")
    print(f"  Labels   : {', '.join(LABELS)} (current result: {UNKNOWN})")
    print("Press Q to quit.\n")
    run(cap, model, args.confidence, cap.get(cv2.CAP_PROP_FPS))


if __name__ == "__main__":
    main()
