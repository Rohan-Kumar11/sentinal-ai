"""
Milestone 2: OpenCV → YOLO → person detection.

Reads frames with the Milestone 1 VideoCapture helper, runs a pretrained
YOLO model, and draws boxes only for the COCO "person" class.

This is detection only. It does not track people across frames, and it
does not count attendance.
"""

import argparse
import os
import sys
import time

import cv2
from ultralytics import YOLO

# Reuse webcam / file opening from Milestone 1 instead of copying it.
_INGESTION_DIR = os.path.join(os.path.dirname(__file__), "..", "video_ingestion")
sys.path.insert(0, os.path.abspath(_INGESTION_DIR))
from video_capture import frame_delay_ms, open_source  # noqa: E402

# Nano model: small, fast, good enough for a first prototype.
# Ultralytics downloads the pretrained COCO weights on first run.
MODEL_FILENAME = "yolo11n.pt"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

# In the COCO dataset used by this pretrained model, class 0 is "person".
PERSON_CLASS_ID = 0

DEFAULT_CONFIDENCE = 0.5
WINDOW_TITLE = "Sentinal AI - Person Detection"

BOX_COLOR = (0, 200, 0)  # BGR green
TEXT_COLOR = (255, 255, 255)
COUNTER_BG = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect people in a webcam or video file using pretrained YOLO."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to a local video file. If omitted, the default webcam is used.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Minimum detection confidence from 0 to 1 (default: {DEFAULT_CONFIDENCE}).",
    )
    return parser.parse_args()


def load_model(model_name):
    """
    Load a pretrained Ultralytics YOLO model.

    The first time this runs, Ultralytics may download the .pt weight file.
    No custom training happens here — we use the public COCO checkpoint.
    """
    try:
        model = YOLO(model_name)
    except Exception as exc:
        print(f"Error: YOLO model could not be loaded ({model_name}): {exc}")
        sys.exit(1)
    return model


def person_boxes_from_result(result):
    """
    Read YOLO's output for one frame and keep only person detections.

    Each box has:
      - xyxy: left, top, right, bottom pixel coordinates
      - conf: confidence (how sure the model is, 0 to 1)
      - cls: class id (0 = person in COCO)
    """
    people = []
    if result.boxes is None:
        return people

    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        confidence = float(box.conf[0])
        people.append((x1, y1, x2, y2, confidence))

    return people


def draw_people(frame, people):
    """Draw a bounding box and 'Person 0.91' label for each detection."""
    for x1, y1, x2, y2, confidence in people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label = f"Person {confidence:.2f}"
        label_origin = (x1, max(20, y1 - 8))
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

    # Frame-level count only: how many person boxes this frame has.
    # The same person in the next frame is counted again. That is not attendance.
    counter_text = f"Persons detected: {len(people)}"
    cv2.rectangle(frame, (8, 8), (280, 42), COUNTER_BG, thickness=-1)
    cv2.putText(
        frame,
        counter_text,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return frame


def detect_people(model, frame, confidence):
    """
    Run YOLO on one OpenCV BGR frame.

    classes=[PERSON_CLASS_ID] tells YOLO to skip cars, chairs, dogs, etc.
    verbose=False keeps the terminal quiet during the live loop.
    """
    results = model.predict(
        source=frame,
        conf=confidence,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )
    return person_boxes_from_result(results[0])


def run_detection(cap, model, confidence, fps):
    delay_ms = frame_delay_ms(fps)
    frame_number = 0
    started_at = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                if frame_number == 0:
                    print("Error: a frame could not be read from the source.")
                    sys.exit(1)
                elapsed = time.perf_counter() - started_at
                print(
                    f"\nVideo ended after {frame_number} frame(s) "
                    f"({elapsed:.2f}s elapsed)."
                )
                break

            frame_number += 1
            people = detect_people(model, frame, confidence)
            annotated = draw_people(frame, people)
            cv2.imshow(WINDOW_TITLE, annotated)

            elapsed = time.perf_counter() - started_at
            print(
                f"\rFrame {frame_number} | persons {len(people)} | elapsed {elapsed:.2f}s",
                end="",
                flush=True,
            )

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), ord("Q")):
                print(f"\nQuit requested after {frame_number} frame(s).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        print("Error: --confidence must be between 0 and 1.")
        sys.exit(1)

    # Fail fast on a bad path so we do not download/load YOLO first.
    if args.source is not None and not os.path.isfile(args.source):
        print(f"Error: video file does not exist: {args.source}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    cap, source_type, source_label = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Person detection started.")
    print(f"  Model                 : {MODEL_FILENAME}")
    print(f"  Source type           : {source_type}")
    print(f"  Source                : {source_label}")
    print(f"  Confidence threshold  : {args.confidence}")
    print("Press Q in the video window to quit.\n")

    run_detection(cap, model, args.confidence, fps)


if __name__ == "__main__":
    main()
