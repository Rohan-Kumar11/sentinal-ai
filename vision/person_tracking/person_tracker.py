"""
Milestone 3: OpenCV → YOLO → ByteTrack → tracked persons.

YOLO still finds people in each frame. ByteTrack then tries to give the
same moving person the same temporary ID in the next frames.

A track ID is not a name, not a face ID, and not attendance.
"""

import argparse
import os
import sys
import time

import cv2

PERSON_CLASS_ID = 0


# Ultralytics ships this config. persist=True keeps tracker memory between frames.
TRACKER_NAME = "ByteTrack"
TRACKER_CONFIG = "bytetrack.yaml"

DEFAULT_CONFIDENCE = 0.5
WINDOW_TITLE = "Sentinal AI - Person Tracking"
LOG_EVERY_N_FRAMES = 30

BOX_COLOR = (0, 180, 255)  # BGR orange so this window looks different from detection
TEXT_COLOR = (255, 255, 255)
COUNTER_BG = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track people across frames using YOLO + ByteTrack."
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


def tracks_from_result(result):
    """
    Read ByteTrack output for one frame.

    model.track() still runs YOLO first. ByteTrack then matches this frame's
    person boxes to boxes from recent frames. When a match is found, the same
    integer ID is reused. That ID only lasts for this video/session.

    box.id is missing until the tracker has assigned an ID.
    """
    tracks = []
    if result.boxes is None:
        return tracks

    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id != PERSON_CLASS_ID:
            continue
        if box.id is None:
            continue

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        track_id = int(box.id[0])
        confidence = float(box.conf[0])
        tracks.append((x1, y1, x2, y2, track_id, confidence))

    return tracks


def draw_tracks(frame, tracks):
    """Draw each active track: box + 'Person ID 3' (temporary ID, not a real identity)."""
    for x1, y1, x2, y2, track_id, confidence in tracks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"Person ID {track_id} ({confidence:.2f})"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )

    counter_text = f"Tracked persons: {len(tracks)}"
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


def track_people(model, frame, confidence):
    """
    One tracking step.

    persist=True is what lets ByteTrack remember tracks from the previous
    frame. Without it, IDs would reset every image (that would be detection
    again, not tracking).

    classes=[PERSON_CLASS_ID] keeps cars, chairs, dogs, and bottles out.
    """
    try:
        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
    except Exception as exc:
        print(f"Error: tracking failed unexpectedly: {exc}")
        sys.exit(1)

    return tracks_from_result(results[0])


def run_tracking(cap, model, confidence, fps):
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
            tracks = track_people(model, frame, confidence)
            annotated = draw_tracks(frame, tracks)
            cv2.imshow(WINDOW_TITLE, annotated)

            elapsed = time.perf_counter() - started_at
            if frame_number == 1 or frame_number % LOG_EVERY_N_FRAMES == 0:
                print(
                    f"Frame {frame_number} | active tracks {len(tracks)} | elapsed {elapsed:.1f}s"
                )

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), ord("Q")):
                print(f"Quit requested after {frame_number} frame(s).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        print("Error: --confidence must be between 0 and 1.")
        sys.exit(1)

    if args.source is not None and not os.path.isfile(args.source):
        print(f"Error: video file does not exist: {args.source}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    cap, source_type, source_label = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Person tracking started.")
    print(f"  Model                 : {MODEL_FILENAME}")
    print(f"  Tracker               : {TRACKER_NAME} ({TRACKER_CONFIG})")
    print(f"  Source type           : {source_type}")
    print(f"  Source                : {source_label}")
    print(f"  Confidence threshold  : {args.confidence}")
    print("Press Q in the video window to quit.\n")

    run_tracking(cap, model, args.confidence, fps)


if __name__ == "__main__":
    main()
