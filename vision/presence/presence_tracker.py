"""
Milestone 4: Track IDs → presence duration.

ByteTrack still gives each person a temporary ID. This module only answers:
"About how long has that ID been visible in this session?"

That is not attendance, not a name, and not staff vs beneficiary.
"""

import argparse
import os
import sys
import time

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

DEFAULT_CONFIDENCE = 0.5
WINDOW_TITLE = "Sentinal AI - Presence Duration"
LOG_EVERY_N_FRAMES = 30

BOX_COLOR = (80, 180, 80)
TEXT_COLOR = (255, 255, 255)
COUNTER_BG = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure how long each ByteTrack person ID stays visible."
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


def choose_timing_mode(source_type, fps):
    """
    Webcam: wall-clock. YOLO may be slower than the camera, but we still
    want "how long was this person in front of the camera in real life."

    Video file: media time from FPS. One processed frame is one video frame,
    not 1/30 of wall-clock. If inference is slow, duration still matches the
    clip (person in 90 frames at 30 FPS ≈ 3.0s of video, even if it took 20s
    to process).
    """
    if source_type == "video file" and fps and fps > 0:
        return "media", fps
    return "wall", fps


class PresenceBook:
    """
    Per-track timing for this session only.

    start_time (wall mode):
        perf_counter() when this ByteTrack ID is first seen.

    duration:
        wall mode  — add elapsed wall time only while the ID is visible;
                     freeze when it leaves the frame
        media mode — visible_frame_count / FPS (video seconds)

    A disappeared ID keeps its last duration in the book. If ByteTrack later
    gives a NEW id, that is a new row. We do not glue old and new IDs together
    (a track ID is not a real person's identity).
    """

    def __init__(self, timing_mode, fps):
        self.timing_mode = timing_mode
        self.fps = fps
        self.records = {}

    def update(self, active_ids, frame_number, wall_now):
        active_ids = set(active_ids)

        for track_id, rec in self.records.items():
            if rec["active"] and track_id not in active_ids:
                rec["active"] = False

        durations = {}
        for track_id in active_ids:
            is_new = track_id not in self.records
            returning = (not is_new) and (not self.records[track_id]["active"])

            if is_new:
                self.records[track_id] = {
                    "start_wall": wall_now,
                    "last_update_wall": wall_now,
                    "first_frame": frame_number,
                    "visible_frames": 0,
                    "duration": 0.0,
                    "active": True,
                }

            rec = self.records[track_id]
            if returning:
                # Same temporary ID came back. Resume the clock; do not add the gap.
                rec["last_update_wall"] = wall_now
            rec["active"] = True
            rec["visible_frames"] += 1

            if self.timing_mode == "media":
                rec["duration"] = rec["visible_frames"] / self.fps
            else:
                rec["duration"] += wall_now - rec["last_update_wall"]
                rec["last_update_wall"] = wall_now

            durations[track_id] = rec["duration"]

        return durations


def draw_presence(frame, tracks, durations):
    for x1, y1, x2, y2, track_id, _confidence in tracks:
        seconds = durations.get(track_id, 0.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"Person ID {track_id} | {seconds:.1f}s"
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

    counter_text = f"Active tracks: {len(tracks)}"
    cv2.rectangle(frame, (8, 8), (260, 42), COUNTER_BG, thickness=-1)
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


def print_session_summary(book):
    """Print every ID seen this session, including tracks that already left."""
    print("\nSession summary:")
    if not book.records:
        print("  No tracks were observed in this session.")
        return

    for track_id in sorted(book.records):
        rec = book.records[track_id]
        print(f"  Track ID {track_id}: {rec['duration']:.1f}s")


def run_presence(cap, model, confidence, source_type, fps):
    timing_mode, fps = choose_timing_mode(source_type, fps)
    book = PresenceBook(timing_mode, fps)
    delay_ms = frame_delay_ms(fps)
    frame_number = 0

    print(f"  Timing mode           : {timing_mode}")
    if timing_mode == "media":
        print(f"  Video FPS used        : {fps:.2f}")
    print("Press Q in the video window to quit.\n")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                if frame_number == 0:
                    print("Error: a frame could not be read from the source.")
                    sys.exit(1)
                print(f"\nVideo ended after {frame_number} frame(s).")
                break

            frame_number += 1
            # Reuse Milestone 3. Boxes without a ByteTrack id are skipped there.
            tracks = track_people(model, frame, confidence)
            active_ids = [item[4] for item in tracks]
            durations = book.update(active_ids, frame_number, time.perf_counter())
            annotated = draw_presence(frame, tracks, durations)
            cv2.imshow(WINDOW_TITLE, annotated)

            if frame_number == 1 or frame_number % LOG_EVERY_N_FRAMES == 0:
                print(f"Frame {frame_number} | active tracks {len(tracks)}")

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), ord("Q")):
                print(f"Quit requested after {frame_number} frame(s).")
                break
    finally:
        print_session_summary(book)
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

    print("Presence duration started.")
    print(f"  Model                 : {MODEL_FILENAME}")
    print(f"  Tracker               : {TRACKER_NAME} ({TRACKER_CONFIG})")
    print(f"  Source type           : {source_type}")
    print(f"  Source                : {source_label}")
    print(f"  Confidence threshold  : {args.confidence}")

    run_presence(cap, model, args.confidence, source_type, fps)


if __name__ == "__main__":
    main()
