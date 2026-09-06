"""
Sentinal - Development Mock Role Classifier

IMPORTANT:
This is NOT a trained Staff/Beneficiary AI model.

YOLO11n:
    Detects people.

ByteTrack:
    Assigns temporary Track IDs.

Mock Role Classifier:
    Assigns Staff / Beneficiary / Unknown using Track IDs.

This module exists ONLY so that the downstream Sentinal pipeline
can be developed before a real Staff/Beneficiary classifier is trained.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# DEVELOPMENT-ONLY ROLE MAPPING
#
# Track IDs are temporary.
#
# For example:
#   ID 1, 2 -> Staff
#   ID 3, 4 -> Beneficiary
#   Any other ID -> Unknown
#
# Change these numbers ONLY for development/testing.
# ---------------------------------------------------------------------------

STAFF_TRACK_IDS = {1, 2}
BENEFICIARY_TRACK_IDS = {3, 4}


STAFF = "Staff"
BENEFICIARY = "Beneficiary"
UNKNOWN = "Unknown"


# ============================================================================
# MOCK CLASSIFIER
# ============================================================================

def classify_role(track_id: int) -> dict:
    """
    Development-only role classification.

    This is deterministic Track-ID mapping.
    It is NOT machine learning.
    """

    if track_id in STAFF_TRACK_IDS:
        return {
            "track_id": track_id,
            "role": STAFF,
            "confidence": 1.0,
            "classifier_type": "MOCK",
        }

    if track_id in BENEFICIARY_TRACK_IDS:
        return {
            "track_id": track_id,
            "role": BENEFICIARY,
            "confidence": 1.0,
            "classifier_type": "MOCK",
        }

    return {
        "track_id": track_id,
        "role": UNKNOWN,
        "confidence": 0.0,
        "classifier_type": "MOCK",
    }


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sentinal development-only role classifier "
            "using YOLO11n + ByteTrack."
        )
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file path. If omitted, webcam is used.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO person detection confidence (default: 0.5).",
    )

    return parser.parse_args()


# ============================================================================
# VIDEO SOURCE
# ============================================================================

def get_source(source):
    """
    Return webcam index 0 or validate a supplied video file.
    """

    if source is None:
        return 0

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(
            f"Video source does not exist: {path}"
        )

    return str(path)


# ============================================================================
# MAIN
# ============================================================================

def run():

    args = parse_args()

    source = get_source(args.source)

    print()
    print("=" * 70)
    print("SENTINAL - DEVELOPMENT ROLE CLASSIFIER")
    print("=" * 70)
    print()
    print("WARNING: THIS IS A MOCK CLASSIFIER")
    print("It does NOT determine a person's real-world role.")
    print()
    print(f"Model          : {MODEL_PATH}")
    print(f"Tracker        : ByteTrack ({TRACKER_CONFIG})")
    print(f"Confidence     : {args.confidence}")
    print()
    print("Temporary role mapping:")
    print(f"  Staff        : Track IDs {sorted(STAFF_TRACK_IDS)}")
    print(f"  Beneficiary  : Track IDs {sorted(BENEFICIARY_TRACK_IDS)}")
    print("  Unknown      : All other Track IDs")
    print()
    print("Controls:")
    print("  Q = Quit")
    print()
    print("=" * 70)
    print()

    # ------------------------------------------------------------------------
    # Load YOLO
    # ------------------------------------------------------------------------

    model = YOLO(MODEL_PATH)

    # ------------------------------------------------------------------------
    # Open video source
    # ------------------------------------------------------------------------

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}"
        )

    # ------------------------------------------------------------------------
    # Video information
    # ------------------------------------------------------------------------

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        source_fps = 0.0

    frame_count = 0
    start_time = time.perf_counter()

    # Used to avoid printing the same Track ID information every frame.
    last_active_ids = None

    # =========================================================================
    # FRAME LOOP
    # =========================================================================

    while True:

        success, frame = cap.read()

        if not success:
            print()
            print("Video ended or frame could not be read.")
            break

        frame_count += 1

        # --------------------------------------------------------------------
        # YOLO + ByteTrack
        # --------------------------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=args.confidence,
            classes=[0],  # COCO class 0 = person
            verbose=False,
        )

        result = results[0]

        detection_count = 0
        active_ids = []

        staff_count = 0
        beneficiary_count = 0
        unknown_count = 0

        # --------------------------------------------------------------------
        # Process detections
        # --------------------------------------------------------------------

        if result.boxes is not None:

            detection_count = len(result.boxes)

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):

                active_ids.append(track_id)

                x1, y1, x2, y2 = map(int, box)

                # ------------------------------------------------------------
                # Mock classification
                # ------------------------------------------------------------

                classification = classify_role(track_id)

                role = classification["role"]

                confidence = classification["confidence"]

                # ------------------------------------------------------------
                # Count roles
                # ------------------------------------------------------------

                if role == STAFF:
                    staff_count += 1

                elif role == BENEFICIARY:
                    beneficiary_count += 1

                else:
                    unknown_count += 1

                # ------------------------------------------------------------
                # Draw bounding box
                # ------------------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                )

                # ------------------------------------------------------------
                # Role label
                # ------------------------------------------------------------

                label = (
                    f"ID {track_id} | "
                    f"{role}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # =========================================================================
        # ACTIVE TRACK IDS
        # =========================================================================

        active_ids = sorted(active_ids)

        # Print when active IDs change.
        if active_ids != last_active_ids:

            print(
                f"Frame {frame_count} | "
                f"Detections: {detection_count} | "
                f"Active Track IDs: {active_ids}"
            )

            if active_ids:

                for track_id in active_ids:

                    classification = classify_role(track_id)

                    print(
                        f"    ID {track_id} -> "
                        f"{classification['role']} "
                        f"(confidence={classification['confidence']:.1f}, "
                        f"type={classification['classifier_type']})"
                    )

            else:

                print("    No actively tracked people.")

            last_active_ids = active_ids

        # =========================================================================
        # COUNTS
        # =========================================================================

        total_count = (
            staff_count
            + beneficiary_count
            + unknown_count
        )

        elapsed = time.perf_counter() - start_time

        actual_fps = (
            frame_count / elapsed
            if elapsed > 0
            else 0
        )

        # =========================================================================
        # UI PANEL
        # =========================================================================

        panel_height = 205

        cv2.rectangle(
            frame,
            (10, 10),
            (390, panel_height),
            (0, 0, 0),
            -1,
        )

        # Title
        cv2.putText(
            frame,
            "SENTINAL - MOCK AI",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        # Warning
        cv2.putText(
            frame,
            "DEVELOPMENT ONLY",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # Counts
        cv2.putText(
            frame,
            f"Staff       : {staff_count}",
            (20, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Beneficiary : {beneficiary_count}",
            (20, 119),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Unknown     : {unknown_count}",
            (20, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Total       : {total_count}",
            (20, 169),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS         : {actual_fps:.2f}",
            (210, 169),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Frame       : {frame_count}",
            (210, 144),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # =========================================================================
        # NO PERSON MESSAGE
        # =========================================================================

        if total_count == 0:

            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (20, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # =========================================================================
        # DISPLAY
        # =========================================================================

        cv2.imshow(
            "Sentinal - Development Mock Role Classification",
            frame,
        )

        # =========================================================================
        # PERIODIC TERMINAL STATUS
        # =========================================================================

        if frame_count % 30 == 0:

            print(
                f"STATUS | "
                f"Frame {frame_count} | "
                f"Detections {detection_count} | "
                f"Active {len(active_ids)} | "
                f"Staff {staff_count} | "
                f"Beneficiary {beneficiary_count} | "
                f"Unknown {unknown_count} | "
                f"FPS {actual_fps:.2f}"
            )

        # =========================================================================
        # QUIT
        # =========================================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print()
            print(
                f"Quit requested after "
                f"{frame_count} frame(s)."
            )

            break

    # =========================================================================
    # CLEANUP
    # =========================================================================

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("DEVELOPMENT ROLE CLASSIFICATION STOPPED")
    print("=" * 70)
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run()