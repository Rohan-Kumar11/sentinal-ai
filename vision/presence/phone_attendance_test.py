import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO


# Add the project root to Python's import path.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from vision.presence.attendance_engine import AttendanceEngine
from vision.person_classification.mock_role_classifier import classify_role


def utc_now_iso():
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run phone camera through YOLO, ByteTrack, "
            "mock role classification, and attendance."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="IP Webcam video stream URL",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SENTINAL - PHONE CAMERA ATTENDANCE TEST")
    print("=" * 60)

    print("Pipeline:")
    print("  Phone Camera")
    print("      ↓")
    print("  IP Webcam")
    print("      ↓")
    print("  OpenCV")
    print("      ↓")
    print("  YOLO11n")
    print("      ↓")
    print("  ByteTrack")
    print("      ↓")
    print("  Mock Role")
    print("      ↓")
    print("  Attendance Engine")
    print()

    print(f"Source               : {args.source}")
    print("Model                : yolo11n.pt")
    print("Tracker              : ByteTrack")
    print("Role classifier      : MOCK")
    print(f"Confidence threshold : {args.confidence}")
    print()

    print("IMPORTANT:")
    print("Track IDs are temporary session IDs.")
    print("This is NOT identity-based attendance.")
    print("Role classification is currently MOCK.")
    print()

    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("YOLO model loaded.")
    print("Opening phone camera stream...")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print()
        print("ERROR: Could not open the phone camera stream.")
        print("Make sure IP Webcam is running.")
        print("Also check that the phone and laptop are")
        print("connected to the same network.")
        return

    print("Phone camera stream opened successfully.")
    print()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_started_at = utc_now_iso()

    attendance = AttendanceEngine()

    print(f"Session ID : {session_id}")
    print(f"Started    : {session_started_at}")
    print()
    print("Press Q to quit.")
    print()

    frame_count = 0
    start_time = time.perf_counter()
    last_status_time = start_time

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to read frame from phone camera.")
                break

            frame_count += 1

            # Timing values required by AttendanceEngine.update().
            current_time = time.perf_counter()
            current_timestamp = utc_now_iso()

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=args.confidence,
                verbose=False,
            )

            result = results[0]

            annotated_frame = result.plot()

            current_tracks = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                for index, track_id in enumerate(track_ids):
                    role_data = classify_role(track_id)

                    role = role_data["role"]
                    role_confidence = role_data["confidence"]

                    current_tracks.append(
                        {
                            "track_id": track_id,
                            "role": role,
                            "confidence": role_confidence,
                        }
                    )

                    # Update attendance using the exact
                    # AttendanceEngine API.
                    attendance.update(
                        track_id=track_id,
                        role=role,
                        current_time=current_time,
                        current_timestamp=current_timestamp,
                    )

                    box = (
                        result.boxes.xyxy[index]
                        .cpu()
                        .tolist()
                    )

                    x1, y1, x2, y2 = map(int, box)

                    label = (
                        f"ID {track_id} | "
                        f"{role} "
                        f"({role_confidence:.2f})"
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            elapsed = time.perf_counter() - start_time

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0.0
            )

            active_tracks = len(current_tracks)

            staff_count = sum(
                1
                for item in current_tracks
                if item["role"] == "Staff"
            )

            beneficiary_count = sum(
                1
                for item in current_tracks
                if item["role"] == "Beneficiary"
            )

            unknown_count = sum(
                1
                for item in current_tracks
                if item["role"] == "Unknown"
            )

            cv2.putText(
                annotated_frame,
                "Sentinal - AI Attendance",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"Active: {active_tracks} | "
                    f"Staff: {staff_count} | "
                    f"Beneficiary: {beneficiary_count} | "
                    f"Unknown: {unknown_count}"
                ),
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Sentinal - Phone Camera AI Attendance",
                annotated_frame,
            )

            now = time.perf_counter()

            if now - last_status_time >= 5.0:
                print()
                print("STATUS")
                print(f"  Frame           : {frame_count}")
                print(f"  Active tracks   : {active_tracks}")
                print(f"  Staff           : {staff_count}")
                print(f"  Beneficiary     : {beneficiary_count}")
                print(f"  Unknown         : {unknown_count}")
                print(f"  FPS             : {fps:.1f}")

                attendance.print_summary()

                last_status_time = now

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print()
                print(
                    f"Quit requested after "
                    f"{frame_count} frame(s)."
                )
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    session_ended_at = utc_now_iso()

    print()
    print("=" * 60)
    print("PHONE CAMERA ATTENDANCE SESSION SUMMARY")
    print("=" * 60)

    print(f"Session ID : {session_id}")
    print(f"Started    : {session_started_at}")
    print(f"Ended      : {session_ended_at}")
    print()

    attendance.print_summary()

    # Build the same API-ready structure used by
    # AttendanceEngine.
    session_data = attendance.get_session_data(
        session_id=session_id,
        session_started_at=session_started_at,
        session_ended_at=session_ended_at,
    )

    print()
    print("=" * 60)
    print("API-READY ATTENDANCE DATA")
    print("=" * 60)

    print(
        json.dumps(
            session_data,
            indent=2,
        )
    )

    # Save the attendance session using the existing
    # AttendanceEngine implementation.
    saved_file = attendance.save_session_data(
        session_id=session_id,
        session_started_at=session_started_at,
        session_ended_at=session_ended_at,
    )

    print()
    print("=" * 60)
    print("ATTENDANCE DATA SAVED")
    print("=" * 60)

    print(f"File: {saved_file}")

    print()
    print("Phone camera attendance test finished.")


if __name__ == "__main__":
    main()