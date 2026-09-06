import time
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO

from vision.video_ingestion.camera_config import get_camera_config
from vision.video_ingestion.video_capture import VideoCaptureService
from vision.person_classification.mock_role_classifier import (
    classify_role,
)
from vision.presence.attendance_engine import AttendanceEngine


MODEL_PATH = "yolo11n.pt"


def main():
    print("=" * 60)
    print("SENTINAL - PHONE CAMERA ATTENDANCE PIPELINE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create session information
    # ---------------------------------------------------------

    session_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    session_started_at = datetime.now(
        timezone.utc
    ).isoformat()

    print()
    print(f"Session ID : {session_id}")
    print(f"Started    : {session_started_at}")

    # ---------------------------------------------------------
    # Load YOLO model
    # ---------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully.")
    print()

    # ---------------------------------------------------------
    # Open phone camera
    # ---------------------------------------------------------

    config = get_camera_config("phone")

    capture_service = VideoCaptureService(config)

    if not capture_service.open():
        print()
        print("ERROR: Could not open phone camera.")
        return

    # ---------------------------------------------------------
    # Create attendance engine
    # ---------------------------------------------------------

    attendance_engine = AttendanceEngine()

    frame_count = 0
    total_tracked_detections = 0
    unique_track_ids = set()

    start_time = time.perf_counter()

    print()
    print("Starting full attendance pipeline...")
    print("Press Q to stop.")
    print()

    # ---------------------------------------------------------
    # Main processing loop
    # ---------------------------------------------------------

    try:
        while True:
            success, frame = capture_service.read()

            if not success:
                print("ERROR: Failed to read frame.")
                break

            frame_count += 1

            current_time = time.perf_counter()

            current_timestamp = (
                datetime.now(timezone.utc)
                .isoformat()
            )

            # -------------------------------------------------
            # YOLO + ByteTrack
            # -------------------------------------------------

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False,
            )

            annotated_frame = results[0].plot()

            current_track_ids = []

            if results[0].boxes is not None:
                boxes = results[0].boxes

                if boxes.id is not None:
                    current_track_ids = (
                        boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

            total_tracked_detections += len(
                current_track_ids
            )

            # -------------------------------------------------
            # Role classification + attendance
            # -------------------------------------------------

            for track_id in current_track_ids:
                track_id = int(track_id)

                unique_track_ids.add(track_id)

                role_result = classify_role(track_id)

                role = role_result["role"]

                attendance_engine.update(
                    track_id=track_id,
                    role=role,
                    current_time=current_time,
                    current_timestamp=current_timestamp,
                )

            # -------------------------------------------------
            # Display information
            # -------------------------------------------------

            cv2.putText(
                annotated_frame,
                "Sentinal - Attendance Pipeline",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Frames: {frame_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Tracked: {len(current_track_ids)}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Unique IDs: {len(unique_track_ids)}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                "Role classifier: MOCK",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.imshow(
                "Sentinal - Phone Attendance Pipeline",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print()
                print(
                    f"Quit requested after "
                    f"{frame_count} frame(s)."
                )
                break

    finally:
        capture_service.release()
        cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # Session end timestamp
    # ---------------------------------------------------------

    session_ended_at = datetime.now(
        timezone.utc
    ).isoformat()

    elapsed = time.perf_counter() - start_time

    # ---------------------------------------------------------
    # Print pipeline statistics
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("ATTENDANCE PIPELINE TEST FINISHED")
    print("=" * 60)

    print(f"Session ID                : {session_id}")
    print(f"Frames processed          : {frame_count}")
    print(
        f"Tracked detections        : "
        f"{total_tracked_detections}"
    )
    print(
        f"Unique track IDs observed : "
        f"{len(unique_track_ids)}"
    )
    print(f"Processing time           : {elapsed:.2f}s")

    # ---------------------------------------------------------
    # Attendance summary
    # ---------------------------------------------------------

    print()
    print("Attendance summary:")

    attendance_engine.print_summary()

    # ---------------------------------------------------------
    # Save attendance session
    # ---------------------------------------------------------

    saved_file = attendance_engine.save_session_data(
        session_id=session_id,
        session_started_at=session_started_at,
        session_ended_at=session_ended_at,
    )

    print()
    print("=" * 60)
    print("ATTENDANCE DATA SAVED SUCCESSFULLY")
    print("=" * 60)

    print(f"File: {saved_file}")

    print()
    print("Session timestamps:")
    print(f"Started : {session_started_at}")
    print(f"Ended   : {session_ended_at}")

    print("=" * 60)


if __name__ == "__main__":
    main()