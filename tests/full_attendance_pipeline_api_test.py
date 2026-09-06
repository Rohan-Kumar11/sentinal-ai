import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


# -------------------------------------------------------------
# Add project root to Python import path
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from api.client import SentinalApiClient
from vision.person_classification.mock_role_classifier import (
    classify_role,
)
from vision.presence.attendance_engine import AttendanceEngine
from vision.video_ingestion.camera_config import (
    get_camera_config,
)
from vision.video_ingestion.video_capture import (
    VideoCaptureService,
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "experiments"
    / "attendance_sessions"
)

MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.5

MAX_FRAMES = 300


def generate_session_id():
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def generate_timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat()


def main():
    print("=" * 60)
    print(
        "SENTINAL - FULL ATTENDANCE "
        "PIPELINE + API TEST"
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Camera configuration
    # ---------------------------------------------------------

    camera_config = get_camera_config("phone")

    print()
    print("Camera:")
    print(
        f"Name: {camera_config.name}"
    )
    print(
        f"Source: {camera_config.source}"
    )
    print(
        f"Type: {camera_config.source_type}"
    )

    # ---------------------------------------------------------
    # 2. Create session
    # ---------------------------------------------------------

    session_id = generate_session_id()

    session_started_at = generate_timestamp()

    print()
    print(
        f"Session ID: {session_id}"
    )

    print(
        f"Session started: "
        f"{session_started_at}"
    )

    # ---------------------------------------------------------
    # 3. Load YOLO
    # ---------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded.")

    # ---------------------------------------------------------
    # 4. Create services
    # ---------------------------------------------------------

    capture_service = VideoCaptureService(
        camera_config
    )

    attendance_engine = AttendanceEngine()

    api_client = SentinalApiClient()

    # ---------------------------------------------------------
    # 5. Open camera
    # ---------------------------------------------------------

    print()
    print("Opening camera...")

    if not capture_service.open():
        print()
        print(
            "ERROR: Could not open camera."
        )
        return

    print()
    print("Camera opened successfully.")

    # ---------------------------------------------------------
    # 6. Process video
    # ---------------------------------------------------------

    frame_count = 0
    tracked_detections = 0
    unique_track_ids = set()

    start_time = time.perf_counter()

    print()
    print(
        f"Processing maximum "
        f"{MAX_FRAMES} frames..."
    )

    try:
        while frame_count < MAX_FRAMES:

            success, frame = (
                capture_service.read()
            )

            if not success or frame is None:
                print(
                    "ERROR: Failed to read "
                    "frame from camera."
                )
                break

            frame_count += 1

            # -------------------------------------------------
            # YOLO + ByteTrack
            # -------------------------------------------------

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            result = results[0]

            if (
                result.boxes is None
                or result.boxes.id is None
            ):
                continue

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            current_time = (
                time.perf_counter()
            )

            current_timestamp = (
                generate_timestamp()
            )

            for track_id in track_ids:

                track_id = int(track_id)

                unique_track_ids.add(
                    track_id
                )

                tracked_detections += 1

                # ---------------------------------------------
                # Mock role classification
                # ---------------------------------------------

                role_result = classify_role(
                    track_id
                )

                role = role_result["role"]

                # ---------------------------------------------
                # Attendance engine
                # ---------------------------------------------

                attendance_engine.update(
                    track_id=track_id,
                    role=role,
                    current_time=current_time,
                    current_timestamp=current_timestamp,
                )

            # -------------------------------------------------
            # Display
            # -------------------------------------------------

            annotated_frame = result.plot()

            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Tracked: {len(unique_track_ids)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Sentinal Full Attendance Pipeline",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print()
                print(
                    "Stopped by user."
                )
                break

    finally:
        capture_service.release()
        cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # 7. Session timing
    # ---------------------------------------------------------

    session_ended_at = generate_timestamp()

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    print()
    print(
        f"Processing time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    print(
        f"Tracked detections: "
        f"{tracked_detections}"
    )

    print(
        f"Unique track IDs: "
        f"{len(unique_track_ids)}"
    )

    # ---------------------------------------------------------
    # 8. Save attendance JSON
    # ---------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    attendance_engine.save_session_data(
        session_id=session_id,
        session_started_at=session_started_at,
        session_ended_at=session_ended_at,
        output_directory=OUTPUT_DIRECTORY,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / f"attendance_{session_id}.json"
    )

    print()
    print(
        "Attendance session saved:"
    )
    print(output_path)

    # ---------------------------------------------------------
    # 9. Load saved JSON
    # ---------------------------------------------------------

    if not output_path.exists():
        print(
            "ERROR: Attendance JSON "
            "was not created."
        )
        return

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        attendance_data = json.load(file)

    print()
    print("Saved attendance data:")

    print(
        f"Session ID: "
        f"{attendance_data['session_id']}"
    )

    print(
        f"Total tracked: "
        f"{attendance_data['total_tracked']}"
    )

    print(
        f"Staff: "
        f"{attendance_data['staff']}"
    )

    print(
        f"Beneficiary: "
        f"{attendance_data['beneficiary']}"
    )

    print(
        f"Unknown: "
        f"{attendance_data['unknown']}"
    )

    # ---------------------------------------------------------
    # 10. Upload to FastAPI
    # ---------------------------------------------------------

    print()
    print(
        "Uploading attendance session "
        "to FastAPI..."
    )

    uploaded = api_client.upload_attendance(
        attendance_data
    )

    print(
        "Upload successful."
    )

    print(
        f"API session ID: "
        f"{uploaded.get('session_id')}"
    )

    # ---------------------------------------------------------
    # 11. Verify exact session from API
    # ---------------------------------------------------------

    print()
    print(
        "Verifying session through API..."
    )

    api_session = (
        api_client.get_attendance_session(
            session_id
        )
    )

    print(
        f"API returned session ID: "
        f"{api_session.get('session_id')}"
    )

    print(
        f"API total tracked: "
        f"{api_session.get('total_tracked')}"
    )

    # ---------------------------------------------------------
    # 12. Verify API role statistics
    # ---------------------------------------------------------

    role_statistics = (
        api_client.get_session_role_statistics(
            session_id
        )
    )

    print()
    print(
        "API role statistics:"
    )

    for statistic in role_statistics.get(
        "statistics",
        [],
    ):
        print(
            f"{statistic.get('role')}: "
            f"{statistic.get('people')} people"
        )

    # ---------------------------------------------------------
    # 13. Final validation
    # ---------------------------------------------------------

    success = (
        uploaded.get("session_id")
        == session_id
        and api_session.get("session_id")
        == session_id
        and role_statistics.get("session_id")
        == session_id
    )

    print()
    print("=" * 60)

    if success:
        print(
            "FULL ATTENDANCE PIPELINE + API "
            "SUCCESSFUL"
        )
    else:
        print(
            "FULL ATTENDANCE PIPELINE + API "
            "FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()