import cv2
from ultralytics import YOLO

from vision.video_ingestion.camera_config import get_camera_config
from vision.video_ingestion.video_capture import VideoCaptureService


MODEL_PATH = "yolo11n.pt"


def main():
    print("=" * 60)
    print("SENTINAL - PHONE CAMERA BYTE TRACK PIPELINE TEST")
    print("=" * 60)

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully.")
    print()

    config = get_camera_config("phone")

    capture_service = VideoCaptureService(config)

    if not capture_service.open():
        print()
        print("ERROR: Could not open phone camera.")
        return

    frame_count = 0
    total_tracked_detections = 0
    unique_track_ids = set()

    print()
    print("Starting YOLO + ByteTrack tracking...")
    print("Press Q to stop.")
    print()

    try:
        while True:
            success, frame = capture_service.read()

            if not success:
                print("ERROR: Failed to read frame.")
                break

            frame_count += 1

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
                        boxes.id.int()
                        .cpu()
                        .tolist()
                    )

            for track_id in current_track_ids:
                unique_track_ids.add(track_id)

            total_tracked_detections += len(
                current_track_ids
            )

            cv2.putText(
                annotated_frame,
                "Sentinal - YOLO + ByteTrack",
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
                f"Tracking: {len(current_track_ids)}",
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

            cv2.imshow(
                "Sentinal - Phone Camera ByteTrack",
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

    print()
    print("=" * 60)
    print("BYTE TRACK PIPELINE TEST FINISHED")
    print("=" * 60)

    print(f"Frames processed          : {frame_count}")
    print(
        f"Tracked detections        : "
        f"{total_tracked_detections}"
    )
    print(
        f"Unique track IDs observed : "
        f"{len(unique_track_ids)}"
    )

    if unique_track_ids:
        sorted_ids = sorted(unique_track_ids)

        print(
            f"Track IDs                 : "
            f"{sorted_ids}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()