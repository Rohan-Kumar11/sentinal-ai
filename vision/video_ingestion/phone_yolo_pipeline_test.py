import cv2
from ultralytics import YOLO

from vision.video_ingestion.camera_config import get_camera_config
from vision.video_ingestion.video_capture import VideoCaptureService


MODEL_PATH = "yolo11n.pt"


def main():
    print("=" * 60)
    print("SENTINAL - PHONE CAMERA YOLO PIPELINE TEST")
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
    total_persons_detected = 0

    print()
    print("Starting YOLO person detection...")
    print("Press Q to stop.")
    print()

    try:
        while True:
            success, frame = capture_service.read()

            if not success:
                print("ERROR: Failed to read frame.")
                break

            frame_count += 1

            results = model(
                frame,
                classes=[0],
                verbose=False,
            )

            annotated_frame = results[0].plot()

            person_count = 0

            if results[0].boxes is not None:
                person_count = len(results[0].boxes)

            total_persons_detected += person_count

            cv2.putText(
                annotated_frame,
                "Sentinal - YOLO Person Detection",
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
                f"Persons: {person_count}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Sentinal - Phone Camera YOLO Pipeline",
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
    print("PIPELINE TEST FINISHED")
    print("=" * 60)

    print(f"Frames processed : {frame_count}")
    print(f"Total detections : {total_persons_detected}")
    print("=" * 60)


if __name__ == "__main__":
    main()