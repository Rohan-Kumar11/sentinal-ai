import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection on an IP Webcam stream."
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
    print("SENTINAL - PHONE CAMERA YOLO TEST")
    print("=" * 60)
    print(f"Source     : {args.source}")
    print("Model      : yolo11n.pt")
    print("Detection  : Person only")
    print(f"Confidence : {args.confidence}")
    print()
    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("YOLO model loaded.")
    print("Opening phone camera stream...")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print()
        print("ERROR: Could not open the phone camera stream.")
        print("Make sure IP Webcam is running and the URL is correct.")
        return

    print("Phone camera stream opened successfully.")
    print()
    print("Press Q to quit.")
    print()

    frame_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read frame from phone camera.")
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=args.confidence,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        person_count = 0

        boxes = results[0].boxes

        if boxes is not None:
            person_count = len(boxes)

        cv2.putText(
            annotated_frame,
            "Sentinal - Phone Camera + YOLO",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Persons: {person_count}  FPS: {fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Sentinal - Phone Camera + YOLO",
            annotated_frame,
        )

        if frame_count % 30 == 0:
            print(
                f"Frame {frame_count} | "
                f"persons {person_count} | "
                f"FPS {fps:.1f} | "
                f"elapsed {elapsed:.1f}s"
            )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print()
            print(
                f"Quit requested after {frame_count} frame(s)."
            )
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("Phone camera YOLO test finished.")


if __name__ == "__main__":
    main()