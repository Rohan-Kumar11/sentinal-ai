import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO + ByteTrack on an IP Webcam stream."
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
    print("SENTINAL - PHONE CAMERA PERSON TRACKING")
    print("=" * 60)
    print(f"Source                 : {args.source}")
    print("Model                  : yolo11n.pt")
    print("Tracker                : ByteTrack")
    print("Detection              : Person only")
    print(f"Confidence threshold   : {args.confidence}")
    print()
    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("YOLO model loaded.")
    print("Opening phone camera stream...")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print()
        print("ERROR: Could not open the phone camera stream.")
        print("Check that IP Webcam is running.")
        print("Also check that the phone and laptop are on the same network.")
        return

    print("Phone camera stream opened successfully.")
    print()
    print("Press Q to quit.")
    print()

    frame_count = 0
    start_time = time.perf_counter()

    last_track_ids = set()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read frame from phone camera.")
            break

        frame_count += 1

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

        current_track_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()

            for track_id in track_ids:
                current_track_ids.add(track_id)

        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        active_tracks = len(current_track_ids)

        cv2.putText(
            annotated_frame,
            "Sentinal - Phone Camera + ByteTrack",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Active persons: {active_tracks}  FPS: {fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Sentinal - Phone Camera + ByteTrack",
            annotated_frame,
        )

        if current_track_ids != last_track_ids:
            if current_track_ids:
                print(
                    f"Frame {frame_count} | "
                    f"Active track IDs: "
                    f"{sorted(current_track_ids)} | "
                    f"FPS: {fps:.1f}"
                )
            else:
                print(
                    f"Frame {frame_count} | "
                    f"No active tracks | "
                    f"FPS: {fps:.1f}"
                )

            last_track_ids = current_track_ids.copy()

        if frame_count % 30 == 0:
            print(
                f"Frame {frame_count} | "
                f"active tracks {active_tracks} | "
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
    print("Phone camera tracking test finished.")


if __name__ == "__main__":
    main()