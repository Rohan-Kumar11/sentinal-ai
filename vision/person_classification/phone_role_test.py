import argparse
import time

import cv2
from ultralytics import YOLO

from mock_role_classifier import classify_role


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO + ByteTrack + mock role classification "
        "on an IP Webcam stream."
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
    print("SENTINAL - PHONE CAMERA ROLE TEST")
    print("=" * 60)
    print(f"Source               : {args.source}")
    print("Model                : yolo11n.pt")
    print("Tracker              : ByteTrack")
    print("Role classifier      : MOCK")
    print(f"Confidence threshold : {args.confidence}")
    print()
    print("IMPORTANT:")
    print("Role classification is currently MOCK.")
    print("This is NOT a trained Staff/Beneficiary model.")
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
        return

    print("Phone camera stream opened successfully.")
    print()
    print("Press Q to quit.")
    print()

    frame_count = 0
    start_time = time.perf_counter()

    last_roles = {}

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

        current_roles = {}

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()

            for track_id in track_ids:
                role_data = classify_role(track_id)

                role = role_data["role"]
                role_confidence = role_data["confidence"]

                current_roles[track_id] = role

                # Find the bounding box belonging to this track.
                for index, detected_id in enumerate(track_ids):
                    if detected_id != track_id:
                        continue

                    box = result.boxes.xyxy[index].cpu().tolist()

                    x1, y1, x2, y2 = map(int, box)

                    label = (
                        f"ID {track_id} | "
                        f"{role} | "
                        f"{role_confidence:.2f}"
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
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    break

        elapsed = time.perf_counter() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        staff_count = sum(
            1 for role in current_roles.values()
            if role == "Staff"
        )

        beneficiary_count = sum(
            1 for role in current_roles.values()
            if role == "Beneficiary"
        )

        unknown_count = sum(
            1 for role in current_roles.values()
            if role == "Unknown"
        )

        cv2.putText(
            annotated_frame,
            "Sentinal - Phone Camera + Mock Roles",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            (
                f"Staff: {staff_count} | "
                f"Beneficiary: {beneficiary_count} | "
                f"Unknown: {unknown_count}"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
            "Sentinal - Phone Camera + Mock Roles",
            annotated_frame,
        )

        if current_roles != last_roles:
            print(
                f"Frame {frame_count} | "
                f"Roles: {current_roles} | "
                f"Staff: {staff_count} | "
                f"Beneficiary: {beneficiary_count} | "
                f"Unknown: {unknown_count} | "
                f"FPS: {fps:.1f}"
            )

            last_roles = current_roles.copy()

        if frame_count % 30 == 0:
            print(
                f"Frame {frame_count} | "
                f"active tracks {len(current_roles)} | "
                f"Staff {staff_count} | "
                f"Beneficiary {beneficiary_count} | "
                f"Unknown {unknown_count} | "
                f"FPS {fps:.1f}"
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
    print("Phone camera role test finished.")


if __name__ == "__main__":
    main()