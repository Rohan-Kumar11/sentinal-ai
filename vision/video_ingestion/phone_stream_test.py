import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser(
        description="Test an IP Webcam video stream using OpenCV."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="IP Webcam video stream URL",
    )

    args = parser.parse_args()

    source = args.source

    print("=" * 60)
    print("SENTINAL - IP CAMERA STREAM TEST")
    print("=" * 60)
    print(f"Source: {source}")
    print()
    print("Opening IP camera stream...")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print()
        print("ERROR: Could not open the IP camera stream.")
        print()
        print("Check that:")
        print("1. IP Webcam server is running on the phone.")
        print("2. Phone and laptop are on the same network.")
        print("3. The stream URL is correct.")
        print("4. The phone has not disconnected from Wi-Fi.")
        print()
        return

    print("SUCCESS: IP camera stream opened.")
    print()
    print("Press Q to stop the test.")
    print()

    frame_count = 0
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to read frame from IP camera.")
            break

        frame_count += 1

        elapsed = time.perf_counter() - start_time

        fps = frame_count / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            frame,
            "Sentinal - Phone Camera",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Frames: {frame_count}  FPS: {fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Sentinal - IP Camera Test", frame)

        if frame_count % 30 == 0:
            print(
                f"Frame {frame_count} | "
                f"FPS {fps:.1f} | "
                f"elapsed {elapsed:.1f}s"
            )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print()
            print(f"Quit requested after {frame_count} frame(s).")
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("IP camera test finished.")


if __name__ == "__main__":
    main()