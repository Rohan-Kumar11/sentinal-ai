import cv2

from vision.video_ingestion.camera_config import CameraConfig


class VideoCaptureService:
    """
    Reusable video capture service for Sentinal.

    This class handles opening, reading, and releasing
    different video sources.

    The source can be:
    - IP camera
    - RTSP camera
    - webcam
    - video file
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture = None

    def open(self) -> bool:
        """
        Open the configured video source.

        Returns:
            True if the source opened successfully.
            False otherwise.
        """

        if not self.config.enabled:
            print(
                f"Camera '{self.config.name}' is disabled."
            )
            return False

        print(
            f"Opening camera: {self.config.name}"
        )

        print(
            f"Source type: {self.config.source_type}"
        )

        print(
            f"Source: {self.config.source}"
        )

        self.capture = cv2.VideoCapture(
            self.config.source
        )

        if not self.capture.isOpened():
            print()
            print(
                f"ERROR: Could not open camera "
                f"'{self.config.name}'."
            )

            self.capture.release()
            self.capture = None

            return False

        print(
            f"SUCCESS: Camera '{self.config.name}' "
            f"opened successfully."
        )

        return True

    def read(self):
        """
        Read one frame from the configured source.

        Returns:
            Tuple of:
                success: bool
                frame: OpenCV frame or None
        """

        if self.capture is None:
            return False, None

        return self.capture.read()

    def is_opened(self) -> bool:
        """
        Check whether the video source is currently open.
        """

        return (
            self.capture is not None
            and self.capture.isOpened()
        )

    def release(self) -> None:
        """
        Release the video source.
        """

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        print(
            f"Camera released: {self.config.name}"
        )


def main():
    """
    Basic test for the video capture service.
    """

    from vision.video_ingestion.camera_config import (
        get_camera_config,
    )

    print("=" * 60)
    print("SENTINAL - VIDEO CAPTURE SERVICE TEST")
    print("=" * 60)

    config = get_camera_config("phone")

    capture_service = VideoCaptureService(config)

    if not capture_service.open():
        print()
        print("Video capture test failed.")
        return

    print()
    print("Reading frames from camera...")
    print("Press Q to stop.")
    print()

    frame_count = 0

    try:
        while True:
            success, frame = capture_service.read()

            if not success:
                print(
                    "ERROR: Failed to read frame."
                )
                break

            frame_count += 1

            cv2.putText(
                frame,
                "Sentinal - Video Capture Service",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Frames: {frame_count}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Sentinal - Video Capture Service",
                frame,
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
    print("VIDEO CAPTURE SERVICE TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()