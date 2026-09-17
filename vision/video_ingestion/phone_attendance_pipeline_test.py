
"""
Phone Camera Attendance Pipeline

Phase 2:
- Supports programmatic start/stop.
- Runs AI attendance processing in a background thread.
- Keeps the existing CLI/demo behavior.
- Safely finalizes and saves an attendance session.
- Prevents duplicate sessions.
- Exposes runtime status for future FastAPI integration.

Pipeline:

Phone/IP Camera
       ↓
OpenCV Frame Capture
       ↓
YOLO Person Detection + ByteTrack Tracking
       ↓
Mock Role Classification
       ↓
AttendanceEngine
       ↓
Session JSON
"""

import os
import threading
import time
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO

from vision.video_ingestion.camera_config import get_camera_config
from vision.video_ingestion.video_capture import VideoCaptureService
from vision.person_classification.mock_role_classifier import classify_role
from vision.presence.attendance_engine import AttendanceEngine


MODEL_PATH = "yolo11n.pt"


class PhoneAttendancePipelineService:
    """
    Background service for phone-camera AI attendance.

    This class is designed so that FastAPI can later control the
    attendance pipeline without directly managing the camera loop.

    Current responsibilities:
    - Start an AI attendance session
    - Run camera + YOLO + tracking in background
    - Maintain active AttendanceEngine state
    - Stop the session safely
    - Save finalized session data
    - Expose runtime status
    """

    def __init__(
        self,
        camera_name: str = "phone",
        model_path: str = MODEL_PATH,
    ):
        self.camera_name = camera_name
        self.model_path = model_path

        # AI model
        self._model = None

        # Camera service
        self._capture_service = None

        # Attendance engine for the current session
        self._attendance_engine = None

        # Background worker
        self._thread = None
        self._stop_event = threading.Event()

        # General state lock
        self._lock = threading.Lock()

        # Prevent multiple simultaneous finalizations
        self._finalize_lock = threading.Lock()

        # Runtime state
        self._running = False

        self._session_id = None
        self._session_started_at = None
        self._session_ended_at = None

        self._frame_count = 0

        self._last_saved_file = None
        self._last_error = None

        # Cached result after successful finalization
        self._finalized_result = None

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def is_running(self):
        """
        Returns True when an attendance session is currently open.

        This represents session state, not necessarily worker-thread
        health. Use is_loop_alive() to check whether the processing
        thread is actually alive.
        """
        with self._lock:
            return self._running

    # ------------------------------------------------------------------
    # THREAD STATE
    # ------------------------------------------------------------------

    def is_loop_alive(self):
        """
        Returns True when the background processing thread is alive.
        """
        with self._lock:
            thread = self._thread

        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------------
    # ACTIVE ATTENDANCE
    # ------------------------------------------------------------------

    def get_active_count(self):
        """
        Returns the current number of unique attendance records
        maintained by AttendanceEngine for this session.
        """

        with self._lock:
            engine = self._attendance_engine

            if engine is None:
                return 0

            try:
                return len(engine.get_records())
            except AttributeError:
                return 0

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def get_status(self):
        """
        Returns current pipeline status.

        This structure is intentionally suitable for future
        FastAPI responses.
        """

        with self._lock:
            thread = self._thread

            active_count = 0

            if self._attendance_engine is not None:
                try:
                    active_count = len(
                        self._attendance_engine.get_records()
                    )
                except AttributeError:
                    active_count = 0

            status = {
                "running": self._running,
                "loop_alive": (
                    thread is not None
                    and thread.is_alive()
                ),
                "session_id": self._session_id,
                "session_started_at": self._session_started_at,
                "session_ended_at": self._session_ended_at,
                "frame_count": self._frame_count,
                "active_count": active_count,
                "last_saved_file": self._last_saved_file,
                "last_error": self._last_error,
            }

        return status

    # ------------------------------------------------------------------
    # START
    # ------------------------------------------------------------------

    def start(self, display: bool = False):
        """
        Start a new AI attendance session.

        display=False:
            Intended for FastAPI/background service use.

        display=True:
            Intended for local CLI/demo testing with OpenCV window.
        """

        with self._lock:

            # ----------------------------------------------------------
            # Prevent duplicate sessions
            # ----------------------------------------------------------

            if self._running:

                if (
                    self._thread is not None
                    and self._thread.is_alive()
                ):
                    raise RuntimeError(
                        "AI attendance pipeline is already running."
                    )

                raise RuntimeError(
                    "AI attendance session exists but its "
                    "processing loop has stopped. "
                    "Finalize the existing session before "
                    "starting another one."
                )

            # ----------------------------------------------------------
            # Reset runtime state for a new session
            # ----------------------------------------------------------

            self._stop_event.clear()

            self._frame_count = 0

            self._last_error = None
            self._last_saved_file = None

            self._session_ended_at = None

            self._finalized_result = None

            # ----------------------------------------------------------
            # Create session ID
            # ----------------------------------------------------------

            now = datetime.now(timezone.utc)

            self._session_id = now.strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            self._session_started_at = now.isoformat()

            # ----------------------------------------------------------
            # Load YOLO model
            # ----------------------------------------------------------

            print("Loading YOLO model...")

            try:
                self._model = YOLO(self.model_path)

            except Exception as exc:

                self._last_error = (
                    f"Failed to load YOLO model: {exc}"
                )

                self._session_id = None
                self._session_started_at = None

                raise RuntimeError(
                    self._last_error
                ) from exc

            print("YOLO model loaded successfully.")

            # ----------------------------------------------------------
            # Load camera configuration
            # ----------------------------------------------------------

            try:
                camera_config = get_camera_config(
                    self.camera_name
                )

            except Exception as exc:

                self._last_error = (
                    f"Failed to load camera configuration: {exc}"
                )

                raise RuntimeError(
                    self._last_error
                ) from exc

            # ----------------------------------------------------------
            # Open camera
            # ----------------------------------------------------------

            try:

                self._capture_service = VideoCaptureService(
                    camera_config
                )

                if not self._capture_service.open():

                    raise RuntimeError(
                        "Camera could not be opened."
                    )

            except Exception as exc:

                self._last_error = (
                    f"Failed to start camera: {exc}"
                )

                self._capture_service = None

                raise RuntimeError(
                    self._last_error
                ) from exc

            # ----------------------------------------------------------
            # Create fresh AttendanceEngine
            # ----------------------------------------------------------

            self._attendance_engine = AttendanceEngine()

            # ----------------------------------------------------------
            # Start background thread
            # ----------------------------------------------------------

            self._thread = threading.Thread(
                target=self._run_loop,
                args=(display,),
                daemon=True,
                name=f"PhoneAttendance-{self._session_id}",
            )

            self._running = True

            self._thread.start()

        print(
            f"AI attendance session started: "
            f"{self._session_id}"
        )

        return self.get_status()

    # ------------------------------------------------------------------
    # MAIN AI LOOP
    # ------------------------------------------------------------------

    def _run_loop(self, display: bool = False):
        """
        Background AI processing loop.

        Phone Camera
             ↓
        Frame
             ↓
        YOLO + ByteTrack
             ↓
        Role Classification
             ↓
        AttendanceEngine
        """

        print(
            "AI attendance processing loop started."
        )

        try:

            while not self._stop_event.is_set():

                # ------------------------------------------------------
                # Capture frame
                # ------------------------------------------------------

                capture_service = self._capture_service

                if capture_service is None:

                    raise RuntimeError(
                        "Camera service is not available."
                    )

                success, frame = capture_service.read()

                if not success or frame is None:

                    with self._lock:
                        self._last_error = (
                            "Camera frame could not be read."
                        )

                    print(
                        "Camera frame could not be read. "
                        "Stopping processing loop."
                    )

                    break

                # ------------------------------------------------------
                # Frame counter
                # ------------------------------------------------------

                with self._lock:
                    self._frame_count += 1

                # ------------------------------------------------------
                # YOLO detection + ByteTrack tracking
                # ------------------------------------------------------

                results = self._model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    verbose=False,
                )

                # ------------------------------------------------------
                # Process tracked people
                # ------------------------------------------------------

                if results:

                    result = results[0]

                    boxes = result.boxes

                    if (
                        boxes is not None
                        and boxes.id is not None
                    ):

                        track_ids = (
                            boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                        # --------------------------------------------------
                        # Process each tracked person
                        # --------------------------------------------------

                        for track_id in track_ids:

                            # ----------------------------------------------
                            # Existing mock role classifier
                            # ----------------------------------------------

                            role = classify_role(track_id)

                            # ----------------------------------------------
                            # Current timestamp
                            # ----------------------------------------------

                            current_time = time.time()

                            current_timestamp = (
                                datetime.now(
                                    timezone.utc
                                ).isoformat()
                            )

                            # ----------------------------------------------
                            # AttendanceEngine update
                            # ----------------------------------------------

                            with self._lock:

                                engine = (
                                    self._attendance_engine
                                )

                                if engine is not None:

                                    engine.update(
                                        track_id=track_id,
                                        role=role,
                                        current_time=current_time,
                                        current_timestamp=(
                                            current_timestamp
                                        ),
                                    )

                # ------------------------------------------------------
                # Optional OpenCV display
                # ------------------------------------------------------

                if display:

                    annotated_frame = frame

                    if results:

                        try:

                            annotated_frame = (
                                results[0].plot()
                            )

                        except Exception:

                            annotated_frame = frame

                    cv2.imshow(
                        "Phone Camera Attendance",
                        annotated_frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):

                        print(
                            "Q pressed. "
                            "Stopping AI attendance loop."
                        )

                        self._stop_event.set()

                        break

        except Exception as exc:

            with self._lock:
                self._last_error = str(exc)

            print(
                f"AI attendance processing error: {exc}"
            )

        finally:

            # ----------------------------------------------------------
            # Release camera resources
            # ----------------------------------------------------------

            try:

                if self._capture_service is not None:

                    self._capture_service.release()

            except Exception as exc:

                with self._lock:

                    if self._last_error is None:

                        self._last_error = (
                            f"Camera cleanup error: {exc}"
                        )

            # ----------------------------------------------------------
            # Close OpenCV windows
            # ----------------------------------------------------------

            if display:

                try:
                    cv2.destroyAllWindows()

                except Exception:
                    pass

            print(
                "AI attendance processing loop ended."
            )

    # ------------------------------------------------------------------
    # STOP
    # ------------------------------------------------------------------

    def stop(self, timeout: float = 10):
        """
        Safely stop and finalize the current attendance session.

        Important:
        The session is saved ONLY after the worker thread has
        completely stopped.

        If the worker does not stop within timeout, the method
        raises RuntimeError and does NOT save partial data.
        """

        # --------------------------------------------------------------
        # Capture current state
        # --------------------------------------------------------------

        with self._lock:

            # ----------------------------------------------------------
            # Already finalized
            # ----------------------------------------------------------

            if not self._running:

                if self._finalized_result is not None:

                    return self._finalized_result

                return {
                    "message": (
                        "No AI attendance session is running."
                    )
                }

            thread = self._thread

            engine = self._attendance_engine

            session_id = self._session_id

            session_started_at = (
                self._session_started_at
            )

            # Tell worker to stop
            self._stop_event.set()

        print(
            f"Stopping AI attendance session: "
            f"{session_id}"
        )

        # --------------------------------------------------------------
        # Wait for worker thread
        # --------------------------------------------------------------

        if thread is not None and thread.is_alive():

            thread.join(timeout=timeout)

        # --------------------------------------------------------------
        # Never finalize while worker is still running
        # --------------------------------------------------------------

        if thread is not None and thread.is_alive():

            raise RuntimeError(
                "AI attendance processing loop did not "
                "stop within the requested timeout. "
                "Session was not finalized."
            )

        # --------------------------------------------------------------
        # Finalize exactly once
        # --------------------------------------------------------------

        with self._finalize_lock:

            # Another stop() call may have finalized it
            # while this call was waiting.

            with self._lock:

                if self._finalized_result is not None:

                    return self._finalized_result

            # ----------------------------------------------------------
            # Validate engine
            # ----------------------------------------------------------

            if engine is None:

                ended_at = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                with self._lock:

                    self._session_ended_at = ended_at

                    self._running = False

                    self._thread = None

                    result = {
                        "session": session_id,
                        "saved_file": None,
                        "summary": None,
                    }

                    self._finalized_result = result

                return result

            # ----------------------------------------------------------
            # Finalization timestamp
            # ----------------------------------------------------------

            ended_at = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            # ----------------------------------------------------------
            # Save session
            # ----------------------------------------------------------

            try:

                saved_file = (
                    engine.save_session_data(
                        session_id=session_id,
                        session_started_at=(
                            session_started_at
                        ),
                        session_ended_at=ended_at,
                    )
                )

                summary = (
                    engine.get_session_data(
                        session_id=session_id,
                        session_started_at=(
                            session_started_at
                        ),
                        session_ended_at=ended_at,
                    )
                )

            except Exception as exc:

                with self._lock:

                    self._last_error = (
                        f"Failed to finalize session: "
                        f"{exc}"
                    )

                # Keep _running=True so the caller can retry.

                raise RuntimeError(
                    f"Failed to finalize attendance session: "
                    f"{exc}"
                ) from exc

            # ----------------------------------------------------------
            # Update final session state
            # ----------------------------------------------------------

            result = {
                "session": session_id,
                "saved_file": saved_file,
                "summary": summary,
            }

            with self._lock:

                self._session_ended_at = ended_at

                self._last_saved_file = saved_file

                self._running = False

                self._thread = None

                self._finalized_result = result

            print(
                f"AI attendance session finalized: "
                f"{session_id}"
            )

            print(
                f"Saved attendance file: "
                f"{saved_file}"
            )

            return result


# ======================================================================
# CLI / LOCAL TEST MODE
# ======================================================================

def main():
    """
    Existing manual CLI behavior.

    Run:

        python -m vision.video_ingestion.phone_attendance_pipeline_test

    Press Q in the OpenCV window to stop the session.
    """

    print("=" * 60)

    print(
        "Phone Camera Attendance Pipeline"
    )

    print("=" * 60)

    service = PhoneAttendancePipelineService()

    try:

        status = service.start(
            display=True
        )

        print(
            "\nInitial status:"
        )

        print(status)

        # --------------------------------------------------------------
        # Wait while worker thread is alive
        # --------------------------------------------------------------

        while service.is_loop_alive():

            time.sleep(0.5)

        # --------------------------------------------------------------
        # Finalize session
        # --------------------------------------------------------------

        result = service.stop()

        print(
            "\nFinal result:"
        )

        print(result)

    except KeyboardInterrupt:

        print(
            "\nKeyboard interrupt received."
        )

        try:

            result = service.stop()

            print(
                "\nFinal result:"
            )

            print(result)

        except Exception as exc:

            print(
                f"Failed to stop service: {exc}"
            )

    except Exception as exc:

        print(
            f"\nPipeline error: {exc}"
        )

        # If a session was successfully started,
        # attempt graceful cleanup.

        if service.is_running:

            try:

                service.stop()

            except Exception as stop_exc:

                print(
                    f"Cleanup error: {stop_exc}"
                )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
