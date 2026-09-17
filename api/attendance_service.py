from vision.video_ingestion.phone_attendance_pipeline_test import (
    PhoneAttendancePipelineService,
)


class AttendanceService:
    """
    Application-level controller for the AI attendance pipeline.

    Lifecycle:

        START
          ↓
        LIVE MONITORING
          ↓
        STOP
          ↓
        FINALIZE + SAVE
          ↓
        RESTART
          ↓
        NEW SESSION
    """

    def __init__(self):
        self.pipeline = PhoneAttendancePipelineService(
            camera_name="phone"
        )

    def start(self) -> dict:
        if self.pipeline.is_running:
            return {
                "success": False,
                "message": "AI attendance is already running.",
                "status": self.pipeline.get_status(),
            }

        try:
            status = self.pipeline.start(display=False)

            return {
                "success": True,
                "message": "AI attendance started.",
                "status": status,
            }

        except Exception as exc:
            return {
                "success": False,
                "message": f"Failed to start AI attendance: {exc}",
                "status": self.pipeline.get_status(),
            }

    def stop(self) -> dict:
        if not self.pipeline.is_running:
            return {
                "success": False,
                "message": "AI attendance is not running.",
                "status": self.pipeline.get_status(),
            }

        try:
            result = self.pipeline.stop()

            return {
                "success": True,
                "message": "AI attendance stopped and session finalized.",
                "result": result,
                "status": self.pipeline.get_status(),
            }

        except Exception as exc:
            return {
                "success": False,
                "message": f"Failed to stop AI attendance: {exc}",
                "status": self.pipeline.get_status(),
            }

    def restart(self) -> dict:
        """
        Finalize the current session and immediately start
        a completely new AI attendance session.

        The previous session remains permanently saved.
        """

        previous_session = None

        if self.pipeline.is_running:
            try:
                stop_result = self.pipeline.stop()
                previous_session = stop_result

            except Exception as exc:
                return {
                    "success": False,
                    "message": (
                        "Could not finalize current AI attendance session: "
                        f"{exc}"
                    ),
                    "previous_session": None,
                    "new_session": None,
                }

        try:
            new_status = self.pipeline.start(display=False)

            return {
                "success": True,
                "message": (
                    "AI attendance restarted. "
                    "A new monitoring session is now active."
                ),
                "previous_session": previous_session,
                "new_session": new_status,
            }

        except Exception as exc:
            return {
                "success": False,
                "message": (
                    "Previous AI session was finalized, "
                    "but the new session could not start: "
                    f"{exc}"
                ),
                "previous_session": previous_session,
                "new_session": None,
                "status": self.pipeline.get_status(),
            }

    def status(self) -> dict:
        return {
            "success": True,
            "status": self.pipeline.get_status(),
        }

    def active_count(self) -> dict:
        status = self.pipeline.get_status()

        # Current/live attendance must only represent
        # an actively running AI monitoring session.
        if status["running"] and status["loop_alive"]:
            active_count = self.pipeline.get_active_count()
        else:
            active_count = 0

        return {
            "success": True,
            "active_count": active_count,
            "running": status["running"],
            "loop_alive": status["loop_alive"],
            "session_id": status["session_id"],
        }


attendance_service = AttendanceService()