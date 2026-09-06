import requests


class SentinalApiClient:
    """
    Simple client for communicating with the
    Sentinal FastAPI backend.
    """

    def __init__(
        self,
        base_url="http://127.0.0.1:8000",
        timeout=10,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path):
        url = f"{self.base_url}{path}"

        response = requests.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def _post(self, path, data):
        url = f"{self.base_url}{path}"

        response = requests.post(
            url,
            json=data,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def health(self):
        """
        Check whether the FastAPI backend is healthy.
        """
        return self._get("/health")

    def get_all_attendance(self):
        """
        Get all stored attendance sessions.
        """
        return self._get(
            "/api/v1/attendance"
        )

    def get_latest_attendance(self):
        """
        Get the latest attendance session.
        """
        return self._get(
            "/api/v1/attendance/latest"
        )

    def get_attendance_summary(self):
        """
        Get overall attendance summary.
        """
        return self._get(
            "/api/v1/attendance/summary"
        )

    def get_role_statistics(self):
        """
        Get overall role statistics.
        """
        return self._get(
            "/api/v1/attendance/role-statistics"
        )

    def get_attendance_session(self, session_id):
        """
        Get one specific attendance session.
        """
        return self._get(
            f"/api/v1/attendance/{session_id}"
        )

    def get_session_role_statistics(
        self,
        session_id,
    ):
        """
        Get role statistics for one
        specific attendance session.
        """
        return self._get(
            f"/api/v1/attendance/"
            f"{session_id}/role-statistics"
        )

    def upload_attendance(self, attendance_data):
        """
        Upload an attendance session to
        the FastAPI backend.
        """
        return self._post(
            "/api/v1/attendance",
            attendance_data,
        )


def main():
    print("=" * 60)
    print("SENTINAL API CLIENT TEST")
    print("=" * 60)

    client = SentinalApiClient()

    # ---------------------------------------------------------
    # Health check
    # ---------------------------------------------------------

    health = client.health()

    print()
    print("Backend health:")
    print(health)

    # ---------------------------------------------------------
    # Latest attendance
    # ---------------------------------------------------------

    latest = client.get_latest_attendance()

    print()
    print("Latest attendance:")
    print(
        f"Session ID: "
        f"{latest.get('session_id')}"
    )

    print(
        f"Total tracked: "
        f"{latest.get('total_tracked')}"
    )

    # ---------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------

    summary = client.get_attendance_summary()

    print()
    print("Attendance summary:")
    print(
        f"Total sessions: "
        f"{summary.get('total_sessions')}"
    )

    print(
        f"Total tracked: "
        f"{summary.get('total_tracked')}"
    )

    print(
        f"Total staff: "
        f"{summary.get('total_staff')}"
    )

    print(
        f"Total beneficiary: "
        f"{summary.get('total_beneficiary')}"
    )

    print(
        f"Total unknown: "
        f"{summary.get('total_unknown')}"
    )

    # ---------------------------------------------------------
    # Role statistics
    # ---------------------------------------------------------

    role_statistics = client.get_role_statistics()

    print()
    print("Role statistics:")

    for statistic in role_statistics.get(
        "statistics",
        [],
    ):
        print(
            f"{statistic.get('role')}: "
            f"{statistic.get('people')} people"
        )

    print()
    print("=" * 60)
    print("API CLIENT TEST SUCCESSFUL")
    print("=" * 60)


if __name__ == "__main__":
    main()
    