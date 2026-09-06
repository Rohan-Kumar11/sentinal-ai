import json
from pathlib import Path

import requests


API_BASE_URL = "http://127.0.0.1:8000"
ATTENDANCE_DIRECTORY = Path("experiments") / "attendance_sessions"


def find_latest_attendance_file():
    files = sorted(
        ATTENDANCE_DIRECTORY.glob("attendance_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            "No attendance JSON files found."
        )

    return files[0]


def check_get_endpoint(
    name,
    url,
):
    print()
    print("-" * 60)
    print(name)
    print("-" * 60)
    print(f"GET: {url}")

    response = requests.get(
        url,
        timeout=10,
    )

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        print("FAILED")
        print(response.text)
        return None

    print("SUCCESS")

    return response.json()


def main():
    print("=" * 60)
    print("SENTINAL - FULL ATTENDANCE API TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Find a real attendance session
    # ---------------------------------------------------------

    latest_file = find_latest_attendance_file()

    with latest_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        attendance_data = json.load(file)

    session_id = attendance_data["session_id"]

    print()
    print("Using attendance session:")
    print(f"File: {latest_file}")
    print(f"Session ID: {session_id}")
    print(
        f"Total tracked: "
        f"{attendance_data['total_tracked']}"
    )

    # ---------------------------------------------------------
    # 2. GET all attendance sessions
    # ---------------------------------------------------------

    all_sessions = check_get_endpoint(
        "ALL ATTENDANCE SESSIONS",
        f"{API_BASE_URL}/api/v1/attendance",
    )

    if all_sessions is None:
        return

    print(
        f"Sessions returned: {len(all_sessions)}"
    )

    # ---------------------------------------------------------
    # 3. GET latest attendance
    # ---------------------------------------------------------

    latest_session = check_get_endpoint(
        "LATEST ATTENDANCE",
        f"{API_BASE_URL}/api/v1/attendance/latest",
    )

    if latest_session is None:
        return

    print(
        f"Latest session ID: "
        f"{latest_session.get('session_id')}"
    )

    # ---------------------------------------------------------
    # 4. GET overall attendance summary
    # ---------------------------------------------------------

    summary = check_get_endpoint(
        "OVERALL ATTENDANCE SUMMARY",
        f"{API_BASE_URL}/api/v1/attendance/summary",
    )

    if summary is None:
        return

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

    print(
        f"Total observed seconds: "
        f"{summary.get('total_observed_seconds')}"
    )

    # ---------------------------------------------------------
    # 5. GET overall role statistics
    # ---------------------------------------------------------

    role_statistics = check_get_endpoint(
        "OVERALL ROLE STATISTICS",
        f"{API_BASE_URL}/api/v1/attendance/"
        "role-statistics",
    )

    if role_statistics is None:
        return

    print(
        f"Total tracked: "
        f"{role_statistics.get('total_tracked')}"
    )

    for statistic in role_statistics.get(
        "statistics",
        [],
    ):
        print(
            f"Role: {statistic.get('role')} | "
            f"People: {statistic.get('people')} | "
            f"Observed seconds: "
            f"{statistic.get('observed_seconds')} | "
            f"Percentage: "
            f"{statistic.get('percentage_of_people')}%"
        )

    # ---------------------------------------------------------
    # 6. GET specific attendance session
    # ---------------------------------------------------------

    specific_session = check_get_endpoint(
        "SPECIFIC ATTENDANCE SESSION",
        f"{API_BASE_URL}/api/v1/attendance/"
        f"{session_id}",
    )

    if specific_session is None:
        return

    print(
        f"Session ID: "
        f"{specific_session.get('session_id')}"
    )

    print(
        f"Total tracked: "
        f"{specific_session.get('total_tracked')}"
    )

    print(
        f"Records: "
        f"{len(specific_session.get('records', []))}"
    )

    # ---------------------------------------------------------
    # 7. GET specific session role statistics
    # ---------------------------------------------------------

    session_role_statistics = check_get_endpoint(
        "SPECIFIC SESSION ROLE STATISTICS",
        f"{API_BASE_URL}/api/v1/attendance/"
        f"{session_id}/role-statistics",
    )

    if session_role_statistics is None:
        return

    print(
        f"Session ID: "
        f"{session_role_statistics.get('session_id')}"
    )

    print(
        f"Total tracked: "
        f"{session_role_statistics.get('total_tracked')}"
    )

    for statistic in session_role_statistics.get(
        "statistics",
        [],
    ):
        print(
            f"Role: {statistic.get('role')} | "
            f"People: {statistic.get('people')} | "
            f"Observed seconds: "
            f"{statistic.get('observed_seconds')} | "
            f"Percentage: "
            f"{statistic.get('percentage_of_people')}%"
        )

    # ---------------------------------------------------------
    # 8. Final verification
    # ---------------------------------------------------------

    checks = [
        all_sessions is not None,
        latest_session is not None,
        summary is not None,
        role_statistics is not None,
        specific_session is not None,
        session_role_statistics is not None,
        specific_session.get("session_id")
        == session_id,
        session_role_statistics.get("session_id")
        == session_id,
    ]

    print()
    print("=" * 60)

    if all(checks):
        print("FULL ATTENDANCE API TEST SUCCESSFUL")
    else:
        print("FULL ATTENDANCE API TEST FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()