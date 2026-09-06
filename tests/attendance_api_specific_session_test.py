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
            "No attendance JSON files found in "
            "experiments/attendance_sessions."
        )

    return files[0]


def main():
    print("=" * 60)
    print("SENTINAL - SPECIFIC ATTENDANCE SESSION API TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Find attendance JSON
    # ---------------------------------------------------------

    latest_file = find_latest_attendance_file()

    print()
    print("Attendance file selected:")
    print(latest_file)

    # ---------------------------------------------------------
    # 2. Load JSON
    # ---------------------------------------------------------

    with latest_file.open("r", encoding="utf-8") as file:
        attendance_data = json.load(file)

    session_id = attendance_data["session_id"]

    print()
    print("Attendance JSON loaded successfully.")
    print(f"Session ID: {session_id}")
    print(f"Total tracked: {attendance_data['total_tracked']}")

    # ---------------------------------------------------------
    # 3. POST attendance
    # ---------------------------------------------------------

    post_url = f"{API_BASE_URL}/api/v1/attendance"

    print()
    print("Sending attendance session to FastAPI...")
    print(f"POST: {post_url}")

    post_response = requests.post(
        post_url,
        json=attendance_data,
        timeout=10,
    )

    print()
    print(f"POST HTTP status: {post_response.status_code}")

    if post_response.status_code != 200:
        print()
        print("POST FAILED")
        print(post_response.text)
        return

    print("POST successful.")

    # ---------------------------------------------------------
    # 4. GET the exact session
    # ---------------------------------------------------------

    get_url = (
        f"{API_BASE_URL}/api/v1/attendance/"
        f"{session_id}"
    )

    print()
    print("Requesting the exact uploaded session...")
    print(f"GET: {get_url}")

    get_response = requests.get(
        get_url,
        timeout=10,
    )

    print()
    print(f"GET HTTP status: {get_response.status_code}")

    if get_response.status_code != 200:
        print()
        print("GET SPECIFIC SESSION FAILED")
        print(get_response.text)
        return

    returned_data = get_response.json()

    returned_session_id = returned_data.get(
        "session_id"
    )

    returned_total_tracked = returned_data.get(
        "total_tracked"
    )

    # ---------------------------------------------------------
    # 5. Verify returned session
    # ---------------------------------------------------------

    print()
    print("Returned session:")
    print(f"Session ID: {returned_session_id}")
    print(f"Total tracked: {returned_total_tracked}")

    print()
    print("Verifying persisted data...")

    if (
        returned_session_id == session_id
        and returned_total_tracked
        == attendance_data["total_tracked"]
    ):
        print()
        print("=" * 60)
        print("SPECIFIC SESSION PERSISTENCE SUCCESSFUL")
        print("=" * 60)
        print()
        print(
            "The attendance session was successfully "
            "saved and retrieved through the API."
        )
    else:
        print()
        print("=" * 60)
        print("SPECIFIC SESSION VERIFICATION FAILED")
        print("=" * 60)
        print()
        print(
            f"Expected session ID: {session_id}"
        )
        print(
            f"Returned session ID: {returned_session_id}"
        )
        print(
            f"Expected total tracked: "
            f"{attendance_data['total_tracked']}"
        )
        print(
            f"Returned total tracked: "
            f"{returned_total_tracked}"
        )


if __name__ == "__main__":
    main()