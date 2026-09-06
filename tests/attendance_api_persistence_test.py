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
            "No attendance JSON files found in experiments/attendance_sessions."
        )

    return files[0]


def main():
    print("=" * 60)
    print("SENTINAL - ATTENDANCE API PERSISTENCE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Find latest attendance JSON
    # ---------------------------------------------------------
    latest_file = find_latest_attendance_file()

    print()
    print("Latest attendance file:")
    print(latest_file)

    # ---------------------------------------------------------
    # 2. Load attendance JSON
    # ---------------------------------------------------------
    with latest_file.open("r", encoding="utf-8") as file:
        attendance_data = json.load(file)

    session_id = attendance_data["session_id"]

    print()
    print("Attendance JSON loaded successfully.")
    print(f"Session ID: {session_id}")
    print(f"Total tracked: {attendance_data['total_tracked']}")

    # ---------------------------------------------------------
    # 3. POST attendance session
    # ---------------------------------------------------------
    post_url = f"{API_BASE_URL}/api/v1/attendance"

    print()
    print("Sending attendance data to FastAPI...")
    print(f"API: {post_url}")

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
    # 4. Request latest attendance through GET API
    # ---------------------------------------------------------
    latest_url = f"{API_BASE_URL}/api/v1/attendance/latest"

    print()
    print("Requesting latest attendance from GET API...")
    print(f"API: {latest_url}")

    get_response = requests.get(
        latest_url,
        timeout=10,
    )

    print()
    print(f"GET HTTP status: {get_response.status_code}")

    if get_response.status_code != 200:
        print()
        print("GET FAILED")
        print(get_response.text)
        return

    latest_data = get_response.json()

    latest_session_id = latest_data.get("session_id")

    print()
    print("Latest API session:")
    print(f"Session ID: {latest_session_id}")

    # ---------------------------------------------------------
    # 5. Compare uploaded session with latest API session
    # ---------------------------------------------------------
    print()
    print("Comparing session IDs...")

    if latest_session_id == session_id:
        print()
        print("=" * 60)
        print("PERSISTENCE VERIFICATION SUCCESSFUL")
        print("=" * 60)
        print()
        print(
            "The uploaded attendance session is available through "
            "GET /api/v1/attendance/latest."
        )
    else:
        print()
        print("=" * 60)
        print("PERSISTENCE VERIFICATION RESULT")
        print("=" * 60)
        print()
        print("The POST request succeeded, but the latest GET session")
        print("does not match the uploaded session.")
        print()
        print(f"Uploaded session : {session_id}")
        print(f"Latest GET session: {latest_session_id}")


if __name__ == "__main__":
    main()