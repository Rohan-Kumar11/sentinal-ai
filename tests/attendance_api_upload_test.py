import json
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/api/v1/attendance"

ATTENDANCE_DIRECTORY = Path(
    "experiments/attendance_sessions"
)


def find_latest_attendance_file() -> Path | None:
    """
    Find the newest attendance JSON file.
    """

    files = sorted(
        ATTENDANCE_DIRECTORY.glob(
            "attendance_*.json"
        )
    )

    if not files:
        return None

    return files[-1]


def main():
    print("=" * 60)
    print("SENTINAL - ATTENDANCE API UPLOAD TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Find latest attendance JSON
    # ---------------------------------------------------------

    attendance_file = find_latest_attendance_file()

    if attendance_file is None:
        print()
        print(
            "ERROR: No attendance JSON files found."
        )
        print(
            f"Expected directory: "
            f"{ATTENDANCE_DIRECTORY}"
        )
        return

    print()
    print("Latest attendance file:")
    print(attendance_file)

    # ---------------------------------------------------------
    # Load JSON
    # ---------------------------------------------------------

    try:
        with attendance_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            attendance_data = json.load(file)

    except Exception as exc:
        print()
        print(
            f"ERROR: Could not read attendance JSON: "
            f"{exc}"
        )
        return

    print()
    print("Attendance JSON loaded successfully.")

    print(
        f"Session ID: "
        f"{attendance_data.get('session_id')}"
    )

    print(
        f"Total tracked: "
        f"{attendance_data.get('total_tracked')}"
    )

    # ---------------------------------------------------------
    # Send attendance data to FastAPI
    # ---------------------------------------------------------

    print()
    print("Sending attendance data to FastAPI...")
    print(f"API: {API_URL}")

    try:
        response = requests.post(
            API_URL,
            json=attendance_data,
            timeout=10,
        )

    except requests.RequestException as exc:
        print()
        print(
            "ERROR: Could not connect to FastAPI."
        )
        print(f"Details: {exc}")
        return

    # ---------------------------------------------------------
    # Process response
    # ---------------------------------------------------------

    print()
    print(
        f"HTTP status: {response.status_code}"
    )

    try:
        response_data = response.json()

    except ValueError:
        print()
        print(
            "ERROR: FastAPI returned invalid JSON."
        )
        print(response.text)
        return

    if response.status_code != 200:
        print()
        print("API request failed.")
        print()
        print(
            json.dumps(
                response_data,
                indent=2,
            )
        )
        return

    # ---------------------------------------------------------
    # Successful response
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("ATTENDANCE API UPLOAD SUCCESSFUL")
    print("=" * 60)

    print()
    print("API response:")

    print(
        json.dumps(
            response_data,
            indent=2,
        )
    )

    print()
    print("=" * 60)
    print("STEP 7 TEST FINISHED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()