
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


@dataclass
class AttendanceRecord:
    track_id: int
    role: str
    first_seen: str
    last_seen: str
    total_observed_seconds: float
    observations: int


class AttendanceEngine:
    """
    Session-based attendance engine.

    IMPORTANT:
    - Track IDs are temporary IDs created by ByteTrack.
    - They are NOT permanent identities.
    - Role classification is currently MOCK.
    - This is a development/prototype attendance system.
    """

    def __init__(self):
        self.records: Dict[int, AttendanceRecord] = {}
        self.previous_seen: Dict[int, float] = {}

    @staticmethod
    def format_timestamp(timestamp: datetime) -> str:
        """
        Convert a datetime object into an ISO 8601 UTC timestamp.
        """
        return timestamp.astimezone(timezone.utc).isoformat()

    def update(
        self,
        track_id: int,
        role: str,
        current_time: float,
        current_timestamp: str,
    ) -> None:
        """
        Update attendance information for one tracked person.

        The role classifier may return either:
        - a string, e.g. "Staff"
        - a dictionary containing a "role" field

        AttendanceRecord always stores the normalized role string.
        """

        # Normalize role classifier output.
        #
        # Current mock classifier returns something like:
        # {
        #     "track_id": 1,
        #     "role": "Staff",
        #     "confidence": 1.0,
        #     "classifier_type": "MOCK"
        # }
        #
        # AttendanceRecord only needs the actual role name.
        if isinstance(role, dict):
            role = role.get("role", "Unknown")

        # Make sure the stored value is always a string.
        if not isinstance(role, str):
            role = str(role)

        if track_id not in self.records:
            self.records[track_id] = AttendanceRecord(
                track_id=track_id,
                role=role,
                first_seen=current_timestamp,
                last_seen=current_timestamp,
                total_observed_seconds=0.0,
                observations=1,
            )

            self.previous_seen[track_id] = current_time
            return

        record = self.records[track_id]

        previous_time = self.previous_seen.get(
            track_id,
            current_time,
        )

        elapsed = current_time - previous_time

        # Protect against unexpected negative timing values.
        if elapsed < 0:
            elapsed = 0.0

        record.last_seen = current_timestamp
        record.total_observed_seconds += elapsed
        record.observations += 1

        # Keep the latest known role.
        record.role = role

        self.previous_seen[track_id] = current_time

    def get_records(self) -> List[dict]:
        """
        Return attendance records as JSON-compatible dictionaries.
        """

        records = []

        for record in self.records.values():
            records.append(
                {
                    "track_id": record.track_id,
                    "role": record.role,
                    "first_seen": record.first_seen,
                    "last_seen": record.last_seen,
                    "duration_seconds": round(
                        record.total_observed_seconds,
                        2,
                    ),
                    "observations": record.observations,
                }
            )

        return records

    def get_session_data(
        self,
        session_id: str,
        session_started_at: str,
        session_ended_at: str,
    ) -> dict:
        """
        Build the complete API-ready attendance session object.
        """

        staff_count = 0
        beneficiary_count = 0
        unknown_count = 0

        for record in self.records.values():
            if record.role == "Staff":
                staff_count += 1
            elif record.role == "Beneficiary":
                beneficiary_count += 1
            else:
                unknown_count += 1

        return {
            "session_id": session_id,
            "session_started_at": session_started_at,
            "session_ended_at": session_ended_at,
            "total_tracked": len(self.records),
            "staff": staff_count,
            "beneficiary": beneficiary_count,
            "unknown": unknown_count,
            "records": self.get_records(),
        }

    def print_summary(self) -> None:
        """
        Print a readable attendance summary.
        """

        print()
        print("=" * 70)
        print("ATTENDANCE SESSION SUMMARY")
        print("=" * 70)

        if not self.records:
            print("No persons were tracked during this session.")
            return

        for record in self.records.values():
            print(
                f"Track ID {record.track_id:<3} | "
                f"Role: {record.role:<11} | "
                f"Observed: {record.total_observed_seconds:>7.1f}s | "
                f"Observations: {record.observations}"
            )

    def save_session_data(
        self,
        session_id: str,
        session_started_at: str,
        session_ended_at: str,
        output_directory: str = "experiments/attendance_sessions",
    ) -> Path:
        """
        Save attendance session data to a JSON file.
        """

        output_path = Path(output_directory)

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        session_data = self.get_session_data(
            session_id=session_id,
            session_started_at=session_started_at,
            session_ended_at=session_ended_at,
        )

        file_path = (
            output_path
            / f"attendance_{session_id}.json"
        )

        with file_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                session_data,
                file,
                indent=2,
            )

        return file_path

