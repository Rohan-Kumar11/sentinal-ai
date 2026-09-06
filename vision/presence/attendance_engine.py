import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
from ultralytics import YOLO

from vision.person_classification.mock_role_classifier import classify_role


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
        """

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


def main():
    print("=" * 70)
    print("SENTINAL - ATTENDANCE ENGINE")
    print("=" * 70)

    print("Pipeline:")
    print("  YOLO → ByteTrack → Mock Role → Attendance Engine")

    print()
    print("IMPORTANT:")
    print("  Track IDs are temporary session IDs.")
    print("  This is NOT identity-based attendance.")
    print("  Role classification is currently MOCK.")

    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    session_started_at = (
        datetime.now(timezone.utc).isoformat()
    )

    model_name = "yolo11n.pt"
    confidence = 0.5

    print()
    print(f"Session ID : {session_id}")
    print(f"Started    : {session_started_at}")
    print(f"Model      : {model_name}")
    print(f"Confidence : {confidence}")
    print()
    print("Press Q to quit.")

    model = YOLO(model_name)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print()
        print("ERROR: Could not open webcam.")
        return

    engine = AttendanceEngine()

    frame_count = 0
    start_time = time.perf_counter()
    last_status_time = start_time

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read frame.")
                break

            frame_count += 1

            current_time = time.perf_counter()

            current_timestamp = (
                datetime.now(timezone.utc).isoformat()
            )

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                classes=[0],
                verbose=False,
            )

            active_track_ids = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                if boxes.id is not None:
                    track_ids = (
                        boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

                    xyxy = (
                        boxes.xyxy
                        .cpu()
                        .tolist()
                    )

                    for track_id, box in zip(
                        track_ids,
                        xyxy,
                    ):
                        track_id = int(track_id)

                        active_track_ids.append(
                            track_id
                        )

                        role_data = classify_role(
                            track_id
                        )

                        role = role_data["role"]

                        engine.update(
                            track_id=track_id,
                            role=role,
                            current_time=current_time,
                            current_timestamp=current_timestamp,
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box,
                        )

                        label = (
                            f"ID {track_id} | "
                            f"{role}"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            label,
                            (
                                x1,
                                max(y1 - 10, 20),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

            elapsed = (
                current_time - start_time
            )

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0.0
            )

            cv2.putText(
                frame,
                f"Session: {session_id}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Tracked: {len(engine.records)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Sentinal - Attendance Engine",
                frame,
            )

            if current_time - last_status_time >= 5:
                staff = sum(
                    1
                    for r in engine.records.values()
                    if r.role == "Staff"
                )

                beneficiary = sum(
                    1
                    for r in engine.records.values()
                    if r.role == "Beneficiary"
                )

                unknown = sum(
                    1
                    for r in engine.records.values()
                    if r.role == "Unknown"
                )

                print(
                    f"STATUS | "
                    f"Frame {frame_count} | "
                    f"Active {len(active_track_ids)} | "
                    f"Records {len(engine.records)} | "
                    f"Staff {staff} | "
                    f"Beneficiary {beneficiary} | "
                    f"Unknown {unknown}"
                )

                last_status_time = current_time

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print()
                print(
                    f"Quit requested after "
                    f"{frame_count} frame(s)."
                )
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    session_ended_at = (
        datetime.now(timezone.utc).isoformat()
    )

    engine.print_summary()

    session_data = engine.get_session_data(
        session_id=session_id,
        session_started_at=session_started_at,
        session_ended_at=session_ended_at,
    )

    print()
    print("=" * 70)
    print("API-READY ATTENDANCE DATA")
    print("=" * 70)

    print(
        json.dumps(
            session_data,
            indent=2,
        )
    )

    saved_file = engine.save_session_data(
        session_id=session_id,
        session_started_at=session_started_at,
        session_ended_at=session_ended_at,
    )

    print()
    print("=" * 70)
    print("ATTENDANCE DATA SAVED")
    print("=" * 70)
    print(f"File: {saved_file}")
    print()


if __name__ == "__main__":
    main()