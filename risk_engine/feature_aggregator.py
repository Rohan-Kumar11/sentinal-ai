from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RiskFeatures:
    """
    Normalized features used by the anomaly detector
    and risk engine.

    All ratio/score features are represented on a 0.0-1.0
    scale unless explicitly documented otherwise.
    """

    # ---------------------------------------------------------
    # Attendance features
    # ---------------------------------------------------------

    attendance_total_tracked: int = 0
    attendance_staff_count: int = 0
    attendance_beneficiary_count: int = 0
    attendance_unknown_count: int = 0

    attendance_staff_ratio: float = 0.0
    attendance_beneficiary_ratio: float = 0.0
    attendance_unknown_ratio: float = 0.0

    attendance_duration_seconds: float = 0.0
    attendance_observation_count: int = 0

    # ---------------------------------------------------------
    # Project / application features
    # ---------------------------------------------------------

    project_status: str = "unknown"
    project_risk_level: str = "low"

    project_total_inspections: int = 0
    project_completed_inspections: int = 0
    project_pending_inspections: int = 0

    project_high_risk_findings: int = 0

    inspection_completion_ratio: float = 0.0
    pending_inspection_ratio: float = 0.0
    high_risk_finding_ratio: float = 0.0

    # ---------------------------------------------------------
    # Inspection history features
    # ---------------------------------------------------------

    inspection_history_count: int = 0
    inspection_high_risk_count: int = 0
    inspection_medium_risk_count: int = 0
    inspection_low_risk_count: int = 0

    inspection_high_risk_ratio: float = 0.0
    inspection_medium_risk_ratio: float = 0.0

    open_finding_count: int = 0
    repeated_issue_count: int = 0

    # ---------------------------------------------------------
    # Overall derived indicators
    # ---------------------------------------------------------

    attendance_anomaly_signal: float = 0.0
    inspection_anomaly_signal: float = 0.0
    project_anomaly_signal: float = 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.
    """
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Calculate a ratio safely and clamp it to [0.0, 1.0].
    """
    if denominator <= 0:
        return 0.0

    ratio = numerator / denominator

    return max(
        0.0,
        min(1.0, ratio),
    )


def _normalize_risk_level(value: Any) -> str:
    """
    Normalize project/inspection risk labels.
    """
    if value is None:
        return "low"

    normalized = str(value).strip().lower()

    if normalized in {
        "critical",
        "high",
        "medium",
        "low",
    }:
        return normalized

    return "low"


def _normalize_status(value: Any) -> str:
    """
    Normalize project status.
    """
    if value is None:
        return "unknown"

    normalized = str(value).strip().lower()

    return normalized if normalized else "unknown"


def _normalize_role(value: Any) -> str:
    """
    Normalize attendance role labels.

    The real attendance pipeline currently emits:
        Staff
        Beneficiary
        Unknown

    Risk Engine comparisons are therefore performed against
    normalized lowercase values.
    """
    if value is None:
        return "unknown"

    normalized = str(value).strip().lower()

    if normalized in {
        "staff",
        "beneficiary",
        "unknown",
    }:
        return normalized

    return "unknown"


def _calculate_attendance_features(
    attendance: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert an attendance session into normalized features.

    Compatible with the real AttendanceEngine session output:

    {
        "session_id": "...",
        "session_started_at": "...",
        "session_ended_at": "...",
        "total_tracked": 10,
        "staff": 2,
        "beneficiary": 7,
        "unknown": 1,
        "records": [
            {
                "track_id": 1,
                "role": "Staff",
                "first_seen": "...",
                "last_seen": "...",
                "duration_seconds": 120.0,
                "observations": 30
            }
        ]
    }

    The Risk Engine does not modify the attendance pipeline.
    It only normalizes its output at the integration boundary.
    """

    if not attendance:
        return {
            "attendance_total_tracked": 0,
            "attendance_staff_count": 0,
            "attendance_beneficiary_count": 0,
            "attendance_unknown_count": 0,
            "attendance_staff_ratio": 0.0,
            "attendance_beneficiary_ratio": 0.0,
            "attendance_unknown_ratio": 0.0,
            "attendance_duration_seconds": 0.0,
            "attendance_observation_count": 0,
        }

    records = attendance.get("records") or []

    if not isinstance(records, list):
        records = []

    # ---------------------------------------------------------
    # Summary counts
    # ---------------------------------------------------------

    total_tracked = _safe_int(
        attendance.get("total_tracked"),
    )

    staff_count = _safe_int(
        attendance.get("staff"),
    )

    beneficiary_count = _safe_int(
        attendance.get("beneficiary"),
    )

    unknown_count = _safe_int(
        attendance.get("unknown"),
    )

    # ---------------------------------------------------------
    # Derive counts from real records when summary values
    # are missing or zero.
    # ---------------------------------------------------------

    normalized_roles = [
        _normalize_role(
            record.get("role")
        )
        for record in records
        if isinstance(record, dict)
    ]

    if total_tracked <= 0:
        total_tracked = len(normalized_roles)

    if staff_count <= 0:
        staff_count = sum(
            1
            for role in normalized_roles
            if role == "staff"
        )

    if beneficiary_count <= 0:
        beneficiary_count = sum(
            1
            for role in normalized_roles
            if role == "beneficiary"
        )

    if unknown_count <= 0:
        unknown_count = sum(
            1
            for role in normalized_roles
            if role == "unknown"
        )

    # ---------------------------------------------------------
    # Ratio denominator
    #
    # Prefer the declared total_tracked value, but ensure that
    # the denominator cannot be smaller than the role counts.
    # ---------------------------------------------------------

    total_for_ratio = max(
        total_tracked,
        staff_count + beneficiary_count + unknown_count,
    )

    # ---------------------------------------------------------
    # Duration
    #
    # Real AttendanceEngine output stores duration per record.
    # Some future API adapters may provide a summary-level value.
    # Support both.
    # ---------------------------------------------------------

    duration_seconds = _safe_float(
        attendance.get("duration_seconds"),
    )

    if duration_seconds <= 0:
        duration_seconds = _safe_float(
            attendance.get("total_duration_seconds"),
        )

    if duration_seconds <= 0:
        duration_seconds = sum(
            _safe_float(
                record.get("duration_seconds")
            )
            for record in records
            if isinstance(record, dict)
        )

    # ---------------------------------------------------------
    # Observation count
    #
    # Real AttendanceEngine output stores observations per record.
    # ---------------------------------------------------------

    observation_count = _safe_int(
        attendance.get("observation_count"),
    )

    if observation_count <= 0:
        observation_count = sum(
            _safe_int(
                record.get("observations")
            )
            for record in records
            if isinstance(record, dict)
        )

    return {
        "attendance_total_tracked": total_tracked,
        "attendance_staff_count": staff_count,
        "attendance_beneficiary_count": beneficiary_count,
        "attendance_unknown_count": unknown_count,
        "attendance_staff_ratio": _safe_ratio(
            staff_count,
            total_for_ratio,
        ),
        "attendance_beneficiary_ratio": _safe_ratio(
            beneficiary_count,
            total_for_ratio,
        ),
        "attendance_unknown_ratio": _safe_ratio(
            unknown_count,
            total_for_ratio,
        ),
        "attendance_duration_seconds": duration_seconds,
        "attendance_observation_count": observation_count,
    }


def _calculate_project_features(
    project: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convert project/application information into
    normalized risk features.
    """

    if not project:
        return {
            "project_status": "unknown",
            "project_risk_level": "low",
            "project_total_inspections": 0,
            "project_completed_inspections": 0,
            "project_pending_inspections": 0,
            "project_high_risk_findings": 0,
            "inspection_completion_ratio": 0.0,
            "pending_inspection_ratio": 0.0,
            "high_risk_finding_ratio": 0.0,
        }

    total_inspections = _safe_int(
        project.get("total_inspections"),
    )

    completed_inspections = _safe_int(
        project.get("completed_inspections"),
    )

    pending_inspections = _safe_int(
        project.get("pending_inspections"),
    )

    high_risk_findings = _safe_int(
        project.get("high_risk_findings"),
    )

    return {
        "project_status": _normalize_status(
            project.get("status"),
        ),
        "project_risk_level": _normalize_risk_level(
            project.get("risk_level"),
        ),
        "project_total_inspections": total_inspections,
        "project_completed_inspections": completed_inspections,
        "project_pending_inspections": pending_inspections,
        "project_high_risk_findings": high_risk_findings,
        "inspection_completion_ratio": _safe_ratio(
            completed_inspections,
            total_inspections,
        ),
        "pending_inspection_ratio": _safe_ratio(
            pending_inspections,
            total_inspections,
        ),
        "high_risk_finding_ratio": _safe_ratio(
            high_risk_findings,
            total_inspections,
        ),
    }


def _calculate_inspection_features(
    inspections: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Convert inspection history and findings into
    normalized features.

    Each inspection may contain a risk_level and optionally
    finding information.
    """

    inspections = inspections or []

    history_count = len(inspections)

    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0

    open_finding_count = 0
    repeated_issue_count = 0

    issue_titles: Dict[str, int] = {}

    for inspection in inspections:
        if not isinstance(inspection, dict):
            continue

        risk_level = _normalize_risk_level(
            inspection.get("risk_level"),
        )

        if risk_level == "high":
            high_risk_count += 1
        elif risk_level == "medium":
            medium_risk_count += 1
        else:
            low_risk_count += 1

        findings = inspection.get("findings") or []

        if isinstance(findings, list):
            for finding in findings:
                if not isinstance(finding, dict):
                    continue

                status = str(
                    finding.get("status", "open")
                ).strip().lower()

                if status in {
                    "open",
                    "pending",
                    "unresolved",
                }:
                    open_finding_count += 1

                title = str(
                    finding.get("title", "")
                ).strip().lower()

                if title:
                    issue_titles[title] = (
                        issue_titles.get(title, 0) + 1
                    )

        # Some callers may already provide finding counts.
        open_finding_count += _safe_int(
            inspection.get("open_finding_count"),
        )

        repeated_issue_count += _safe_int(
            inspection.get("repeated_issue_count"),
        )

    for count in issue_titles.values():
        if count > 1:
            repeated_issue_count += count - 1

    return {
        "inspection_history_count": history_count,
        "inspection_high_risk_count": high_risk_count,
        "inspection_medium_risk_count": medium_risk_count,
        "inspection_low_risk_count": low_risk_count,
        "inspection_high_risk_ratio": _safe_ratio(
            high_risk_count,
            history_count,
        ),
        "inspection_medium_risk_ratio": _safe_ratio(
            medium_risk_count,
            history_count,
        ),
        "open_finding_count": open_finding_count,
        "repeated_issue_count": repeated_issue_count,
    }


def _calculate_signal_features(
    features: RiskFeatures,
) -> Dict[str, float]:
    """
    Create explainable anomaly signals.

    These are NOT final risk scores.

    They are intermediate indicators that the anomaly
    detector can evaluate.
    """

    attendance_anomaly_signal = min(
        1.0,
        features.attendance_unknown_ratio * 1.5,
    )

    inspection_anomaly_signal = min(
        1.0,
        (
            features.inspection_high_risk_ratio * 0.5
            + min(
                1.0,
                features.open_finding_count / 10.0,
            ) * 0.3
            + min(
                1.0,
                features.repeated_issue_count / 5.0,
            ) * 0.2
        ),
    )

    project_anomaly_signal = min(
        1.0,
        (
            features.pending_inspection_ratio * 0.5
            + features.high_risk_finding_ratio * 0.5
        ),
    )

    return {
        "attendance_anomaly_signal": attendance_anomaly_signal,
        "inspection_anomaly_signal": inspection_anomaly_signal,
        "project_anomaly_signal": project_anomaly_signal,
    }


def aggregate_features(
    attendance: Optional[Dict[str, Any]] = None,
    project: Optional[Dict[str, Any]] = None,
    inspections: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Aggregate attendance, project, and inspection data.

    Returns a JSON-serializable dictionary that can later be
    passed to the anomaly detector and risk engine.

    This function intentionally does NOT calculate the final
    risk score.
    """

    attendance_features = _calculate_attendance_features(
        attendance,
    )

    project_features = _calculate_project_features(
        project,
    )

    inspection_features = _calculate_inspection_features(
        inspections,
    )

    features = RiskFeatures(
        **attendance_features,
        **project_features,
        **inspection_features,
    )

    signal_features = _calculate_signal_features(
        features,
    )

    for key, value in signal_features.items():
        setattr(
            features,
            key,
            value,
        )

    return asdict(features)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Test using the REAL AttendanceEngine output format.
    # ---------------------------------------------------------

    sample_attendance = {
        "session_id": "test-session",
        "session_started_at": "2026-09-06T10:00:00",
        "session_ended_at": "2026-09-06T10:10:00",
        "total_tracked": 4,
        "staff": 1,
        "beneficiary": 2,
        "unknown": 1,
        "records": [
            {
                "track_id": 1,
                "role": "Staff",
                "first_seen": "2026-09-06T10:00:01",
                "last_seen": "2026-09-06T10:09:00",
                "duration_seconds": 539.0,
                "observations": 100,
            },
            {
                "track_id": 2,
                "role": "Beneficiary",
                "first_seen": "2026-09-06T10:00:05",
                "last_seen": "2026-09-06T10:08:00",
                "duration_seconds": 475.0,
                "observations": 90,
            },
            {
                "track_id": 3,
                "role": "Beneficiary",
                "first_seen": "2026-09-06T10:01:00",
                "last_seen": "2026-09-06T10:07:00",
                "duration_seconds": 360.0,
                "observations": 70,
            },
            {
                "track_id": 4,
                "role": "Unknown",
                "first_seen": "2026-09-06T10:02:00",
                "last_seen": "2026-09-06T10:05:00",
                "duration_seconds": 180.0,
                "observations": 30,
            },
        ],
    }

    sample_project = {
        "status": "active",
        "risk_level": "medium",
        "total_inspections": 10,
        "completed_inspections": 7,
        "pending_inspections": 3,
        "high_risk_findings": 1,
    }

    sample_inspections = [
        {
            "risk_level": "medium",
            "findings": [
                {
                    "title": "Attendance mismatch",
                    "status": "open",
                }
            ],
        },
        {
            "risk_level": "high",
            "findings": [
                {
                    "title": "Attendance mismatch",
                    "status": "open",
                }
            ],
        },
    ]

    result = aggregate_features(
        attendance=sample_attendance,
        project=sample_project,
        inspections=sample_inspections,
    )

    print("=" * 60)
    print("FEATURE AGGREGATOR REAL ATTENDANCE TEST")
    print("=" * 60)

    print(
        f"Staff count       : "
        f"{result['attendance_staff_count']}"
    )
    print(
        f"Beneficiary count : "
        f"{result['attendance_beneficiary_count']}"
    )
    print(
        f"Unknown count     : "
        f"{result['attendance_unknown_count']}"
    )
    print(
        f"Staff ratio       : "
        f"{result['attendance_staff_ratio']}"
    )
    print(
        f"Beneficiary ratio : "
        f"{result['attendance_beneficiary_ratio']}"
    )
    print(
        f"Unknown ratio     : "
        f"{result['attendance_unknown_ratio']}"
    )
    print(
        f"Duration seconds  : "
        f"{result['attendance_duration_seconds']}"
    )
    print(
        f"Observations      : "
        f"{result['attendance_observation_count']}"
    )

    print("-" * 60)

    expected_duration = 539.0 + 475.0 + 360.0 + 180.0
    expected_observations = 100 + 90 + 70 + 30

    assert result["attendance_total_tracked"] == 4
    assert result["attendance_staff_count"] == 1
    assert result["attendance_beneficiary_count"] == 2
    assert result["attendance_unknown_count"] == 1

    assert result["attendance_staff_ratio"] == 0.25
    assert result["attendance_beneficiary_ratio"] == 0.5
    assert result["attendance_unknown_ratio"] == 0.25

    assert result["attendance_duration_seconds"] == expected_duration
    assert result["attendance_observation_count"] == expected_observations

    print("Real attendance contract: PASS")
    print("Role normalization: PASS")
    print("Duration derivation: PASS")
    print("Observation derivation: PASS")
    print("=" * 60)
    print("FEATURE AGGREGATOR TEST SUCCESSFUL")
    print("=" * 60)