from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class Anomaly:
    """
    Represents one explainable anomaly.

    severity:
        low / medium / high

    signal:
        Normalized strength of the anomaly from 0.0 to 1.0.
    """

    code: str
    category: str
    severity: str
    signal: float
    message: str


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely convert a value to int.
    """
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp a numeric value to a fixed range.
    """
    return max(
        minimum,
        min(maximum, value),
    )


def _severity_from_signal(
    signal: float,
) -> str:
    """
    Convert anomaly strength into an explainable severity.
    """

    if signal >= 0.75:
        return "high"

    if signal >= 0.40:
        return "medium"

    return "low"


def detect_attendance_anomalies(
    features: Dict[str, Any],
) -> List[Anomaly]:
    """
    Detect unusual attendance patterns.

    This does not claim fraud or misconduct.
    It only identifies patterns that deserve attention.
    """

    anomalies: List[Anomaly] = []

    unknown_ratio = _safe_float(
        features.get("attendance_unknown_ratio"),
    )

    total_tracked = _safe_int(
        features.get("attendance_total_tracked"),
    )

    duration_seconds = _safe_float(
        features.get("attendance_duration_seconds"),
    )

    staff_ratio = _safe_float(
        features.get("attendance_staff_ratio"),
    )

    beneficiary_ratio = _safe_float(
        features.get("attendance_beneficiary_ratio"),
    )

    # ---------------------------------------------------------
    # Unknown-person anomaly
    # ---------------------------------------------------------

    if unknown_ratio >= 0.30:
        signal = _clamp(
            unknown_ratio * 1.25,
        )

        anomalies.append(
            Anomaly(
                code="HIGH_UNKNOWN_ATTENDANCE",
                category="attendance",
                severity=_severity_from_signal(signal),
                signal=signal,
                message=(
                    "A significant portion of observed "
                    "people could not be assigned a known role."
                ),
            )
        )

    # ---------------------------------------------------------
    # Very small attendance observation
    # ---------------------------------------------------------

    if 0 < total_tracked < 3:
        signal = 0.45

        anomalies.append(
            Anomaly(
                code="LOW_ATTENDANCE_SAMPLE",
                category="attendance",
                severity="medium",
                signal=signal,
                message=(
                    "The attendance session contains very few "
                    "observed people, so attendance evidence "
                    "may be insufficient."
                ),
            )
        )

    # ---------------------------------------------------------
    # Very short attendance session
    # ---------------------------------------------------------

    if 0 < duration_seconds < 120:
        signal = 0.50

        anomalies.append(
            Anomaly(
                code="SHORT_ATTENDANCE_SESSION",
                category="attendance",
                severity="medium",
                signal=signal,
                message=(
                    "The attendance observation session was "
                    "very short and may provide limited evidence."
                ),
            )
        )

    # ---------------------------------------------------------
    # Staff-only / beneficiary-only pattern
    # ---------------------------------------------------------

    if total_tracked >= 5:
        if staff_ratio >= 0.90:
            signal = 0.60

            anomalies.append(
                Anomaly(
                    code="STAFF_DOMINANT_ATTENDANCE",
                    category="attendance",
                    severity="medium",
                    signal=signal,
                    message=(
                        "The observed attendance population is "
                        "strongly dominated by staff."
                    ),
                )
            )

        elif beneficiary_ratio >= 0.95:
            signal = 0.45

            anomalies.append(
                Anomaly(
                    code="BENEFICIARY_ONLY_ATTENDANCE",
                    category="attendance",
                    severity="medium",
                    signal=signal,
                    message=(
                        "The observed attendance population "
                        "contains almost no staff observations."
                    ),
                )
            )

    return anomalies


def detect_project_anomalies(
    features: Dict[str, Any],
) -> List[Anomaly]:
    """
    Detect unusual project/application patterns.
    """

    anomalies: List[Anomaly] = []

    pending_ratio = _safe_float(
        features.get("pending_inspection_ratio"),
    )

    high_risk_finding_ratio = _safe_float(
        features.get("high_risk_finding_ratio"),
    )

    project_status = str(
        features.get("project_status", "unknown")
    ).strip().lower()

    project_risk_level = str(
        features.get("project_risk_level", "low")
    ).strip().lower()

    pending_inspections = _safe_int(
        features.get("project_pending_inspections"),
    )

    # ---------------------------------------------------------
    # Large pending inspection backlog
    # ---------------------------------------------------------

    if pending_ratio >= 0.50:
        signal = _clamp(
            pending_ratio,
        )

        anomalies.append(
            Anomaly(
                code="HIGH_PENDING_INSPECTIONS",
                category="project",
                severity=_severity_from_signal(signal),
                signal=signal,
                message=(
                    "A large proportion of scheduled inspections "
                    "are still pending."
                ),
            )
        )

    elif pending_inspections >= 3:
        signal = 0.40

        anomalies.append(
            Anomaly(
                code="INSPECTION_BACKLOG",
                category="project",
                severity="medium",
                signal=signal,
                message=(
                    "The project has multiple pending inspections "
                    "that may require attention."
                ),
            )
        )

    # ---------------------------------------------------------
    # High-risk findings
    # ---------------------------------------------------------

    if high_risk_finding_ratio >= 0.30:
        signal = _clamp(
            high_risk_finding_ratio * 1.25,
        )

        anomalies.append(
            Anomaly(
                code="HIGH_RISK_FINDING_PATTERN",
                category="project",
                severity=_severity_from_signal(signal),
                signal=signal,
                message=(
                    "High-risk findings represent a significant "
                    "portion of the project's inspection activity."
                ),
            )
        )

    # ---------------------------------------------------------
    # Project already marked high risk
    # ---------------------------------------------------------

    if project_risk_level == "high":
        signal = 0.75

        anomalies.append(
            Anomaly(
                code="PROJECT_HIGH_RISK_STATUS",
                category="project",
                severity="high",
                signal=signal,
                message=(
                    "The project is currently marked as high risk "
                    "in the application data."
                ),
            )
        )

    # ---------------------------------------------------------
    # Unknown project status
    # ---------------------------------------------------------

    if project_status == "unknown":
        signal = 0.35

        anomalies.append(
            Anomaly(
                code="UNKNOWN_PROJECT_STATUS",
                category="project",
                severity="low",
                signal=signal,
                message=(
                    "Project status information is unavailable "
                    "or could not be interpreted."
                ),
            )
        )

    return anomalies


def detect_inspection_anomalies(
    features: Dict[str, Any],
) -> List[Anomaly]:
    """
    Detect unusual inspection-history patterns.
    """

    anomalies: List[Anomaly] = []

    high_risk_ratio = _safe_float(
        features.get("inspection_high_risk_ratio"),
    )

    open_finding_count = _safe_int(
        features.get("open_finding_count"),
    )

    repeated_issue_count = _safe_int(
        features.get("repeated_issue_count"),
    )

    inspection_history_count = _safe_int(
        features.get("inspection_history_count"),
    )

    # ---------------------------------------------------------
    # Repeated high-risk inspections
    # ---------------------------------------------------------

    if (
        inspection_history_count >= 2
        and high_risk_ratio >= 0.50
    ):
        signal = _clamp(
            high_risk_ratio,
        )

        anomalies.append(
            Anomaly(
                code="REPEATED_HIGH_RISK_INSPECTIONS",
                category="inspection",
                severity=_severity_from_signal(signal),
                signal=signal,
                message=(
                    "A substantial portion of recent inspections "
                    "have been classified as high risk."
                ),
            )
        )

    # ---------------------------------------------------------
    # Open findings
    # ---------------------------------------------------------

    if open_finding_count >= 3:
        signal = _clamp(
            open_finding_count / 10.0,
        )

        anomalies.append(
            Anomaly(
                code="MULTIPLE_OPEN_FINDINGS",
                category="inspection",
                severity=_severity_from_signal(signal),
                signal=signal,
                message=(
                    "Multiple inspection findings remain open "
                    "or unresolved."
                ),
            )
        )

    # ---------------------------------------------------------
    # Repeated issues
    # ---------------------------------------------------------

    if repeated_issue_count >= 2:
        signal = _clamp(
            repeated_issue_count / 5.0,
        )

        anomalies.append(
            Anomaly(
                code="REPEATED_INSPECTION_ISSUES",
                category="inspection",
                severity=_severity_from_signal(signal),
                signal=signal,
                message=(
                    "Similar issues have appeared repeatedly "
                    "across inspection observations."
                ),
            )
        )

    return anomalies


def detect_anomalies(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all anomaly detectors.

    Returns JSON-serializable output.
    """

    attendance_anomalies = (
        detect_attendance_anomalies(features)
    )

    project_anomalies = (
        detect_project_anomalies(features)
    )

    inspection_anomalies = (
        detect_inspection_anomalies(features)
    )

    anomalies = (
        attendance_anomalies
        + project_anomalies
        + inspection_anomalies
    )

    high_count = sum(
        1
        for anomaly in anomalies
        if anomaly.severity == "high"
    )

    medium_count = sum(
        1
        for anomaly in anomalies
        if anomaly.severity == "medium"
    )

    low_count = sum(
        1
        for anomaly in anomalies
        if anomaly.severity == "low"
    )

    strongest_signal = max(
        (
            anomaly.signal
            for anomaly in anomalies
        ),
        default=0.0,
    )

    return {
        "anomalies": [
            asdict(anomaly)
            for anomaly in anomalies
        ],
        "total_anomalies": len(anomalies),
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "strongest_signal": strongest_signal,
    }


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Local sanity test
    # ---------------------------------------------------------

    sample_features = {
        "attendance_total_tracked": 10,
        "attendance_staff_ratio": 0.10,
        "attendance_beneficiary_ratio": 0.50,
        "attendance_unknown_ratio": 0.40,
        "attendance_duration_seconds": 90,

        "project_status": "active",
        "project_risk_level": "high",
        "project_pending_inspections": 5,
        "pending_inspection_ratio": 0.60,
        "high_risk_finding_ratio": 0.40,

        "inspection_history_count": 4,
        "inspection_high_risk_ratio": 0.75,
        "open_finding_count": 4,
        "repeated_issue_count": 3,
    }

    result = detect_anomalies(
        sample_features,
    )

    print("=" * 60)
    print("ANOMALY DETECTOR TEST")
    print("=" * 60)

    for anomaly in result["anomalies"]:
        print(
            f"[{anomaly['severity'].upper()}] "
            f"{anomaly['code']}: "
            f"{anomaly['message']}"
        )

    print()
    print(
        f"Total anomalies: "
        f"{result['total_anomalies']}"
    )

    print(
        f"High: "
        f"{result['high_count']}"
    )

    print(
        f"Medium: "
        f"{result['medium_count']}"
    )

    print(
        f"Low: "
        f"{result['low_count']}"
    )

    print(
        f"Strongest signal: "
        f"{result['strongest_signal']:.2f}"
    )

    print("=" * 60)
    print("ANOMALY DETECTOR TEST SUCCESSFUL")
    print("=" * 60)