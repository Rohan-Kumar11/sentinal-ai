from __future__ import annotations

from typing import Any, Dict, List


RISK_LEVELS = (
    "low",
    "medium",
    "high",
    "critical",
)


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""
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
    """Safely convert a value to int."""
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """Clamp a value to a fixed range."""
    return max(
        minimum,
        min(maximum, value),
    )


def _risk_level_from_score(
    score: float,
) -> str:
    """
    Convert a 0-100 risk score into a risk level.

    0-24   -> low
    25-49  -> medium
    50-74  -> high
    75-100 -> critical
    """

    if score >= 75:
        return "critical"

    if score >= 50:
        return "high"

    if score >= 25:
        return "medium"

    return "low"


def _project_risk_base_score(
    project_risk_level: str,
) -> float:
    """
    Convert an existing project risk label into
    a small baseline contribution.

    This is intentionally not allowed to dominate
    the final score.
    """

    normalized = (
        str(project_risk_level or "low")
        .strip()
        .lower()
    )

    mapping = {
        "low": 0.0,
        "medium": 10.0,
        "high": 20.0,
        "critical": 25.0,
    }

    return mapping.get(
        normalized,
        0.0,
    )


def _attendance_score(
    features: Dict[str, Any],
) -> float:
    """
    Calculate the attendance contribution.

    Maximum contribution: 25 points.

    A stale/missing 24h monitoring window adds a fixed
    contribution on top of the unknown-ratio and anomaly
    signal terms — an institute with no recent AI monitoring
    should not default to a zero attendance-risk score,
    since "not being watched" is itself a risk signal.
    """

    unknown_ratio = _safe_float(
        features.get("attendance_unknown_ratio"),
    )

    attendance_signal = _safe_float(
        features.get("attendance_anomaly_signal"),
    )

    is_stale = bool(
        features.get("attendance_stale", False)
    )

    score = (
        unknown_ratio * 15.0
        + attendance_signal * 10.0
        + (8.0 if is_stale else 0.0)
    )

    return _clamp(
        score,
        0.0,
        25.0,
    )

def _project_score(
    features: Dict[str, Any],
) -> float:
    """
    Calculate the project/application contribution.

    Maximum contribution: 25 points.
    """

    pending_ratio = _safe_float(
        features.get("pending_inspection_ratio"),
    )

    high_risk_finding_ratio = _safe_float(
        features.get("high_risk_finding_ratio"),
    )

    project_signal = _safe_float(
        features.get("project_anomaly_signal"),
    )

    existing_risk = _project_risk_base_score(
        features.get("project_risk_level"),
    )

    score = (
        pending_ratio * 8.0
        + high_risk_finding_ratio * 7.0
        + project_signal * 5.0
        + existing_risk
    )

    return _clamp(
        score,
        0.0,
        25.0,
    )


def _inspection_score(
    features: Dict[str, Any],
) -> float:
    """
    Calculate the inspection-history contribution.

    Maximum contribution: 30 points.
    """

    high_risk_ratio = _safe_float(
        features.get("inspection_high_risk_ratio"),
    )

    open_findings = _safe_int(
        features.get("open_finding_count"),
    )

    repeated_issues = _safe_int(
        features.get("repeated_issue_count"),
    )

    inspection_signal = _safe_float(
        features.get("inspection_anomaly_signal"),
    )

    score = (
        high_risk_ratio * 12.0
        + min(open_findings / 10.0, 1.0) * 8.0
        + min(repeated_issues / 5.0, 1.0) * 5.0
        + inspection_signal * 5.0
    )

    return _clamp(
        score,
        0.0,
        30.0,
    )


def _anomaly_score(
    anomalies: List[Dict[str, Any]],
) -> float:
    """
    Calculate a contribution from detected anomalies.

    Maximum contribution: 20 points.

    High-severity anomalies contribute more than
    medium/low anomalies.
    """

    if not anomalies:
        return 0.0

    severity_weights = {
        "low": 1.0,
        "medium": 2.0,
        "high": 3.5,
        "critical": 4.5,
    }

    weighted_total = 0.0

    for anomaly in anomalies:
        severity = str(
            anomaly.get("severity", "low")
        ).strip().lower()

        signal = _safe_float(
            anomaly.get("signal"),
        )

        weight = severity_weights.get(
            severity,
            1.0,
        )

        weighted_total += (
            signal * weight
        )

    # Normalize the anomaly contribution.
    #
    # More anomalies increase the score, but the
    # result is capped so anomaly count cannot
    # independently create an extreme score.
    normalized = min(
        1.0,
        weighted_total / 12.0,
    )

    return normalized * 20.0


def _build_reasons(
    features: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
) -> List[str]:
    """
    Build human-readable reasons explaining
    the calculated risk score.
    """

    reasons: List[str] = []

    unknown_ratio = _safe_float(
        features.get("attendance_unknown_ratio"),
    )

    pending_ratio = _safe_float(
        features.get("pending_inspection_ratio"),
    )

    high_risk_finding_ratio = _safe_float(
        features.get("high_risk_finding_ratio"),
    )

    high_risk_inspection_ratio = _safe_float(
        features.get("inspection_high_risk_ratio"),
    )

    open_findings = _safe_int(
        features.get("open_finding_count"),
    )

    repeated_issues = _safe_int(
        features.get("repeated_issue_count"),
    )

    project_risk = str(
        features.get("project_risk_level", "low")
    ).strip().lower()
    
    if bool(features.get("attendance_stale", False)):
        reasons.append(
            "No AI attendance monitoring session has run "
            "in the last 24 hours."
        )

    if unknown_ratio >= 0.30:
        reasons.append(
            "A significant share of attendance "
            "observations has an unknown role."
        )

    if pending_ratio >= 0.50:
        reasons.append(
            "A large proportion of inspections "
            "are pending."
        )

    if high_risk_finding_ratio >= 0.30:
        reasons.append(
            "High-risk findings represent a "
            "significant portion of inspection activity."
        )

    if high_risk_inspection_ratio >= 0.50:
        reasons.append(
            "Recent inspection history contains "
            "a substantial proportion of high-risk results."
        )

    if open_findings >= 3:
        reasons.append(
            f"{open_findings} inspection findings "
            "remain open or unresolved."
        )

    if repeated_issues >= 2:
        reasons.append(
            f"{repeated_issues} repeated inspection "
            "issues were identified."
        )

    if project_risk in {
        "high",
        "critical",
    }:
        reasons.append(
            f"The project is currently marked "
            f"as {project_risk} risk."
        )

    # Add important anomaly messages while avoiding
    # duplicate explanations.
    existing_reasons = {
        reason.lower()
        for reason in reasons
    }

    for anomaly in sorted(
        anomalies,
        key=lambda item: _safe_float(
            item.get("signal")
        ),
        reverse=True,
    ):
        message = str(
            anomaly.get("message", "")
        ).strip()

        if not message:
            continue

        if message.lower() not in existing_reasons:
            reasons.append(message)
            existing_reasons.add(
                message.lower()
            )

    if not reasons:
        reasons.append(
            "No major anomaly or risk indicator "
            "was identified from the available data."
        )

    return reasons


def calculate_risk(
    features: Dict[str, Any],
    anomaly_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate the final explainable project risk.

    Inputs:
        features:
            Output from feature_aggregator.aggregate_features()

        anomaly_result:
            Output from anomaly_detector.detect_anomalies()

    Returns:
        JSON-serializable risk result containing:

        - risk_score: 0-100
        - risk_level
        - component scores
        - anomaly count
        - reasons
    """

    anomalies = anomaly_result.get(
        "anomalies",
        [],
    )

    attendance_score = _attendance_score(
        features,
    )

    project_score = _project_score(
        features,
    )

    inspection_score = _inspection_score(
        features,
    )

    anomaly_score = _anomaly_score(
        anomalies,
    )

    raw_score = (
        attendance_score
        + project_score
        + inspection_score
        + anomaly_score
    )

    risk_score = round(
        _clamp(raw_score),
        2,
    )

    risk_level = _risk_level_from_score(
        risk_score,
    )

    reasons = _build_reasons(
        features,
        anomalies,
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "component_scores": {
            "attendance": round(
                attendance_score,
                2,
            ),
            "project": round(
                project_score,
                2,
            ),
            "inspection": round(
                inspection_score,
                2,
            ),
            "anomalies": round(
                anomaly_score,
                2,
            ),
        },
        "anomaly_count": len(anomalies),
        "high_anomaly_count": _safe_int(
            anomaly_result.get("high_count"),
        ),
        "medium_anomaly_count": _safe_int(
            anomaly_result.get("medium_count"),
        ),
        "low_anomaly_count": _safe_int(
            anomaly_result.get("low_count"),
        ),
        "reasons": reasons,
    }


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Local integration sanity test
    # ---------------------------------------------------------

    sample_features = {
        "attendance_unknown_ratio": 0.40,
        "attendance_anomaly_signal": 0.50,

        "project_risk_level": "high",
        "pending_inspection_ratio": 0.60,
        "high_risk_finding_ratio": 0.40,
        "project_anomaly_signal": 0.50,

        "inspection_high_risk_ratio": 0.75,
        "open_finding_count": 4,
        "repeated_issue_count": 3,
        "inspection_anomaly_signal": 0.75,
    }

    sample_anomalies = {
        "anomalies": [
            {
                "code": "HIGH_UNKNOWN_ATTENDANCE",
                "category": "attendance",
                "severity": "medium",
                "signal": 0.50,
                "message": (
                    "A significant portion of observed "
                    "people could not be assigned a known role."
                ),
            },
            {
                "code": "PROJECT_HIGH_RISK_STATUS",
                "category": "project",
                "severity": "high",
                "signal": 0.75,
                "message": (
                    "The project is currently marked "
                    "as high risk in the application data."
                ),
            },
            {
                "code": "REPEATED_HIGH_RISK_INSPECTIONS",
                "category": "inspection",
                "severity": "high",
                "signal": 0.75,
                "message": (
                    "A substantial portion of recent inspections "
                    "have been classified as high risk."
                ),
            },
        ],
        "total_anomalies": 3,
        "high_count": 2,
        "medium_count": 1,
        "low_count": 0,
    }

    result = calculate_risk(
        features=sample_features,
        anomaly_result=sample_anomalies,
    )

    print("=" * 60)
    print("RISK ENGINE TEST")
    print("=" * 60)

    print(
        f"Risk score: "
        f"{result['risk_score']}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print()
    print("Component scores:")

    for name, score in result[
        "component_scores"
    ].items():
        print(
            f"  {name}: {score}"
        )

    print()
    print(
        f"Anomaly count: "
        f"{result['anomaly_count']}"
    )

    print(
        f"High anomalies: "
        f"{result['high_anomaly_count']}"
    )

    print(
        f"Medium anomalies: "
        f"{result['medium_anomaly_count']}"
    )

    print(
        f"Low anomalies: "
        f"{result['low_anomaly_count']}"
    )

    print()
    print("Risk reasons:")

    for reason in result["reasons"]:
        print(
            f"  - {reason}"
        )

    print("=" * 60)
    print("RISK ENGINE TEST SUCCESSFUL")
    print("=" * 60)