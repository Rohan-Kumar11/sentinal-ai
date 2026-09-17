import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from risk_engine.feature_aggregator import aggregate_features
from risk_engine.anomaly_detector import detect_anomalies
from risk_engine.risk_engine import calculate_risk


router = APIRouter(
    prefix="/api/v1/risk",
    tags=["Risk Intelligence"],
)


# ------------------------------------------------------------
# Attendance session storage
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

ATTENDANCE_DIRECTORY = (
    BASE_DIR
    / "experiments"
    / "attendance_sessions"
)

ATTENDANCE_WINDOW_HOURS = 24


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_attendance_last_24h() -> Optional[Dict[str, Any]]:
    """
    Aggregate attendance sessions produced in the last
    ATTENDANCE_WINDOW_HOURS hours.

    Replaces the old "just grab the single latest file"
    approach. That approach meant a session from days ago
    could silently keep feeding the risk engine forever,
    with no notion of "monitoring has gone stale".

    Behavior:
      - Sums total_tracked/staff/beneficiary/unknown across
        every session file modified within the window.
      - Returns None if the attendance directory doesn't
        exist or literally no files exist at all (never
        monitored) — downstream this is treated the same
        as "no data available".
      - Returns a dict with session_count_24h == 0 if files
        exist but none fall inside the 24h window — this is
        the "monitoring has gone stale" case, and is
        distinguishable from "never monitored" so the
        feature aggregator can flag it explicitly.
    """

    if not ATTENDANCE_DIRECTORY.exists():
        return None

    all_files = sorted(
        ATTENDANCE_DIRECTORY.glob("attendance_*.json"),
        key=lambda path: path.stat().st_mtime,
    )

    if not all_files:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(
        hours=ATTENDANCE_WINDOW_HOURS
    )

    recent_sessions: List[Dict[str, Any]] = []
    latest_session_ended_at: Optional[str] = None

    for path in all_files:
        modified_at = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(data, dict):
            continue

        # Track the most recent session end time regardless of
        # whether it falls in-window, so we can report "how
        # stale" the data is even when there's nothing recent.
        ended_at = data.get("session_ended_at")
        if isinstance(ended_at, str):
            latest_session_ended_at = ended_at

        if modified_at >= cutoff:
            recent_sessions.append(data)

    if not recent_sessions:
        # Sessions exist historically, but none in the window.
        # This is the "stale monitoring" case.
        return {
            "total_tracked": 0,
            "staff": 0,
            "beneficiary": 0,
            "unknown": 0,
            "session_count_24h": 0,
            "window_hours": ATTENDANCE_WINDOW_HOURS,
            "last_session_ended_at": latest_session_ended_at,
        }

    total_tracked = sum(
        _safe_int(s.get("total_tracked")) for s in recent_sessions
    )
    staff = sum(_safe_int(s.get("staff")) for s in recent_sessions)
    beneficiary = sum(
        _safe_int(s.get("beneficiary")) for s in recent_sessions
    )
    unknown = sum(_safe_int(s.get("unknown")) for s in recent_sessions)

    return {
        "total_tracked": total_tracked,
        "staff": staff,
        "beneficiary": beneficiary,
        "unknown": unknown,
        "session_count_24h": len(recent_sessions),
        "window_hours": ATTENDANCE_WINDOW_HOURS,
        "last_session_ended_at": latest_session_ended_at,
    }


class RiskRequest(BaseModel):
    """
    Input data required by the risk pipeline.

    Attendance:
        Optional. If omitted, the last-24h aggregated
        attendance produced by the AI attendance pipeline
        is used instead of a single latest session.

    Project:
        Comes from the application/project layer.

    Inspection data:
        Comes from inspection history/findings.
    """

    attendance: Optional[Dict[str, Any]] = None
    project: Optional[Dict[str, Any]] = None
    inspections: List[Dict[str, Any]] = Field(
        default_factory=list
    )


class RiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    anomaly_count: int
    component_scores: Dict[str, float]
    reasons: List[str]
    anomalies: List[Dict[str, Any]]
    anomaly_counts: Dict[str, int]
    features: Dict[str, Any]


@router.get("/health")
def risk_health():
    """
    Health check for the risk intelligence module.
    """

    return {
        "status": "ok",
        "module": "risk_engine",
    }


@router.get("/attendance/last-24h")
def get_attendance_last_24h():
    """
    Returns the 24h-aggregated attendance the risk engine
    will use if the caller doesn't supply attendance
    explicitly. Lets the Flutter UI show the same window
    the risk score is actually based on, instead of just
    the single latest session.
    """

    data = load_attendance_last_24h()

    if data is None:
        return {
            "available": False,
            "message": "No AI attendance sessions have ever been recorded.",
        }

    if data.get("session_count_24h", 0) == 0:
        return {
            "available": False,
            "stale": True,
            "message": (
                "No AI attendance sessions in the last "
                f"{ATTENDANCE_WINDOW_HOURS} hours."
            ),
            "last_session_ended_at": data.get("last_session_ended_at"),
        }

    return {"available": True, "stale": False, **data}


@router.post(
    "/calculate",
    response_model=RiskResponse,
)
def calculate_project_risk(
    request: RiskRequest,
):
    """
    Calculate project risk using:

    Attendance (last 24h, aggregated across sessions)
        +
    Project/Application Data
        +
    Inspection History/Findings
        ↓
    Feature Aggregation
        ↓
    Anomaly Detection
        ↓
    Risk Calculation

    If attendance is not explicitly supplied in the request,
    the last-24h aggregated attendance produced by the real
    attendance pipeline is loaded automatically. If no
    sessions ran in that window, the feature aggregator is
    told explicitly ("stale"/"never monitored") so a gap in
    monitoring is itself visible in the risk reasons, rather
    than silently scoring as zero risk.
    """

    # --------------------------------------------------------
    # STEP 1: Resolve attendance input
    # --------------------------------------------------------

    attendance = request.attendance

    if attendance is None:
        attendance = load_attendance_last_24h()

    # --------------------------------------------------------
    # STEP 2: Aggregate raw data into risk features
    # --------------------------------------------------------

    features = aggregate_features(
        attendance=attendance,
        project=request.project,
        inspections=request.inspections,
    )

    # --------------------------------------------------------
    # STEP 3: Detect anomalies
    # --------------------------------------------------------

    anomaly_result = detect_anomalies(
        features
    )

    # --------------------------------------------------------
    # STEP 4: Calculate final risk score
    # --------------------------------------------------------

    risk_result = calculate_risk(
        features=features,
        anomaly_result=anomaly_result,
    )

    # --------------------------------------------------------
    # STEP 5: Build API response
    # --------------------------------------------------------

    return {
        "risk_score": risk_result["risk_score"],
        "risk_level": risk_result["risk_level"],
        "anomaly_count": risk_result["anomaly_count"],
        "component_scores": risk_result["component_scores"],
        "reasons": risk_result["reasons"],
        "anomalies": anomaly_result["anomalies"],
        "anomaly_counts": {
            "high": anomaly_result["high_count"],
            "medium": anomaly_result["medium_count"],
            "low": anomaly_result["low_count"],
        },
        "features": features,
    }