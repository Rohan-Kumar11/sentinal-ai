import json
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


def load_latest_attendance() -> Optional[Dict[str, Any]]:
    """
    Load the most recently saved attendance session.

    Attendance sessions are produced by the real attendance
    pipeline and saved as:

        experiments/attendance_sessions/
        attendance_<session_id>.json
    """

    if not ATTENDANCE_DIRECTORY.exists():
        return None

    attendance_files = sorted(
        ATTENDANCE_DIRECTORY.glob("attendance_*.json"),
        key=lambda path: path.stat().st_mtime,
    )

    if not attendance_files:
        return None

    latest_file = attendance_files[-1]

    try:
        with latest_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return None

        return data

    except (OSError, json.JSONDecodeError):
        return None


class RiskRequest(BaseModel):
    """
    Input data required by the risk pipeline.

    Attendance:
        Optional. If omitted, the latest real attendance
        session produced by the AI attendance pipeline is used.

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


@router.post(
    "/calculate",
    response_model=RiskResponse,
)
def calculate_project_risk(
    request: RiskRequest,
):
    """
    Calculate project risk using:

    Attendance
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
    the latest attendance session produced by the real
    attendance pipeline is loaded automatically.
    """

    # --------------------------------------------------------
    # STEP 1: Resolve attendance input
    # --------------------------------------------------------

    attendance = request.attendance

    if attendance is None:
        attendance = load_latest_attendance()

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