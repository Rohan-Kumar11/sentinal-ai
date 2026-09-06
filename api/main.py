
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.routes.risk import router as risk_router


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ATTENDANCE_DIRECTORY = (
    BASE_DIR
    / "experiments"
    / "attendance_sessions"
)

# Make sure the attendance directory exists.
ATTENDANCE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Sentinal AI API",
    description=(
        "AI backend for Sentinal smart monitoring, "
        "attendance and inspection intelligence."
    ),
    version="0.6.4",
)


# ============================================================
# CORS
# ============================================================

# Flutter Web uses a dynamically assigned localhost port.
#
# Therefore, localhost and 127.0.0.1 are allowed with any port
# during local development.
#
# This is ONLY intended for local development.
# Production CORS should use specific trusted origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC MODELS
# ============================================================


class AttendanceRecord(BaseModel):
    track_id: int
    role: str
    first_seen: str
    last_seen: str
    duration_seconds: float
    observations: int


class AttendanceSession(BaseModel):
    session_id: str
    session_started_at: str
    session_ended_at: str
    total_tracked: int
    staff: int
    beneficiary: int
    unknown: int
    records: List[AttendanceRecord]


class AttendanceSummary(BaseModel):
    total_sessions: int
    total_tracked: int
    total_staff: int
    total_beneficiary: int
    total_unknown: int
    total_observed_seconds: float


class RoleStatistic(BaseModel):
    role: str
    people: int
    observed_seconds: float
    percentage_of_people: float


class RoleStatisticsResponse(BaseModel):
    total_tracked: int
    statistics: List[RoleStatistic]


class SessionRoleStatisticsResponse(BaseModel):
    session_id: str
    session_started_at: str
    session_ended_at: str
    total_tracked: int
    statistics: List[RoleStatistic]


# ============================================================
# DEVELOPMENT STORAGE
# ============================================================

# Kept for compatibility with the existing application.
#
# File storage is now the persistent source for attendance
# sessions.

attendance_storage: List[AttendanceSession] = []


# ============================================================
# FILE HELPERS
# ============================================================


def load_attendance_files() -> List[AttendanceSession]:
    """
    Load all valid attendance JSON files.
    """

    sessions: List[AttendanceSession] = []

    if not ATTENDANCE_DIRECTORY.exists():
        return sessions

    for file_path in sorted(
        ATTENDANCE_DIRECTORY.glob("attendance_*.json")
    ):
        try:
            session = AttendanceSession.model_validate_json(
                file_path.read_text(
                    encoding="utf-8"
                )
            )

            sessions.append(session)

        except Exception as exc:
            print(
                f"WARNING: Could not load "
                f"{file_path.name}: {exc}"
            )

    return sessions


def load_latest_attendance_session() -> AttendanceSession:
    """
    Return the newest valid attendance session.
    """

    sessions = load_attendance_files()

    if not sessions:
        raise HTTPException(
            status_code=404,
            detail="No attendance sessions found.",
        )

    return sessions[-1]


def load_attendance_session(
    session_id: str,
) -> AttendanceSession:
    """
    Load one attendance session by session ID.
    """

    file_path = (
        ATTENDANCE_DIRECTORY
        / f"attendance_{session_id}.json"
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Attendance session "
                f"'{session_id}' not found."
            ),
        )

    try:
        return AttendanceSession.model_validate_json(
            file_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not read attendance session: "
                f"{exc}"
            ),
        )


def save_attendance_session(
    session: AttendanceSession,
) -> Path:
    """
    Persist an attendance session as JSON.

    The filename is based on the session ID so the same
    session can be retrieved later through the GET APIs.
    """

    ATTENDANCE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        ATTENDANCE_DIRECTORY
        / f"attendance_{session.session_id}.json"
    )

    try:
        file_path.write_text(
            session.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not save attendance session: "
                f"{exc}"
            ),
        )

    return file_path


# ============================================================
# ROOT
# ============================================================


@app.get("/")
def root():
    return {
        "name": "Sentinal AI API",
        "version": app.version,
        "status": "running",
    }


# ============================================================
# HEALTH
# ============================================================


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "sentinal-ai",
        "version": app.version,
    }


# ============================================================
# API STATUS
# ============================================================


@app.get("/api/v1/status")
def api_status():
    sessions = load_attendance_files()

    return {
        "status": "online",
        "service": "sentinal-ai",
        "version": app.version,
        "attendance_sessions": len(sessions),
    }


# ============================================================
# ALL ATTENDANCE SESSIONS
# ============================================================


@app.get(
    "/api/v1/attendance",
    response_model=List[AttendanceSession],
)
def get_attendance():
    return load_attendance_files()


# ============================================================
# LATEST ATTENDANCE
# ============================================================


@app.get(
    "/api/v1/attendance/latest",
    response_model=AttendanceSession,
)
def get_latest_attendance():
    return load_latest_attendance_session()


# ============================================================
# OVERALL ATTENDANCE SUMMARY
# ============================================================


@app.get(
    "/api/v1/attendance/summary",
    response_model=AttendanceSummary,
)
def get_attendance_summary():
    sessions = load_attendance_files()

    total_tracked = 0
    total_staff = 0
    total_beneficiary = 0
    total_unknown = 0
    total_observed_seconds = 0.0

    for session in sessions:
        total_tracked += session.total_tracked
        total_staff += session.staff
        total_beneficiary += session.beneficiary
        total_unknown += session.unknown

        total_observed_seconds += sum(
            record.duration_seconds
            for record in session.records
        )

    return AttendanceSummary(
        total_sessions=len(sessions),
        total_tracked=total_tracked,
        total_staff=total_staff,
        total_beneficiary=total_beneficiary,
        total_unknown=total_unknown,
        total_observed_seconds=round(
            total_observed_seconds,
            2,
        ),
    )


# ============================================================
# OVERALL ROLE STATISTICS
# ============================================================


@app.get(
    "/api/v1/attendance/role-statistics",
    response_model=RoleStatisticsResponse,
)
def get_role_statistics():
    sessions = load_attendance_files()

    role_data = {
        "Staff": {
            "people": 0,
            "observed_seconds": 0.0,
        },
        "Beneficiary": {
            "people": 0,
            "observed_seconds": 0.0,
        },
        "Unknown": {
            "people": 0,
            "observed_seconds": 0.0,
        },
    }

    total_tracked = 0

    for session in sessions:
        total_tracked += session.total_tracked

        for record in session.records:
            role = record.role

            if role not in role_data:
                role_data[role] = {
                    "people": 0,
                    "observed_seconds": 0.0,
                }

            role_data[role]["people"] += 1
            role_data[role]["observed_seconds"] += (
                record.duration_seconds
            )

    statistics = []

    for role, data in role_data.items():
        percentage = 0.0

        if total_tracked > 0:
            percentage = (
                data["people"]
                / total_tracked
            ) * 100

        statistics.append(
            RoleStatistic(
                role=role,
                people=data["people"],
                observed_seconds=round(
                    data["observed_seconds"],
                    2,
                ),
                percentage_of_people=round(
                    percentage,
                    2,
                ),
            )
        )

    return RoleStatisticsResponse(
        total_tracked=total_tracked,
        statistics=statistics,
    )


# ============================================================
# SESSION ROLE STATISTICS
# ============================================================


@app.get(
    "/api/v1/attendance/{session_id}/role-statistics",
    response_model=SessionRoleStatisticsResponse,
)
def get_session_role_statistics(
    session_id: str,
):
    session = load_attendance_session(
        session_id
    )

    role_data = {
        "Staff": {
            "people": 0,
            "observed_seconds": 0.0,
        },
        "Beneficiary": {
            "people": 0,
            "observed_seconds": 0.0,
        },
        "Unknown": {
            "people": 0,
            "observed_seconds": 0.0,
        },
    }

    for record in session.records:
        role = record.role

        if role not in role_data:
            role_data[role] = {
                "people": 0,
                "observed_seconds": 0.0,
            }

        role_data[role]["people"] += 1
        role_data[role]["observed_seconds"] += (
            record.duration_seconds
        )

    statistics = []

    for role, data in role_data.items():
        percentage = 0.0

        if session.total_tracked > 0:
            percentage = (
                data["people"]
                / session.total_tracked
            ) * 100

        statistics.append(
            RoleStatistic(
                role=role,
                people=data["people"],
                observed_seconds=round(
                    data["observed_seconds"],
                    2,
                ),
                percentage_of_people=round(
                    percentage,
                    2,
                ),
            )
        )

    return SessionRoleStatisticsResponse(
        session_id=session.session_id,
        session_started_at=session.session_started_at,
        session_ended_at=session.session_ended_at,
        total_tracked=session.total_tracked,
        statistics=statistics,
    )


# ============================================================
# SPECIFIC ATTENDANCE SESSION
# ============================================================
#
# IMPORTANT:
# This dynamic route MUST remain after every fixed
# attendance route.
#
# It handles:
#
# /api/v1/attendance/20260904_192538
#
# It must NOT handle:
#
# /summary
# /role-statistics
# /latest
# ============================================================


@app.get(
    "/api/v1/attendance/{session_id}",
    response_model=AttendanceSession,
)
def get_attendance_session(
    session_id: str,
):
    return load_attendance_session(
        session_id
    )


# ============================================================
# POST ATTENDANCE
# ============================================================


@app.post(
    "/api/v1/attendance",
    response_model=AttendanceSession,
)
def create_attendance(
    session: AttendanceSession,
):
    """
    Validate and persist an attendance session.

    If the same session_id is submitted again, its JSON file
    is replaced with the latest version.
    """

    # Keep the in-memory storage for development compatibility.
    attendance_storage.append(session)

    # Persist the validated session to disk.
    save_attendance_session(session)

    return session


# ============================================================
# RISK INTELLIGENCE ROUTES
# ============================================================
#
# These routes connect:
#
# Attendance
#     +
# Project/Application Data
#     +
# Inspection Data
#     ↓
# Feature Aggregation
#     ↓
# Anomaly Detection
#     ↓
# Risk Engine
#
# The router implementation lives in:
#
# api/routes/risk.py
# ============================================================

app.include_router(risk_router)

