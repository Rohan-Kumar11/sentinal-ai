from fastapi import APIRouter, HTTPException

from api.attendance_service import attendance_service


router = APIRouter(
    prefix="/api/v1/attendance/ai",
    tags=["AI Attendance Monitoring"],
)


@router.post("/start")
def start_ai_attendance():
    result = attendance_service.start()

    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result["message"],
        )

    return result


@router.post("/stop")
def stop_ai_attendance():
    result = attendance_service.stop()

    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result["message"],
        )

    return result


@router.post("/restart")
def restart_ai_attendance():
    result = attendance_service.restart()

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result["message"],
        )

    return result


@router.get("/status")
def get_ai_attendance_status():
    return attendance_service.status()


@router.get("/current")
def get_current_ai_attendance():
    return attendance_service.active_count()