from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    """
    Configuration for a Sentinal video source.

    Supported source types can include:
    - ip_camera
    - rtsp
    - webcam
    - video_file
    """

    name: str
    source: str
    source_type: str
    location: str = "Unknown"
    enabled: bool = True


# Development phone camera.
#
# IMPORTANT:
# The IP address may change when the phone reconnects
# to Wi-Fi. Update this value when necessary.
PHONE_CAMERA = CameraConfig(
    name="Development Phone Camera",
    source="http://10.218.232.56:8080/video",
    source_type="ip_camera",
    location="Development Test",
)


# Example structure for future CCTV cameras.
#
# Do not enable or use these yet.
#
# CCTV_CAMERA_01 = CameraConfig(
#     name="CCTV Camera 01",
#     source="rtsp://username:password@camera-ip:554/stream",
#     source_type="rtsp",
#     location="Institute Main Hall",
#     enabled=False,
# )


def get_camera_config(name: str) -> CameraConfig:
    """
    Return a configured camera by name.
    """

    cameras = {
        "phone": PHONE_CAMERA,
    }

    if name not in cameras:
        available = ", ".join(cameras.keys())

        raise ValueError(
            f"Unknown camera '{name}'. "
            f"Available cameras: {available}"
        )

    return cameras[name]


def print_camera_config(config: CameraConfig) -> None:
    """
    Print camera configuration in a readable format.
    """

    print("=" * 60)
    print("SENTINAL CAMERA CONFIGURATION")
    print("=" * 60)

    print(f"Name        : {config.name}")
    print(f"Type        : {config.source_type}")
    print(f"Source      : {config.source}")
    print(f"Location    : {config.location}")
    print(f"Enabled     : {config.enabled}")

    print("=" * 60)