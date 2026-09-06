"""
Milestone 6: collect labeled person crops for a future classifier.

Does not train a model.
Saves a crop only when you press S, B, or U.

Uses the existing YOLO + ByteTrack pipeline.
This file only handles manual labeling and dataset collection.
"""

import argparse
import os
import sys
import time

import cv2


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _HERE)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(_HERE, "..", "video_ingestion")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(_HERE, "..", "person_detection")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(_HERE, "..", "person_tracking")),
)


from person_classifier import crop_person  # noqa: E402
from person_detector import MODEL_PATH, load_model  # noqa: E402
from person_tracker import track_people  # noqa: E402
from video_capture import frame_delay_ms, open_source  # noqa: E402


# ---------------------------------------------------------------------------
# Project / dataset configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(_HERE, "..", "..")
)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "person_roles",
)

ROLE_DIRS = {
    "staff": os.path.join(DATASET_ROOT, "staff"),
    "beneficiary": os.path.join(DATASET_ROOT, "beneficiary"),
    "unknown": os.path.join(DATASET_ROOT, "unknown"),
}

ROLE_LABELS = (
    "staff",
    "beneficiary",
    "unknown",
)


# ---------------------------------------------------------------------------
# Collection settings
# ---------------------------------------------------------------------------

# Reject very small crops because they are unlikely to be useful
# for person-role classification.
MIN_CROP_WIDTH = 80
MIN_CROP_HEIGHT = 80

# Prevent a held-down key from saving many copies of the same person.
SAVE_COOLDOWN_SECONDS = 0.5

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
}

WINDOW_TITLE = "Sentinal AI - Dataset Collector"


# Keyboard -> dataset class
KEY_TO_ROLE = {
    ord("s"): "staff",
    ord("S"): "staff",
    ord("b"): "beneficiary",
    ord("B"): "beneficiary",
    ord("u"): "unknown",
    ord("U"): "unknown",
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def ensure_role_dirs():
    """Create Staff / Beneficiary / Unknown directories if necessary."""
    for path in ROLE_DIRS.values():
        os.makedirs(path, exist_ok=True)


def count_images(role):
    """
    Count existing image files for one role.

    This function does not modify the dataset.
    """
    folder = ROLE_DIRS[role]

    if not os.path.isdir(folder):
        return 0

    total = 0

    for name in os.listdir(folder):
        extension = os.path.splitext(name)[1].lower()

        if extension in IMAGE_EXTS:
            total += 1

    return total


def dataset_counts():
    """Return the current number of images in each class."""
    return {
        role: count_images(role)
        for role in ROLE_LABELS
    }


def format_counts(counts):
    """Format dataset counts for console/UI display."""
    return (
        f"Staff: {counts['staff']}",
        f"Beneficiary: {counts['beneficiary']}",
        f"Unknown: {counts['unknown']}",
    )


def print_counts(counts, heading=None):
    """Print dataset counts."""
    if heading:
        print(heading)

    for line in format_counts(counts):
        print(line)


# ---------------------------------------------------------------------------
# Tracking / selection helpers
# ---------------------------------------------------------------------------

def find_track(tracks, track_id):
    """
    Return the current track with the requested ByteTrack ID.

    Returns None if the track is no longer present.
    """
    if track_id is None:
        return None

    for track in tracks:
        if track[4] == track_id:
            return track

    return None


def resolve_selection(tracks, selected_track_id):
    """
    Keep the currently selected ByteTrack ID only while it is visible.

    If the selected person disappears, clear the selection.
    The user must explicitly select another person using 1-9.

    Returns:
        (selected_track_id, selected_track)
    """

    # No selected person.
    if selected_track_id is None:
        return None, None

    # Check whether the selected person is still visible.
    current = find_track(
        tracks,
        selected_track_id,
    )

    if current is not None:
        return selected_track_id, current

    # Selected person disappeared.
    # IMPORTANT:
    # Do NOT automatically select another person.
    return None, None


# ---------------------------------------------------------------------------
# Crop validation / saving
# ---------------------------------------------------------------------------

def crop_reject_reason(crop):
    """
    Validate a person crop.

    Returns:
        None when valid.
        String describing the rejection reason otherwise.
    """

    if crop is None or getattr(crop, "size", 0) == 0:
        return "empty crop"

    height, width = crop.shape[:2]

    if (
        width < MIN_CROP_WIDTH
        or height < MIN_CROP_HEIGHT
    ):
        return (
            f"crop too small "
            f"({width}x{height}; "
            f"min {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT})"
        )

    return None


def save_crop(crop, track_id, role):
    """
    Save one manually labeled crop.

    Filename format:
        role_track_<track_id>_<timestamp>.jpg

    Returns:
        (path, None) on success
        (None, reason) on failure
    """

    reason = crop_reject_reason(crop)

    if reason:
        return None, reason

    timestamp = int(
        time.time() * 1000
    )

    filename = (
        f"{role}_track_{track_id}_{timestamp}.jpg"
    )

    path = os.path.join(
        ROLE_DIRS[role],
        filename,
    )

    success = cv2.imwrite(
        path,
        crop,
    )

    if not success:
        return None, f"could not write {path}"

    return path, None


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

def draw_ui(
    frame,
    tracks,
    selected_track,
    counts,
    status,
):
    """Draw bounding boxes, selection and dataset information."""

    # Draw detected people.
    for i, (
        x1,
        y1,
        x2,
        y2,
        track_id,
        _c,
    ) in enumerate(tracks):

        is_selected = (
            selected_track is not None
            and track_id == selected_track[4]
        )

        color = (
            (0, 220, 0)
            if is_selected
            else (200, 200, 200)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            frame,
            f"[{i + 1}] Person ID {track_id}",
            (
                x1,
                max(20, y1 - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # Selected person information.
    if selected_track is None:
        selected_id = "none"
        selected_box = "none"
    else:
        (
            x1,
            y1,
            x2,
            y2,
            track_id,
            _c,
        ) = selected_track

        selected_id = str(track_id)

        selected_box = (
            f"({x1}, {y1}, {x2}, {y2})  "
            f"{x2 - x1}x{y2 - y1}"
        )

    lines = [
        (
            f"Staff: {counts['staff']}   "
            f"Beneficiary: {counts['beneficiary']}   "
            f"Unknown: {counts['unknown']}"
        ),
        (
            f"Active persons: {len(tracks)}   "
            f"Selected ID: {selected_id}"
        ),
        f"Selected bbox: {selected_box}",
        "1-9 select   S staff   B beneficiary   U unknown   Q quit",
        f"Status: {status}",
    ]

    # Background panel.
    panel_height = 118

    cv2.rectangle(
        frame,
        (6, 6),
        (900, panel_height),
        (0, 0, 0),
        -1,
    )

    # Text.
    y = 22

    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 22


# ---------------------------------------------------------------------------
# Manual save operation
# ---------------------------------------------------------------------------

def try_save(
    frame,
    tracks,
    selected_track,
    key,
    last_save,
):
    """
    Save exactly one crop after an explicit S/B/U press.

    last_save:
        (monotonic_time, track_id)

    Returns:
        (counts_need_refresh, status, last_save)
    """

    now = time.monotonic()

    role = KEY_TO_ROLE[key]

    last_time, last_track_id = last_save

    # ---------------------------------------------------------------
    # Validate detected persons.
    # ---------------------------------------------------------------

    if not tracks:
        status = "rejected: no detected persons"

        print(status)

        return (
            False,
            status,
            last_save,
        )

    # ---------------------------------------------------------------
    # Validate selection.
    # ---------------------------------------------------------------

    if selected_track is None:
        status = (
            "rejected: no selected person; "
            "select a person using 1-9"
        )

        print(status)

        return (
            False,
            status,
            last_save,
        )

    (
        x1,
        y1,
        x2,
        y2,
        track_id,
        _c,
    ) = selected_track

    if track_id is None:
        status = "rejected: invalid track ID"

        print(status)

        return (
            False,
            status,
            last_save,
        )

    # ---------------------------------------------------------------
    # Save cooldown.
    # ---------------------------------------------------------------

    if (
        last_track_id == track_id
        and (
            now - last_time
            < SAVE_COOLDOWN_SECONDS
        )
    ):
        status = (
            "rejected: save cooldown "
            f"({SAVE_COOLDOWN_SECONDS}s) "
            f"for track {track_id}"
        )

        print(status)

        return (
            False,
            status,
            last_save,
        )

    # ---------------------------------------------------------------
    # Make sure selected track still exists.
    # ---------------------------------------------------------------

    current_track = find_track(
        tracks,
        track_id,
    )

    if current_track is None:
        status = (
            f"rejected: selected person ID "
            f"{track_id} left the frame; "
            f"select another person using 1-9"
        )

        print(status)

        return (
            False,
            status,
            last_save,
        )

    # Use the latest bounding box from the current track.
    (
        x1,
        y1,
        x2,
        y2,
        track_id,
        _c,
    ) = current_track

    # ---------------------------------------------------------------
    # Create crop.
    # ---------------------------------------------------------------

    crop = crop_person(
        frame,
        x1,
        y1,
        x2,
        y2,
    )

    # ---------------------------------------------------------------
    # Save.
    #
    # IMPORTANT:
    # This is the ONLY place where save_crop() is called.
    # ---------------------------------------------------------------

    path, reason = save_crop(
        crop,
        track_id,
        role,
    )

    # Record attempted save time for cooldown.
    last_save = (
        now,
        track_id,
    )

    if reason:
        status = (
            f"rejected: {reason}"
        )

        print(status)

        return (
            False,
            status,
            last_save,
        )

    status = (
        f"saved as {role.upper()}  "
        f"ID {track_id}  "
        f"bbox ({x1}, {y1}, {x2}, {y2})  "
        f"{os.path.basename(path)}"
    )

    print(status)

    return (
        True,
        status,
        last_save,
    )


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run(
    cap,
    model,
    confidence,
    fps,
):
    """Run the interactive dataset collection loop."""

    ensure_role_dirs()

    delay_ms = frame_delay_ms(fps)

    # No automatic person selection.
    selected_track_id = None

    counts = dataset_counts()

    status = (
        "select a person using 1-9"
    )

    # No previous save.
    last_save = (
        -1e9,
        None,
    )

    try:
        while True:

            # -------------------------------------------------------
            # Read frame.
            # -------------------------------------------------------

            ok, frame = cap.read()

            if not ok:
                print(
                    "Error: a frame could not be "
                    "read from the source."
                )
                break

            # -------------------------------------------------------
            # YOLO person detection + ByteTrack.
            # -------------------------------------------------------

            tracks = track_people(
                model,
                frame,
                confidence,
            )

            # -------------------------------------------------------
            # Keep selected ByteTrack ID only if still visible.
            # -------------------------------------------------------

            previous_id = selected_track_id

            (
                selected_track_id,
                selected_track,
            ) = resolve_selection(
                tracks,
                selected_track_id,
            )

            # -------------------------------------------------------
            # Detect when selected person disappears.
            # -------------------------------------------------------

            if (
                previous_id is not None
                and selected_track_id is None
            ):
                status = (
                    f"selected person ID "
                    f"{previous_id} left the frame; "
                    f"select another person using 1-9"
                )

            # -------------------------------------------------------
            # Draw UI.
            # -------------------------------------------------------

            draw_ui(
                frame,
                tracks,
                selected_track,
                counts,
                status,
            )

            cv2.imshow(
                WINDOW_TITLE,
                frame,
            )

            # -------------------------------------------------------
            # Keyboard input.
            # -------------------------------------------------------

            key = (
                cv2.waitKey(delay_ms)
                & 0xFF
            )

            # -------------------------------------------------------
            # Quit.
            # -------------------------------------------------------

            if key in (
                ord("q"),
                ord("Q"),
            ):
                print("Quit requested.")
                break

            # -------------------------------------------------------
            # Select person using 1-9.
            # -------------------------------------------------------

            if (
                ord("1")
                <= key
                <= ord("9")
            ):
                index = (
                    key - ord("1")
                )

                if index < len(tracks):

                    selected_track_id = (
                        tracks[index][4]
                    )

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        track_id,
                        _c,
                    ) = tracks[index]

                    status = (
                        f"selected ID {track_id}  "
                        f"bbox "
                        f"({x1}, {y1}, "
                        f"{x2}, {y2})"
                    )

                    print(status)

                else:

                    status = (
                        f"rejected: no person "
                        f"at key {index + 1} "
                        f"({len(tracks)} in frame)"
                    )

                    print(status)

                continue

            # -------------------------------------------------------
            # Save Staff / Beneficiary / Unknown.
            # -------------------------------------------------------

            if key in KEY_TO_ROLE:

                (
                    refreshed,
                    status,
                    last_save,
                ) = try_save(
                    frame,
                    tracks,
                    selected_track,
                    key,
                    last_save,
                )

                if refreshed:
                    counts = dataset_counts()

                continue

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print()

        print_counts(
            dataset_counts(),
            heading="Final dataset counts.",
        )


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Save labeled person crops "
            "(S/B/U). Does not train."
        )
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Video file path. "
            "If omitted, webcam is used."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help=(
            "YOLO person detection confidence "
            "(default: 0.5)."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate optional video source.
    # ---------------------------------------------------------------

    if (
        args.source is not None
        and not os.path.isfile(args.source)
    ):
        print(
            f"Error: video file does not exist: "
            f"{args.source}"
        )
        sys.exit(1)

    # ---------------------------------------------------------------
    # Load YOLO model.
    # ---------------------------------------------------------------

    model = load_model(
        MODEL_PATH
    )

    # ---------------------------------------------------------------
    # Open webcam/video.
    # ---------------------------------------------------------------

    (
        cap,
        source_type,
        source_label,
    ) = open_source(
        args.source
    )

    # ---------------------------------------------------------------
    # Initial information.
    # ---------------------------------------------------------------

    ensure_role_dirs()

    print(
        "Dataset collection started."
    )

    print(
        f"Source : "
        f"{source_type} / "
        f"{source_label}"
    )

    print(
        f"Output : "
        f"{DATASET_ROOT}"
    )

    print()

    print_counts(
        dataset_counts()
    )

    print()

    print(
        "Controls:"
    )

    print(
        "  1-9 = select person"
    )

    print(
        "  S   = save as Staff"
    )

    print(
        "  B   = save as Beneficiary"
    )

    print(
        "  U   = save as Unknown"
    )

    print(
        "  Q   = quit"
    )

    print()

    run(
        cap,
        model,
        args.confidence,
        cap.get(
            cv2.CAP_PROP_FPS
        ),
    )


if __name__ == "__main__":
    main()