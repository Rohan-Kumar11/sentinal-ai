# Milestone 1 — Video ingestion

This folder is a small OpenCV experiment. It only opens a camera or a video file, reads frames, and shows them. It does not detect people, run YOLO, or count attendance.

## What video ingestion means

**Video ingestion** is the first step of a computer-vision pipeline: taking a live camera or a recorded file and turning it into a sequence of still images (frames).

Every later step (object detection, tracking, attendance) needs those frames. If ingestion fails, nothing downstream can run.

A typical later pipeline looks like:

`OpenCV (this module) → YOLO → Tracking → Attendance`

## What `VideoCapture` does

`cv2.VideoCapture` is OpenCV’s handle to a video source.

- You **open** a webcam (device index, usually `0`) or a file path.
- You **read** one frame at a time with `cap.read()`.
- Each frame is an image (width × height pixels).
- When you are done, you **release** the capture so the camera or file is freed, then **close** the display windows.

This script wraps that loop in `open_source()` and `run_ingestion()` so later milestones can reuse the same reading step.

## Webcam vs video-file input

| Mode | How you select it | What OpenCV opens |
| --- | --- | --- |
| Webcam (default) | No `--source` argument | Camera device index `0` |
| Local video file | `--source "path/to/video.mp4"` | That file on disk |

Use the webcam to test a live stream. Use a file when you want a repeatable clip (same frames every run).

## What FPS, width, and height mean

When the source opens, the script prints:

- **Source type** — webcam or video file.
- **FPS** — frames per second the source reports. A 30 FPS video aims to show 30 images each second. Some webcams report `0`; that just means “unknown,” not a crash.
- **Frame width** — how many pixels across each image.
- **Frame height** — how many pixels down each image.

Example: `640 × 480` at `30` FPS is a common webcam size.

## How to run webcam mode

From the `sentinal-ai` project root, with the project virtual environment activated:

```bash
python vision/video_ingestion/video_capture.py
```

A window should appear with the live camera. The terminal prints source info, then updates the current frame number and elapsed processing time.

## How to run video-file mode

```bash
python vision/video_ingestion/video_capture.py --source "path/to/video.mp4"
```

Replace the path with a real file on your machine. If the file is missing, or OpenCV cannot decode it, the script prints an error and exits.

## How to quit

Focus the OpenCV video window and press **Q**.

The script then releases the capture and closes all OpenCV windows. A video file also stops on its own when there are no more frames.

## Why this is the first step before YOLO

YOLO (and any detector) expects **images**. Video is just images in order.

This milestone only proves we can:

1. Open a real source.
2. Read frames reliably.
3. Show them.
4. Stop cleanly.

Once that is stable, a later milestone can pass each frame into a model. Mixing detection into this script now would hide ingestion bugs behind model errors, so we stop here on purpose.
