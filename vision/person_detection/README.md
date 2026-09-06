# Milestone 2 — Person detection (YOLO)

This module takes frames from Milestone 1 (OpenCV `VideoCapture`) and runs a **pretrained YOLO** model to find people. It draws a box around each detected person in the **current frame**.

It does **not** track people over time. It does **not** mark attendance.

## What YOLO is

**YOLO** (You Only Look Once) is a pretrained object-detection model. You give it an image; it returns a list of objects it thinks it sees, each with a class name, a confidence score, and a box.

This prototype uses the Ultralytics Python package and the small **YOLO11 nano** checkpoint (`yolo11n.pt`), trained on the public **COCO** dataset. We do not train our own model here.

## What object detection means

**Object detection** answers: “What is in this image, and where is it?”

For every object it accepts, the model returns:

1. A **class** (for us: person only)
2. A **bounding box** (where it is)
3. A **confidence** (how sure it is)

Detection is per image. Frame 100 and frame 101 are treated as two separate pictures.

## What a bounding box is

A **bounding box** is a rectangle around a detected object, defined by four numbers: left, top, right, bottom (pixel coordinates).

In this script the rectangle is drawn in green, with a label like `Person 0.91` above it.

## What confidence means

**Confidence** is a number from `0` to `1` (shown as `0.91` for 91%).

- Higher means the model is more sure the box contains a person.
- `--confidence 0.5` means “ignore detections weaker than 50%.”
- Raising it (for example `0.7`) shows fewer boxes and fewer false positives.
- Lowering it (for example `0.3`) shows more boxes and may include mistakes.

## Why we only use the person class

COCO YOLO can detect many classes (car, chair, dog, bottle, …). Sentinal only needs **people** for later attendance work.

The script tells YOLO to predict class `0` (`person`) and also skips any other class if it appears in the results. Cars and furniture are not drawn or counted.

## Why this is NOT tracking

**Tracking** would give the same person a stable ID across frames (person #3 in frame 10 is still person #3 in frame 11).

This milestone does not do that. If someone walks across the camera, each frame is a new set of boxes. There are no IDs and no ByteTrack / BoT-SORT.

## Why this is NOT attendance

**Attendance** would decide *who* is present (for example staff vs beneficiary) over a period of time, usually after tracking and identification.

The on-screen text `Persons detected: 3` is only **how many person boxes are in this frame**. If the same three people stay in view for 100 frames, you still see `3` each time — that is not a headcount over the day, and it is not attendance.

## How to run webcam mode

From the `sentinal-ai` project root:

```bash
python vision/person_detection/person_detector.py
```

The first run may download `yolo11n.pt`. Then a window should show the camera with person boxes.

## How to run video-file mode

```bash
python vision/person_detection/person_detector.py --source "path/to/video.mp4"
```

## How to change the confidence threshold

Default is `0.5`:

```bash
python vision/person_detection/person_detector.py --confidence 0.5
```

Stricter example:

```bash
python vision/person_detection/person_detector.py --confidence 0.7
```

You can combine `--source` and `--confidence`.

## How to quit

Focus the OpenCV window and press **Q**. The capture is released and windows are closed.

## Pipeline

```
Video source (webcam or file)
        ↓
OpenCV VideoCapture  (vision/video_ingestion)
        ↓
Frame
        ↓
YOLO inference       (this module)
        ↓
Person detections only
        ↓
Bounding boxes + live frame count
```
