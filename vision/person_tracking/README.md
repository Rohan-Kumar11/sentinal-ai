# Milestone 3 — Person tracking (ByteTrack)

This module takes YOLO **person detections** from Milestone 2 and assigns **temporary track IDs** that can stay with the same moving person across consecutive frames.

It does **not** mark anyone present or absent. It does **not** decide who is staff or a beneficiary. It does **not** recognize faces.

## What object tracking means

**Object tracking** answers: “Is this the same moving object I saw a moment ago?”

Detection looks at **one frame**. Tracking looks at **a sequence of frames** and tries to connect boxes that belong to the same moving person.

Example:

- Frame 100: Person ID 1, Person ID 2
- Frame 101: Person ID 1, Person ID 2

The tracker believes those two objects continued from the previous frame. If someone walks out of view, their ID can disappear. If they come back later, they may get a **new** ID.

## Detection vs tracking

| | Detection (Milestone 2) | Tracking (this milestone) |
| --- | --- | --- |
| Question | “Is there a person in this image?” | “Is it the same person as last frame?” |
| Output | Boxes + confidence | Boxes + **temporary ID** |
| Across frames | Each frame starts over | IDs can persist while the person stays visible |

You still need YOLO. ByteTrack does not replace detection; it **associates** detections over time.

## What a track ID means

A **track ID** is a small integer the tracker assigns for this video or webcam session, for example `1`, `2`, `3`.

The on-screen label looks like `Person ID 1 (0.91)`:

- `1` is the temporary track ID
- `0.91` is the current detection confidence

`Tracked persons: 2` is how many **active tracks are in this frame**, not attendance.

## Why IDs are temporary

The ID is **not**:

- a national ID
- a name
- a face identity
- a database user
- proof that the same human returned tomorrow

If the person is occluded, leaves the frame, or the tracker gets confused, the ID can be lost or reused later for someone else. Treat IDs as **session-only labels for moving blobs**.

## What ByteTrack does (high level)

**ByteTrack** is a multi-object tracker. In this project we use it through Ultralytics (`model.track(..., tracker="bytetrack.yaml", persist=True)`). We do not implement ByteTrack by hand.

At a high level, each frame:

1. YOLO finds person boxes.
2. ByteTrack compares those boxes to tracks it already has (location and motion).
3. Good matches keep the old ID.
4. New unmatched people can start a new track (new ID).
5. Tracks with no matching detection for a while are dropped.

`persist=True` is required so tracker memory survives from one `track()` call to the next.

## Why tracking is needed before attendance

Attendance needs to know that “this person stayed in view” or “this is the same person a few seconds later.” Detection alone cannot do that: it only says “there are three people **right now**.”

Tracking is the bridge: stable-enough IDs over time. Later milestones can use those IDs. This milestone stops at the IDs.

## Why this is NOT attendance

Attendance would compute present/absent, duration, or a percentage. This script never does that. Seeing `Tracked persons: 2` for 100 frames is still just “two tracks in the current frame,” repeated.

## Why this is NOT staff/beneficiary classification

Those labels need extra information (role, clothing, ID card, a trained classifier, etc.). ByteTrack only follows **person** boxes. Everyone is just `Person ID N`.

## How to run webcam mode

From the `sentinal-ai` project root:

```bash
python vision/person_tracking/person_tracker.py
```

## How to run video-file mode

```bash
python vision/person_tracking/person_tracker.py --source "path/to/video.mp4"
```

## How to change the confidence threshold

Default is `0.5`:

```bash
python vision/person_tracking/person_tracker.py --confidence 0.5
```

Higher values (for example `0.7`) keep fewer detections for the tracker to follow.

## How to quit

Focus the OpenCV window and press **Q**. The camera/file is released so the webcam is not left locked.

## Pipeline

```
Webcam / video
        ↓
OpenCV VideoCapture     (vision/video_ingestion)
        ↓
Frame
        ↓
YOLO11n person boxes    (same model as Milestone 2)
        ↓
ByteTrack
        ↓
Temporary track IDs
        ↓
Boxes + Person ID N
```
