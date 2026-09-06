# Milestone 4 — Presence duration

This module measures **how long each temporary ByteTrack ID stays visible** in the current webcam or video session.

It does **not** mark attendance. It does **not** decide staff vs beneficiary. It does **not** recognize faces.

## What presence duration means

**Presence duration** is a stopwatch on a track ID.

Example:

```
Person ID 1 | 12.4s
```

means: during this session, ByteTrack ID `1` was observed for about **12.4 seconds**.

It does **not** mean the person is registered, present for the day, or officially marked present.

## Why tracking IDs are required

Detection only says “there is a person in this frame.” It cannot say “this is the same blob as 2 seconds ago.”

ByteTrack IDs let us add time to the **same row** while that ID stays in view. Without IDs we would restart the clock every frame.

## How duration is calculated

Each ID has a **start time** the first time it appears.

**Webcam (`timing_mode = wall`)**

- Uses wall-clock (`time.perf_counter()`).
- While the ID is visible: `duration` grows with real elapsed time.
- When the ID leaves: that duration is **frozen**.
- YOLO can be slower than the camera. We still count real time in front of the camera, not “number of processed frames / 30.”

**Video file (`timing_mode = media`)**

- Uses the file’s FPS: `duration = visible_frames / FPS`.
- A person in 90 frames of a 30 FPS clip is about **3.0s of video**, even if the computer took 20 seconds to process those frames.
- We do **not** treat one slow inference step as 1/30 of a second of wall time.

If a video reports FPS as 0, the script falls back to wall-clock.

## Active tracks vs presence duration

| | Meaning |
| --- | --- |
| **Active tracks** | How many IDs are in **this frame** (on-screen: `Active tracks: 2`) |
| **Presence duration** | How long each ID has been visible **so far this session** |

A track can leave. Its duration stays in the **session summary**. Active tracks drops. That is not “absent for the day.”

## Why this is NOT attendance

Attendance would use rules (who they are, how long they must stay, present/absent, percentages). This milestone only stores seconds per temporary ID. The next milestone can turn reliable presence into an attendance engine. This one stops at the stopwatch.

## Why a track ID is not a real person’s identity

`Person ID 1` is a **session label** for a moving box. It is not a name, face, or database user. The same human tomorrow can get a different number.

## Why a person can receive a new ID after disappearing

If someone walks out and comes back, ByteTrack may assign a **new** ID. This module treats that as a **new** stopwatch. It does not try to merge old and new IDs (that would be identity, which we do not do here).

## Webcam timing vs video-file timing

- **Webcam:** real-world seconds (wall-clock).
- **Video file:** seconds **inside the clip** (frames ÷ FPS).

That keeps duration meaningful when inference is slower than real-time playback.

## How to run webcam mode

From the `sentinal-ai` project root:

```bash
python vision/presence/presence_tracker.py
```

## How to run video-file mode

```bash
python vision/presence/presence_tracker.py --source "path/to/video.mp4"
```

## How to change confidence

Default is `0.5`:

```bash
python vision/presence/presence_tracker.py --confidence 0.5
```

## How to quit

Focus the OpenCV window and press **Q**.

The script then prints a **session summary**, for example:

```
Session summary:
  Track ID 1: 8.4s
  Track ID 2: 5.7s
```

If nobody was tracked, it says that no tracks were observed. The camera/file is released.

## Pipeline

```
Webcam / video
        ↓
OpenCV
        ↓
YOLO11n person detection
        ↓
ByteTrack (temporary IDs)
        ↓
Presence duration (seconds per ID)
```
