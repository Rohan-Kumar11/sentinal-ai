# Milestone 5 — Person classification interface (baseline)

This module attaches a **role label** to each tracked person crop:

- Staff
- Beneficiary
- Unknown

There is **no trained classifier yet**. Every person is labeled **Unknown**. This milestone only builds the place where a real model can be connected later.

## Why YOLO cannot determine Staff vs Beneficiary by itself

YOLO11n was trained on **COCO**. It can say “this box is a **person**.” It was not trained on Sentinal roles. Staff and beneficiaries are both people, so detection stops at “person.”

## Detection vs tracking vs classification

| Step | Question |
| --- | --- |
| Detection | Is there a person in this frame? |
| Tracking | Is it the same moving person as last frame? (temporary ID) |
| Classification | Is that person Staff, Beneficiary, or Unknown? |

Classification uses a **person crop** (the pixels inside the bounding box), not the whole frame.

## Why Unknown is the correct baseline

Without a trained model or labeled data, guessing Staff or Beneficiary would be wrong and misleading. **Unknown** is honest: we detected a person, we have a track ID, we do **not** know their role.

`classify_person(person_crop)` always returns `"Unknown"` today. Later, replace that function body with a trained model. Do not use random rules or clothing color as if they were identity.

## Why clothing or ID-card information may later be useful

Uniforms, badges, or ID cards can be **clues** in a future custom model. They are not used here, and clothing alone is not a reliable staff detector.

## Why a custom dataset will eventually be required

A real classifier needs labeled examples from **this** setting (staff vs beneficiaries in these sites). COCO and YOLO do not provide those labels. Dataset collection and training are later work.

## Why this milestone does not perform attendance

Attendance needs role **and** presence over time (who stayed, who counts as present). This module only prints `Unknown` next to a track ID. It does not compute present/absent, duration, or percentages.

## Future classification pipeline

```
Person crop
    → trained classifier (not implemented)
    → Staff | Beneficiary | Unknown
```

Until that model exists, the pipeline is:

```
OpenCV → YOLO11n → ByteTrack → crop → classify_person() → Unknown
```

## How to run

```bash
python vision/person_classification/person_classifier.py
python vision/person_classification/person_classifier.py --source "path/to/video.mp4"
python vision/person_classification/person_classifier.py --confidence 0.5
```

Press **Q** in the video window to quit.
