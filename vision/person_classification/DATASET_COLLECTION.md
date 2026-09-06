# Dataset collection (Milestone 6)

This tool saves **person crops** (images cut from tracking boxes) into:

```
datasets/person_roles/staff/
datasets/person_roles/beneficiary/
datasets/person_roles/unknown/
```

It does **not** train a classifier. Crops are saved only when you press **S**, **B**, or **U**.

## Why a custom Staff / Beneficiary dataset is required

YOLO only labels **person**. Staff and beneficiaries are both people. A future role classifier needs examples from **your** cameras and sites, not a public COCO download.

## What a person crop is

The pixels inside one person’s bounding box. The saved `.jpg` is that rectangle, not the full webcam frame. We do not run face recognition and we do not store names.

## What the labels mean

| Folder | Meaning |
| --- | --- |
| `staff/` | You judged this crop to be staff |
| `beneficiary/` | You judged this crop to be a beneficiary |
| `unknown/` | Role is unclear, mixed, or you do not want to guess |

Labels are **your** judgment for training later. They are not attendance and not identity.

## How to collect balanced samples

Aim for a similar number of staff and beneficiary crops. If one class has 200 images and the other has 10, the future model will be biased.

## Why the same person should not dominate the dataset

If most staff images are one volunteer, the model may memorize that person instead of the role. Collect several different people, on different days, in both classes.

## Lighting, angles, distance, pose

Save crops in bright and dim light, near and far, standing and sitting, different cameras if you have them. A model trained on one desk lamp will fail in a hallway.

## Why Unknown examples are useful

Unknown teaches the future model that “not sure” is valid. Use it for occluded people, crowds, and cases you would not want auto-labeled as staff or beneficiary.

## This dataset is for a future classifier

A later milestone can train on these folders. This milestone only **collects** files.

## How to run

```bash
python vision/person_classification/dataset_collector.py
```

- Numbers **1–9** select which person (shown as `[1] Person ID …`)
- **S** / **B** / **U** save the selected crop
- **Q** quits
