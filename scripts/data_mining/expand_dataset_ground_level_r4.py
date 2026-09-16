"""Round 4: adds 11 more hand-verified ground-level sightings on top of
round 3's 5, found by mining the ~206 kick events round 3 didn't sample
(see mine_ground_level.py / README.md). All bboxes measured directly
against zoomed native-resolution crops of the exact frame -- the first
coordinate pass (estimated from thumbnail-sized review crops) was wrong
for nearly every one when checked against a tight crop, so don't skip
that verification step next time either."""
import os
import random

import cv2

DATASET_DIR = "/Users/nmoff/claudeprojects/hackysack-cv/dataset"
VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
IMG_W, IMG_H = 1920, 1440

POSITIVES = [
    (8284, (1126, 1085, 1151, 1105)),
    (13810, (1173, 1073, 1193, 1096)),
    (17462, (1157, 1023, 1169, 1032)),
    (19240, (273, 1033, 297, 1052)),
    (19886, (909, 1147, 956, 1190)),
    (19988, (1003, 1042, 1026, 1062)),
    (21590, (1182, 1054, 1201, 1072)),
    (23642, (1271, 1040, 1292, 1059)),
    (24072, (1168, 879, 1201, 904)),
    (29432, (1336, 1055, 1359, 1075)),
    (30540, (1096, 1045, 1114, 1057)),
]

random.seed(23)
random.shuffle(POSITIVES)
n_val = 2
val_set = POSITIVES[:n_val]
train_set = POSITIVES[n_val:]
print(f"{len(train_set)} train, {len(val_set)} val")

cap = cv2.VideoCapture(VIDEO)


def save(frame_idx, bbox, split):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"WARN: could not read frame {frame_idx}")
        return
    name = f"ground4_{frame_idx}"
    cv2.imwrite(f"{DATASET_DIR}/images/{split}/{name}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / IMG_W
    cy = (y0 + y1) / 2 / IMG_H
    w = (x1 - x0) / IMG_W
    h = (y1 - y0) / IMG_H
    with open(f"{DATASET_DIR}/labels/{split}/{name}.txt", "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


for fi, bbox in train_set:
    save(fi, bbox, "train")
for fi, bbox in val_set:
    save(fi, bbox, "val")

cap.release()
print("done")
