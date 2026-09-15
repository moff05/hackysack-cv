"""Build a YOLO detection dataset from the hand-verified sack sightings plus
hard negatives (the false-positive candidates found during manual review —
fence spikes, skin shadows, shoe soles, tree canopy — exactly what a
color-threshold detector kept confusing for the ball)."""
import json
import os
import random

import cv2

VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
DATASET_DIR = "/Users/nmoff/claudeprojects/hackysack-cv/dataset"
IMG_W, IMG_H = 1920, 1440

# Hand-verified positives: (frame_idx, (x0, y0, x1, y1))
POSITIVES = [
    (33240, (918, 633, 960, 650)),   # t~1108.2, original manual find (re-measured precisely against this exact frame)
    (32012, (648, 122, 660, 136)),   # t=1067.17
    (26118, (962, 683, 981, 703)),   # t=870.68, clean/sharp
    (27964, (953, 474, 967, 490)),   # t=932.22
    (28276, (638, 322, 656, 340)),   # t=942.62
    (29700, (766, 434, 778, 446)),   # t=990.09
    (35678, (930, 524, 941, 531)),   # t=1189.38
    (35816, (854, 520, 867, 528)),   # t=1193.98
]

random.seed(42)

os.makedirs(f"{DATASET_DIR}/images/train", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/images/val", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels/val", exist_ok=True)

cap = cv2.VideoCapture(VIDEO)


def save_example(frame_idx, bbox, split, tag):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"WARN: could not read frame {frame_idx}")
        return
    name = f"{tag}_{frame_idx}"
    cv2.imwrite(f"{DATASET_DIR}/images/{split}/{name}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    label_path = f"{DATASET_DIR}/labels/{split}/{name}.txt"
    if bbox is None:
        open(label_path, "w").close()  # background image, no objects
        return
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / IMG_W
    cy = (y0 + y1) / 2 / IMG_H
    w = (x1 - x0) / IMG_W
    h = (y1 - y0) / IMG_H
    with open(label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# Positives: 8 total, split ~roughly 6 train / 2 val (val needs at least 1-2 to mean anything)
random.shuffle(POSITIVES)
n_val_pos = 2
for i, (frame_idx, bbox) in enumerate(POSITIVES):
    split = "val" if i < n_val_pos else "train"
    save_example(frame_idx, bbox, split, "pos")

# Hard negatives: every sky-candidate cluster representative that was NOT
# confirmed as the real sack -- these are exactly the confusors (fence
# spikes, skin shadows, shoe soles, tree gaps) a naive detector keeps
# mistaking for the ball.
reps = json.load(open("scripts/data_mining/work/review_crops/reps.json"))
positive_frames = {fi for fi, _ in POSITIVES}
negative_frame_candidates = [r["frame_idx"] for r in reps if r["frame_idx"] not in positive_frames]
random.shuffle(negative_frame_candidates)
negative_frames = sorted(set(negative_frame_candidates[:70]))

# Plus a handful of frames with no flagged candidate at all, for generic
# background diversity (players/trees/fence with nothing unusual detected).
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
generic_negatives = sorted(random.sample(range(0, total_frames, 300), 15))

all_negatives = sorted(set(negative_frames) | set(generic_negatives))
print(f"{len(POSITIVES)} positives, {len(all_negatives)} negatives")

n_val_neg = max(1, len(all_negatives) // 6)
val_negatives = set(random.sample(all_negatives, n_val_neg))
for frame_idx in all_negatives:
    split = "val" if frame_idx in val_negatives else "train"
    save_example(frame_idx, None, split, "neg")

cap.release()
print("dataset built")
