# Sack detector training data pipeline

How `models/sack_detector.pt` was built, and how to add more training data
to it later. All scripts are hardcoded to `/Users/nmoff/Desktop/IMG_7251.mov`
(the only source video so far) — update `VIDEO` at the top of each if
training from a different clip.

Run everything from the repo root (`python3 scripts/data_mining/<script>.py`),
with the project's `.venv` active.

## Round 1 (cold start — no model yet)

Getting the first labeled examples when there's no detector to bootstrap
from is the hard part: raw color/shape heuristics (an HSV range for the
sack's color, a "dark blob against open sky" search) threw mostly false
positives — fence-spike tips, skin shadows, shoe soles, backlit palm
fronds. There's no shortcut around this stage: every candidate must be
opened and looked at by eye before it goes in as a label. `build_dataset.py`
assembles the labeled set once you have a `review_crops/reps.json` (hand
verified) plus a list of confirmed-false crops to use as hard negatives.

## Round 2+ (bootstrap from the current model) — the normal path now

Once *any* trained `models/sack_detector.pt` exists, use it to find more
examples instead of color heuristics — it's dramatically more precise
(100% hit rate on manual review last round, vs. constant false positives
from color/sky matching):

1. **`bootstrap_mine.py`** — runs the current model across the full video
   at a very low confidence threshold (0.03) to catch weak-but-real
   detections. Writes every hit to `work/bootstrap_candidates.jsonl`.
   Takes ~20-30 min for the full 20-minute video.
2. **`export_bootstrap_crops.py`** — clusters nearby-in-time/space hits
   (the same physical sighting seen across a few consecutive sampled
   frames) into one representative each, and exports a context crop per
   cluster to `work/bootstrap_review/` for you to look at.
3. **Look at every crop yourself.** Build contact sheets (grids of
   thumbnails) to review efficiently — a few dozen at a time. Do NOT skip
   this and trust the model's own confidence blindly; that's what turns
   this into a legitimate pseudo-labeling bootstrap instead of just
   amplifying the model's own mistakes.
4. **`expand_dataset.py`** — once you know which `work/bootstrap_review/
   reps.json` entries are real (currently: all of them, last round),
   adds them as new positive examples to `dataset/` using the model's own
   predicted bboxes as labels.
5. Retrain (see main README/CLAUDE.md — `ultralytics` `YOLO(...).train(...)`
   on `dataset/data.yaml`), copy the new `best.pt` over
   `models/sack_detector.pt`, and re-check `sack_model_confidence` in
   `config.py` — a better-trained model scores its own correct detections
   much higher, so the acceptance threshold usually needs to come back up
   (was 0.08 after round 1's 8-example model, 0.25 after round 2's 180).

## Known gap as of round 2

Every confirmed example so far is the sack airborne against open sky —
that's what both the original color-threshold false positives and this
model's own predictions stay confined to, since it's all the model has
ever been shown. Ground-level sightings (near feet/pavers, mid-juggle)
are still completely unrepresented. The model can't bootstrap examples of
something it's never seen, so closing this gap needs a fresh round of
manual hunting (the round-1 approach) around ground-level moments
specifically, not another round of `bootstrap_mine.py` alone.
