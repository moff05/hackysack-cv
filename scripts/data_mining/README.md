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

## Round 3 (ground-level / non-sky sightings)

`bootstrap_mine.py` alone can't close this gap — it only surfaces what the
current model already believes might be the sack, which after rounds 1-2
means sky-only. Closing it needs a fresh, differently-targeted manual hunt:

1. **`find_kicks.py`** — scans the full video for each sampled frame's
   highest ankle-lift (foot height above that player's own bbox bottom,
   normalized by their height) as a proxy for "someone's mid-kick right
   now." Writes `work/kick_history.jsonl`. Takes ~12-15 min for the full
   video (pose-only, no sack model involved yet).
2. **`find_kick_peaks.py`** — turns that into ~300 distinct, deduped kick
   events (local maxima in lift ratio) as `work/kick_peaks.json`.
3. **`mine_ground_level.py`** — samples a spread of those events, and for
   each pulls the frame ~8 frames *before* the peak (foot still rising,
   ball not yet launched into open sky) rather than at the peak itself.
   Crops tightly around whichever player's foot is highest at that moment.
   Exports to `work/ground_level_review/`.
4. **Look at every crop yourself** — same discipline as round 2, but
   expect a much lower hit rate. Round 3 found 5 confirmed real sightings
   out of 103 reviewed (~5%, vs. round 2's ~100% on model-bootstrapped
   candidates) — this heuristic (a fixed timing offset before a kick peak)
   is far coarser than "the model's own high-confidence prediction," so
   more false leads are expected. Don't mistake a low hit rate here for a
   bug; it's the tradeoff for hunting somewhere the model has zero prior
   signal to bootstrap from. Double-check any "yes" against a tight,
   zoomed native-resolution crop of the exact frame before trusting it —
   thumbnail-sized contact sheets can read as a hit that isn't really
   there (happened once this round, caught by re-checking before adding
   it as a label).
5. **`expand_dataset_ground_level.py`** — adds the confirmed ones (manually
   transcribed bboxes, not auto-derived — round 3 doesn't have a model
   that already gets these right) to `dataset/`. Also worth adding a couple
   of confirmed-empty ground-level crops as hard negatives (same visual
   domain: shoes, pavers, fence — genuinely no ball) rather than only
   ever-more sky negatives.
6. Retrain and re-verify against real footage, same as round 2.

**Round 3 outcome, verified against real footage, not just training metrics:**
still missed a held-out ground-level test case entirely, and even the 4
training examples only scored ~0.08 confidence -- 5 examples were too thin
a fraction of ~155 total positives to actually get learned strongly.

What looked at first like a second, worse problem -- a "false positive" on
plain pavement -- turned out to be a **verification methodology error, not
a real bug**: zooming into the annotated *output* video at that spot showed
nothing, but the actual source frame at the identical pixel coordinates
has a real, correctly-detected ball there the whole time. The tracker's
own drawn marker (a solid dot with real screen radius) was fully covering
the ~20px ball underneath it in the rendered frame, so inspecting the
annotated output made a correct detection look like empty ground. Caught
during round 4 by cross-checking the same coordinates against the raw
source video instead of the annotated one. **Lesson: when investigating a
suspected false positive on a small object, always check the raw source
frame, never the annotated output** -- the overlay can hide the exact
evidence you're trying to see. (The genuine round-3 finding stands: 5
examples was still too thin to reliably teach a new visual domain, which
is why round 4 happened -- just not because of a false-positive regression
that, on closer inspection, was never real.)

## Round 4 (more ground-level, at real volume this time)

Same method as round 3, but exhaustive instead of sampled — `find_kick_peaks_remaining.py`
computes the ~206 kick events round 3's sampling skipped, then
`mine_ground_level_r4_all.py` (same as `mine_ground_level.py` but reads that
file and processes all of it, no further subsampling) mines every one of
them. Reviewing all 206 found 11 more confirmed sightings (~5%, consistent
with round 3's hit rate — confirms round 3's number wasn't a fluke, this
heuristic just has a real ~5% ceiling). Combined with round 3's 5, that's
16 ground-level examples out of ~163 total positives (~10%), a much
healthier fraction than round 3's ~3%.

**Bbox measurement pitfall, hit hard this round:** estimating a bbox's
absolute frame coordinates by eyeballing its position within a *thumbnail*
review crop (e.g. "the ball looks like it's about 60% across, 70% down
this 280x280 image") and doing that arithmetic in your head is unreliable
enough that it produced a *wrong bbox for nearly every one of the first 11*
when checked against a tight, zoomed, native-resolution crop of the exact
frame — several landed on a fence, a car, or empty pavement instead of the
ball. Always generate the zoomed verification crop and look at it before
trusting a manually-transcribed bbox; don't skip straight from "I can see
the ball in the review thumbnail" to writing down coordinates.

**Round 4 outcome, verified against real footage:** a real, clean
improvement -- a held-out ground-level validation example (never seen in
training) went from zero detections in round 3 to correctly detected at
~0.28 confidence. Sky detection unaffected. No confirmed false positives
(see the round-3 correction above about how to actually check this).
Metrics also improved across the board (P 0.928, R 0.909, mAP50 0.919 vs
round 3's 0.876/0.839/0.854).

**Still open after round 4:** 16 examples is better but still thin for a
domain this visually varied (shoes, tree trunks, fence bars, motion blur,
wildly different apparent ball sizes depending on camera distance). All
309 kick events have now been sampled once (at 8 frames before each peak),
so mining more can't just mean "sample the events you haven't looked at
yet" anymore — the next lever is a *second* timing offset per event (e.g.
4 frames before peak instead of 8) to get an independent second sample of
each kick, since the same events likely show the ball at a different,
still-useful position at a different offset.
