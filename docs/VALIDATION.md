# Validation record

Local validation completed September 29, 2026, using Python 3.13,
Ultralytics 8.4.127, OpenCV 5.0.0, NumPy 2.5.2, and ONNX Runtime 1.30.0.

## Software checks

- 19 offline regression tests pass, including real MP4 decoding/encoding,
  failure cleanup, input/output collision checks, invalid FPS fallback,
  observation lifecycle, ground latching, and conservative event scoring.
- The synthetic CLI completed end to end without loading pose or sack models.
  The continuous mock trajectory was observed on all 120 demo frames.
- The color detector was exercised against an encoded green/black synthetic
  fixture, rather than relying on mocked detections.
- The HTML review loaded browser-playable H.264 with a two-second duration and
  full media readiness. Desktop and 390px mobile layouts had no horizontal overflow.
- Package consistency and Python compilation checks passed.

## Real-footage checks

Private recordings and weights remain local and are excluded from Git.

| Input | Frames analyzed | Observed | Predicted | Missing | Scoring events |
|---|---:|---:|---:|---:|---:|
| 60-second recording | 1,799 | 545 | 617 | 637 | 0 |
| First 10 seconds of kicking recording | 300 | 42 | 89 | 169 | 0 |

The first run used the default 0.92 ground ratio. The second used 0.78 based
on the visible playing area. Both used the existing custom weights and
0.25 confidence threshold. The kicking preview is a limited sample, not an
assessment of the complete 145-second recording.

**These checks establish successful execution, not accurate automatic scoring.**
Sack observation coverage was 30.3% and 14.0%, respectively. Neither recording
has a manually scored reference in this test suite. Zero detected events must
not be interpreted as zero actual touches or drops. The current model/scoring
combination is not validated for unattended scorekeeping.

## Remaining accuracy work

A defensible accuracy claim requires a held-out set with human-verified sack
locations, player identities, touch timestamps, and drop timestamps. Compare
precision/recall and event timing against that set before changing confidence,
contact thresholds, or models. Split recordings by session before training to
avoid leaking nearly identical adjacent frames into validation. Evaluate
perspective-aware ground calibration and contact handling during short occlusions
against those labels. Do not improve reported counts by treating unobserved
Kalman predictions as confirmed events.
