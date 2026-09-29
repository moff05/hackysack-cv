# Hacky Sack CV

Turn footbag footage into an annotated video with player identities, pose overlays,
a sack trajectory, estimated touches and drops, a timestamped JSON report, and an
interactive session review page.
Everything processes locally. No hosted service or API key is required.

## Setup

Use Python 3.11–3.13:

```bash
git clone https://github.com/moff05/hackysack-cv.git
cd hackysack-cv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default player tracker uses YOLOv8 pose and BoT-SORT with appearance
re-identification. Ultralytics downloads the pose and ReID models on first use,
which requires internet access. Keep downloaded weights for subsequent offline runs.
Custom sack weights and private training footage are **not included in Git**.

## Try it without downloading models

```bash
python tests/generate_test_video.py --output test_videos/demo.mp4
python run.py test_videos/demo.mp4 --mock-sack --skip-players --output output/demo.mp4
```

This demo needs only NumPy and OpenCV. `--mock-sack` generates positions; it does
not measure detection accuracy. `--skip-players` disables pose inference and touch
attribution. The output visibly labels synthetic runs, and the report records them.

To exercise actual color detection on the synthetic green/black ball:

```bash
python run.py test_videos/demo.mp4 --color-sack --skip-players --output output/color-demo.mp4
```

## Analyze gameplay

```bash
python run.py footage.mp4 --output output/session.mp4
```

If `models/sack_detector.pt` exists, it is selected automatically. Otherwise the
app announces a color detector tuned for a **green/black footbag on neutral ground**.
Other footbag colors need tuned HSV thresholds or trained weights. Foliage and
grass can confuse the color detector.

For a quick check before analyzing an entire recording:

```bash
python run.py footage.mp4 --max-frames 300 --output output/preview.mp4
```

Use `--display` for a live preview; press **Q** to finish and save the processed
portion. Ctrl+C cancels and removes the incomplete video, preserving prior output.
Video outputs are written to a temporary file and replace the requested path only
after processing succeeds. Successful runs overwrite an existing output at that path.
Source and output paths must differ. The output is silent MP4; original audio is
not copied.

### Options

| Option | Purpose |
|---|---|
| `--output PATH` | Annotated `.mp4`; default `output/annotated.mp4` |
| `--report PATH` | JSON report; defaults beside output, with `.json` extension |
| `--sack-model PATH` | Explicit trained weights; missing paths are errors |
| `--color-sack` | Force green/black color detection |
| `--mock-sack` | Synthetic motion for pipeline testing |
| `--skip-players` | Skip pose inference; no player touch attribution |
| `--pose-model PATH` | Alternate Ultralytics pose weights |
| `--sack-confidence N` | Trained sack confidence, default `0.25` |
| `--ground-line-ratio N` | Ground height divided by image height; default `0.92` |
| `--touch-radius-px N` | Player proximity threshold; default `60` |
| `--reference-height-m N` | Assumed player height; default `1.75` |
| `--max-frames N` | Process the first N frames |
| `--display` | Preview window (requires a graphical desktop) |
| `--quiet` | Hide frame progress |

Model and tracker defaults resolve relative to this repository, so `run.py` can be
invoked from another working directory. Input/output paths remain relative to your
current directory. Detector-selection options are mutually exclusive.

## Read the results

Open the generated `.html` file beside the JSON report to review the video, tracking
coverage, player totals, and event log. Click an event timestamp to replay that moment.
The review page works locally without a server or internet connection. Keep its video
and JSON files in their original relative locations when moving or sharing the page.

If `ffmpeg` is available on your system PATH, the CLI encodes H.264 video for browser
playback. Otherwise it writes MPEG-4 Part 2, which desktop players such as VLC can
open; browser support varies and the review page explains the fallback. On macOS,
install FFmpeg with `brew install ffmpeg`; on Ubuntu use `sudo apt install ffmpeg`.


The HUD shows current-round touches, session touches, best round, total drops,
rough speed/height, and a ranked player table. A bottom status bar distinguishes
**detected**, **predicted through occlusion**, and **missing** sack states.

Each JSON report includes:

- Video dimensions, FPS, processed duration, and whether the run was shortened.
- The detector, synthetic mode, and complete configuration used.
- Observed, predicted, and missing frame counts. Observation coverage measures
  how often the tracker accepted a detection; it is **not a precision/accuracy score**.
- Session and per-player statistics, including unattributed drops.
- Touch/drop events with frame index, timestamp, player ID (or null), and position.

## Scoring and limitations

A touch requires consecutive observed frames showing a downward-to-upward velocity
change near a tracked ankle, knee, or shoulder-midpoint chest estimate. Predicted
positions never create scoring events; gaps clear velocity comparisons. This is
conservative: touches hidden behind feet, stalls, and some sideways kicks can be missed.

A drop is an observed sack at or below the configured ground line without an upward
inflection. A ground latch prevents repeated errors while the sack rests there; it
re-arms after the sack rises at least 20 pixels above the line. Only a nearby player
is charged, otherwise the drop is unattributed. A cooldown separates scoring events.

**These are heuristic estimates, not verified scoring.** Set the ground line to the
actual playing surface for each camera view. One horizontal plane is a poor fit for
perspective-heavy scenes, sloped ground, or a moving camera. Player identities can
still change during long occlusions. Speed and height use bounding-box size and an
assumed player height, not calibrated geometry. Training on more varied labeled
footage and comparing against manually scored held-out clips is necessary before
claiming accuracy across venues or players.

## Train a sack detector

1. Label footbag boxes in representative frames, including motion blur, occlusion,
   and ground contact; include negative background frames.
2. Split by source video/session so near-identical frames do not leak into validation.
3. Train using an Ultralytics dataset YAML:
   ```bash
   yolo detect train data=sack.yaml model=yolov8n.pt epochs=100 imgsz=1280
   ```
4. Copy the selected `best.pt` to `models/sack_detector.pt` or pass `--sack-model`.
5. Review held-out detections and timestamped scoring events; tune confidence and
   geometry to your footage. The runtime sack inference resolution is 1280.

Historical data-mining utilities live in [scripts/data_mining](scripts/data_mining/README.md).
They capture experiments on local footage, not a general-purpose training service.

## Development and verification

```bash
python -m unittest discover -s tests -v
```

The offline suite tests the tracking lifecycle, jump rejection, scoring, real MP4
round-tripping, reports, frame limits, and failure cleanup without loading any model.
GitHub Actions runs it on Python 3.11, 3.12, and 3.13 and exercises the demo CLI.
Real-model quality still needs footage-based evaluation; passing these tests does
not establish detector accuracy.

| File | Responsibility |
|---|---|
| `run.py` | CLI, progress, input options, useful failure messages |
| `config.py` | Validated tracking, scoring, and display configuration |
| `player_tracker.py` | YOLO pose + BoT-SORT/ReID player identities |
| `sack_tracker.py` | YOLO/color/mock detectors, ROI, jump gating, occlusion handling |
| `kalman.py` | Constant-velocity 2D filter |
| `game_logic.py` | Conservative touch/drop events and round lifecycle |
| `kinematics.py` | Approximate pixel-to-meter conversions |
| `hud.py` | Video overlay and tracking status |
| `pipeline.py` | Video I/O, analysis, cleanup, session export |
| `report.py` | Standalone HTML session review and event navigation |
| `tests/` | Regression suite and synthetic fixture generator |
