# Hacky Sack CV

Computer vision analyzer for Hacky Sack (footbag) gameplay video. Tracks
players and their pose, tracks the sack, detects touches and drops, and
overlays a live stats dashboard on the output video.

## Setup

```bash
cd claudeprojects/hackysack-cv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pose model (`yolov8n-pose.pt`) auto-downloads from Ultralytics on first
run — no setup needed there.

## Quick start (no custom sack model required)

Generate a synthetic test clip and run the pipeline end-to-end:

```bash
python tests/generate_test_video.py --output test_videos/synthetic_bounce.mp4
python run.py test_videos/synthetic_bounce.mp4 --mock-sack --display
```

`--mock-sack` swaps in a synthetic bouncing trajectory so you can see the
Kalman smoothing, HUD, and game logic working before you've trained or tuned
a real sack detector. Drop it once you're pointing at real footage.

On real footage with people, drop `--mock-sack`. Detection falls back
automatically:

1. `models/sack_detector.pt` if it exists (a model you've trained — see below)
2. HSV color-threshold blob detection otherwise (tune the color range in
   `sack_tracker.py`'s `ColorThresholdSackDetector` defaults to match your
   actual sack's color — defaults assume a bright orange/yellow footbag)

## Usage

```bash
python run.py <video_path> [options]

  --output PATH             annotated video output (default output/annotated.mp4)
  --sack-model PATH         custom-trained sack detector weights
  --pose-model PATH         Ultralytics pose model (default yolov8n-pose.pt)
  --mock-sack               use synthetic trajectory instead of real detection
  --reference-height-m N    assumed average player height, for pixel->meter scale
  --touch-radius-px N       proximity threshold for counting a touch
  --display                 show a live preview window while processing
```

## How it works

| Module | Responsibility |
|---|---|
| `config.py` | All tunable thresholds in one `AppConfig` dataclass |
| `types.py` | Shared dataclasses (`PlayerState`, `SackState`, `GameStats`, ...) and COCO-17 keypoint layout |
| `player_tracker.py` | YOLO pose + ByteTrack (`model.track(..., persist=True)`), extracts ankle/knee/chest keypoints per persistent `player_id` |
| `sack_tracker.py` | Sack detection (trained model / color threshold / mock) + Kalman filter smoothing and occlusion coasting |
| `kalman.py` | Constant-velocity 2D Kalman filter (state: x, y, vx, vy) |
| `kinematics.py` | Pixel->meter scale from assumed player height; sack speed and apex height |
| `game_logic.py` | Touch detection (proximity + velocity inflection) and ground-contact drop detection, with per-player stats |
| `hud.py` | OpenCV overlay: player boxes/skeletons, sack trail, dashboard, leaderboard |
| `pipeline.py` | `HackySackAnalyzer` — wires everything into a single video read/annotate/write loop |
| `run.py` | CLI entrypoint |

## Detection logic

- **Touch**: sack is within `touch_radius_px` of a tracked ankle/knee/chest
  keypoint AND vertical velocity flips from downward to upward beyond a noise
  threshold (`inflection_min_delta_vy`).
- **Drop**: sack's y-position crosses the ground plane
  (`ground_line_ratio * frame_height`) without that upward deflection. Resets
  `current_round_touches` and charges an error to whichever player was
  nearest.
- An `event_cooldown_frames` window prevents one physical contact from being
  double-counted across a few noisy frames.

## Known approximations

- **Ground plane** is a flat horizontal line at a configurable fraction of
  frame height, not a real calibrated floor plane. Fine for a fixed, roughly
  level camera; wrong for extreme angles.
- **Pixel-to-meter scale** comes from assuming everyone in frame is
  `reference_height_m` tall on average (default 1.75m) and measuring their
  bounding-box pixel height. It's a rough estimate, good enough for relative
  speed/height numbers on the HUD, not lab-grade kinematics.
- **Chest keypoint** is synthesized as the shoulder midpoint — COCO pose
  doesn't have a native torso/chest point.

## Training a real sack detector

Once you've got footage:

1. Label sack bounding boxes across a few hundred frames (Roboflow or
   CVAT both work well for this).
2. Train with Ultralytics directly, e.g.:
   ```bash
   yolo detect train data=sack.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```
3. Drop the resulting `best.pt` at `models/sack_detector.pt` — `run.py`
   picks it up automatically over the color-threshold fallback.

## Project layout

```
hackysack-cv/
├── run.py                        # CLI entrypoint
├── requirements.txt
├── src/hackysack_cv/              # package (see table above)
├── tests/generate_test_video.py   # synthetic bounce clip generator
├── test_videos/                   # drop real/synthetic clips here
├── models/                        # sack_detector.pt goes here
└── output/                        # annotated videos land here
```
