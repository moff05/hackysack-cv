"""Computes the kick events find_kick_peaks.json's sampling (round 3) left
out, so round 4+ can mine them without re-reviewing round 3's candidates.
Run find_kick_peaks.py first."""
import json

peaks = json.load(open("scripts/data_mining/work/kick_peaks.json"))
peaks.sort(key=lambda p: p["time_s"])
STEP = max(1, len(peaks) // 90)  # matches mine_ground_level.py's sampling
already_sampled_idx = set(range(0, len(peaks), STEP))
remaining = [p for i, p in enumerate(peaks) if i not in already_sampled_idx]
print(f"total {len(peaks)}, already sampled {len(already_sampled_idx)}, remaining {len(remaining)}")
with open("scripts/data_mining/work/kick_peaks_remaining.json", "w") as f:
    json.dump(remaining, f)
