"""Offline regression tests: real video I/O, scoring, and track lifecycle."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import cv2
import numpy as np
from hackysack_cv.config import AppConfig
from hackysack_cv.game_logic import GameStateManager
from hackysack_cv.pipeline import HackySackAnalyzer
from hackysack_cv.sack_tracker import SackTracker, ColorThresholdSackDetector
from hackysack_cv.report import write_review
from generate_test_video import generate
from hackysack_cv.types import CocoKeypoint, PlayerState, SackState


class Detector:
    acceptance_threshold = 0.5
    def __init__(self, values):
        self.values = iter(values)
    def detect(self, *args):
        return next(self.values)


class TrackingTests(unittest.TestCase):
    def track(self, values, **kwargs):
        tracker = SackTracker(Detector(values), AppConfig(**kwargs))
        return [tracker.update(np.zeros((100, 100, 3), np.uint8), i) for i in range(len(values))]

    def test_no_phantom_track_before_first_detection(self):
        self.assertEqual(self.track([None, None]), [None, None])

    def test_lost_track_restarts_without_stale_velocity_or_trail(self):
        states = self.track([(10, 10, 1), (20, 20, 1), None, None, (300, 300, 1)], kalman_max_coast_frames=1)
        self.assertIsNone(states[3])
        self.assertEqual(states[4].position, (300, 300))
        self.assertEqual(states[4].velocity, (0, 0))
        self.assertEqual(list(states[4].trail), [(300, 300)])
        self.assertEqual(len(states[0].trail), 1)

    def test_jump_rejected_and_nonfinite_rejected(self):
        states = self.track([(10, 10, 1), (1000, 1000, 1), (float('nan'), 3, 1)])
        self.assertTrue(states[1].is_coasting)
        self.assertTrue(states[2].is_coasting)

    def test_configured_trail_length(self):
        self.assertEqual(len(self.track([(i, i, 1) for i in range(30)], trail_length=20)[-1].trail), 20)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.game = GameStateManager(AppConfig(event_cooldown_frames=2), 100)
        self.player = PlayerState(7, (0, 0, 50, 100), {CocoKeypoint.LEFT_ANKLE: (50, 75)}, {})
    def state(self, y, vy=0, coast=False):
        return SackState((50, y), (0, vy), is_coasting=coast)
    def test_touch_and_best_round(self):
        self.game.process(0, self.state(75, 5), [self.player])
        event = self.game.process(1, self.state(75, -5), [self.player])
        self.assertEqual(event.touch.player_id, 7)
        self.assertEqual(self.game.stats.best_round_touches, 1)
    def test_resting_ball_is_one_drop_and_rearms_after_lift(self):
        for i in range(20):
            self.game.process(i, self.state(95), [])
        self.assertEqual(self.game.stats.total_drops, 1)
        self.game.process(20, self.state(50), [])
        self.game.process(21, self.state(95), [])
        self.assertEqual(self.game.stats.total_drops, 2)
        self.assertEqual(self.game.stats.player_errors, {})
    def test_coasting_and_missing_do_not_invent_events(self):
        self.game.process(0, self.state(75, 5), [self.player])
        self.game.process(1, self.state(95, 5, True), [self.player])
        self.game.process(2, self.state(75, -5), [self.player])
        self.game.process(3, None, [])
        self.assertEqual(self.game.stats.total_touches, 0)
        self.assertEqual(self.game.stats.total_drops, 0)
    def test_frame_gap_does_not_create_touch(self):
        self.game.process(0, self.state(75, 5), [self.player])
        self.game.process(4, self.state(75, -5), [self.player])
        self.assertEqual(self.game.stats.total_touches, 0)
    def test_distant_player_not_charged_for_drop(self):
        self.game.process(0, SackState((500, 95), (0, 2)), [self.player])
        self.assertEqual(self.game.stats.player_errors, {})
        self.assertEqual(self.game.stats.total_drops, 1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'input.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(12):
            writer.write(np.full((240, 320, 3), 90, np.uint8))
        writer.release()
        self.output = self.root / 'nested/output.mp4'
    def test_real_video_roundtrip_and_report(self):
        report = self.root / 'session.json'
        analyzer = HackySackAnalyzer()
        analyzer.run(str(self.source), str(self.output), force_mock_sack=True, skip_players=True, report_path=str(report))
        data = json.loads(report.read_text())
        self.assertEqual(data['processed_frames'], 12)
        self.assertEqual(data['tracking']['observed_frames'], 12)
        self.assertTrue(data['synthetic'])
        capture = cv2.VideoCapture(str(self.output))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        self.assertEqual(count, 12)
    def test_frame_limit(self):
        analyzer = HackySackAnalyzer()
        analyzer.run(str(self.source), str(self.output), force_mock_sack=True, skip_players=True, max_frames=3)
        self.assertEqual(analyzer.last_report['processed_frames'], 3)
        self.assertTrue(analyzer.last_report['stopped_early'])
    def test_failure_preserves_output_and_removes_partial(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b'previous output')
        with patch('hackysack_cv.pipeline.PlayerTracker', side_effect=RuntimeError('model failed')):
            with self.assertRaises(RuntimeError):
                HackySackAnalyzer().run(str(self.source), str(self.output))
        self.assertEqual(self.output.read_bytes(), b'previous output')
        self.assertEqual(list(self.output.parent.glob('.*.mp4')), [])
    def test_same_path_rejected(self):
        with self.assertRaises(ValueError):
            HackySackAnalyzer().run(str(self.source), str(self.source))
    def test_report_cannot_overwrite_input(self):
        with self.assertRaises(ValueError):
            HackySackAnalyzer().run(str(self.source), str(self.output), report_path=str(self.source))

    def test_color_fixture_exercises_actual_detection(self):
        source = self.root / 'color.mp4'
        generate(str(source), width=640, height=480, duration_s=0.5)
        cap = cv2.VideoCapture(str(source))
        detector = ColorThresholdSackDetector()
        detections = 0
        for i in range(30):
            ok, frame = cap.read()
            self.assertTrue(ok)
            found = detector.detect(frame, i)
            detections += int(found is not None and found[2] >= detector.acceptance_threshold)
        cap.release()
        self.assertGreater(detections, 25)

    def test_review_escapes_input_and_contains_seekable_events(self):
        analyzer = HackySackAnalyzer()
        analyzer.run(str(self.source), str(self.output), force_mock_sack=True, skip_players=True, max_frames=2)
        data = analyzer.last_report
        data['input'] = '/tmp/<script>alert(1)</script>.mp4'
        data['events'] = [{'type': 'touch', 'time_s': 0.5, 'player_id': 7, 'frame_idx': 15}]
        page = write_review(data, str(self.root / 'session.html'))
        content = page.read_text()
        self.assertNotIn('<script>alert(1)</script>', content)
        self.assertIn('data-time="0.500000"', content)
        self.assertIn('Synthetic demo', content)

    def test_writer_open_failure_is_reported(self):
        with patch('hackysack_cv.pipeline.cv2.VideoWriter') as writer:
            writer.return_value.isOpened.return_value = False
            with self.assertRaises(OSError):
                HackySackAnalyzer().run(str(self.source), str(self.output), skip_players=True)
            writer.return_value.release.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_bad_fps_uses_config_fallback(self):
        real_capture = cv2.VideoCapture(str(self.source))
        original_get = real_capture.get
        with patch('hackysack_cv.pipeline.cv2.VideoCapture') as factory:
            capture = factory.return_value
            capture.isOpened.return_value = True
            capture.get.side_effect = lambda key: float('nan') if key == cv2.CAP_PROP_FPS else original_get(key)
            capture.read.side_effect = real_capture.read
            capture.release.side_effect = real_capture.release
            analyzer = HackySackAnalyzer(AppConfig(fps=25))
            analyzer.run(str(self.source), str(self.output), skip_players=True, force_mock_sack=True)
        self.assertEqual(analyzer.last_report['fps'], 25)

    def test_invalid_config(self):
        for kwargs in ({'fps': float('nan')}, {'ground_line_ratio': 2}, {'trail_length': 0}, {'touch_radius_px': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AppConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
