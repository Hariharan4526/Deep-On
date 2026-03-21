#!/usr/bin/env python3
"""
run_tests.py  — Full test suite using Python stdlib unittest only.
No pytest or external test frameworks needed.

Usage:
    python run_tests.py          # run everything
    python run_tests.py -v       # verbose
"""
import os
import sys
import time
import tempfile
import unittest

import cv2
import numpy as np

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ─────────────────────────── Shared helpers ──────────────────────────────────

def _dummy_face_f32(seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((256, 256, 3)).astype(np.float32)


def _dummy_face_u8(seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((256, 256, 3)) * 255).astype(np.uint8)


def _real_face_u8():
    """Synthetic OpenCV-drawn face (more realistic than noise)."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 180
    cv2.ellipse(img, (128, 128), (90, 110), 0, 0, 360, (210, 170, 130), -1)
    cv2.ellipse(img, (88,  100), (18, 10),  0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (168, 100), (18, 10),  0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (88,  100), 7, (50, 30, 10), -1)
    cv2.circle(img, (168, 100), 7, (50, 30, 10), -1)
    cv2.line(img, (128, 120), (118, 150), (160, 120, 90), 3)
    cv2.line(img, (128, 120), (138, 150), (160, 120, 90), 3)
    cv2.ellipse(img, (128, 185), (30, 12), 0, 0, 180, (180, 80, 80), -1)
    return img


def _make_video(n_frames=20, size=256):
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (size, size))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MOD-01  VideoHandler
# ══════════════════════════════════════════════════════════════════════════════

class TestVideoHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.video_path = _make_video(20)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.video_path)

    def test_invalid_path_raises(self):
        from src.video_handler import VideoHandler, VideoHandlerError
        with self.assertRaises(VideoHandlerError):
            VideoHandler("/no/such/file.mp4")

    def test_metadata_keys(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path, frame_skip=1) as vh:
            meta = vh.get_metadata()
        for k in ("fps", "frame_count", "width", "height", "duration", "codec"):
            self.assertIn(k, meta, f"Missing metadata key: {k}")

    def test_metadata_values(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path, frame_skip=1) as vh:
            meta = vh.get_metadata()
        self.assertGreater(meta["frame_count"], 0)
        self.assertGreater(meta["fps"], 0)
        self.assertEqual(meta["width"], 256)
        self.assertEqual(meta["height"], 256)

    def test_frame_shape(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path, frame_skip=1) as vh:
            frame, idx = next(vh.extract_frames())
        self.assertEqual(frame.shape, (256, 256, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_frame_count_no_skip(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path, frame_skip=1) as vh:
            frames = list(vh.extract_frames())
        self.assertEqual(len(frames), 20)

    def test_frame_count_with_skip(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path, frame_skip=5) as vh:
            frames = list(vh.extract_frames())
        self.assertGreaterEqual(len(frames), 3)
        self.assertLessEqual(len(frames), 6)

    def test_indices_monotone(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path, frame_skip=2) as vh:
            indices = [i for _, i in vh.extract_frames()]
        self.assertEqual(indices, sorted(indices))

    def test_read_single_frame(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path) as vh:
            f = vh.read_frame(0)
        self.assertIsNotNone(f)
        self.assertEqual(f.shape, (256, 256, 3))

    def test_context_manager_releases(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path) as vh:
            pass
        self.assertIsNone(vh._cap)

    def test_fps_helper(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path) as vh:
            fps = vh.get_fps()
        self.assertGreater(fps, 0)

    def test_frame_count_helper(self):
        from src.video_handler import VideoHandler
        with VideoHandler(self.video_path) as vh:
            n = vh.get_frame_count()
        self.assertEqual(n, 20)


# ══════════════════════════════════════════════════════════════════════════════
# MOD-02  FaceDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestFaceDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.face_detection import FaceDetector
        cls.det = FaceDetector(confidence_threshold=0.5)
        cls.face_u8  = _real_face_u8()
        cls.noise_u8 = _dummy_face_u8()

    @classmethod
    def tearDownClass(cls):
        cls.det.close()

    def test_detect_faces_returns_list(self):
        result = self.det.detect_faces(self.face_u8)
        self.assertIsInstance(result, list)

    def test_detect_faces_dict_structure(self):
        result = self.det.detect_faces(self.face_u8)
        for det in result:
            self.assertIn("bbox",       det)
            self.assertIn("confidence", det)
            self.assertIn("landmarks",  det)
            self.assertEqual(det["landmarks"].shape, (5, 2))
            self.assertGreaterEqual(det["confidence"], 0)
            self.assertLessEqual(   det["confidence"], 1)

    def test_align_face_output_shape(self):
        lm = np.array(
            [[72, 100], [184, 100], [128, 148], [84, 196], [172, 196]],
            dtype=np.float32,
        )
        aligned = self.det.align_face(self.face_u8, lm)
        self.assertEqual(aligned.shape, (256, 256, 3))
        self.assertEqual(aligned.dtype, np.float32)

    def test_align_face_is_normalised(self):
        lm = np.array(
            [[72, 100], [184, 100], [128, 148], [84, 196], [172, 196]],
            dtype=np.float32,
        )
        aligned = self.det.align_face(self.face_u8, lm)
        self.assertLess(aligned.mean(), 2.0,
                        "Should be near 0 after ImageNet normalisation")
        self.assertLess(aligned.min(), 0.0,
                        "Normalised values should include negatives")

    def test_extract_face_crops_type(self):
        crops = self.det.extract_face_crops(self.face_u8)
        self.assertIsInstance(crops, list)
        for c in crops:
            self.assertEqual(c.shape, (256, 256, 3))


# ══════════════════════════════════════════════════════════════════════════════
# MOD-04  OpticalFlowAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class TestOpticalFlowAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.optical_flow import OpticalFlowAnalyzer
        cls.oa   = OpticalFlowAnalyzer()
        cls.f32  = _dummy_face_f32()
        cls.u8   = _dummy_face_u8()

    def test_flow_shape_u8(self):
        flow = self.oa.compute_optical_flow(self.u8, self.u8)
        self.assertEqual(flow.shape, (256, 256, 2))

    def test_flow_shape_f32(self):
        flow = self.oa.compute_optical_flow(self.f32, self.f32)
        self.assertEqual(flow.shape, (256, 256, 2))

    def test_identical_frames_near_zero(self):
        flow = self.oa.compute_optical_flow(self.u8, self.u8)
        self.assertLess(abs(flow).mean(), 1.0)

    def test_metrics_keys(self):
        flow = self.oa.compute_optical_flow(self.u8, self.u8)
        m    = self.oa.extract_motion_metrics(flow)
        for k in ("mean_magnitude", "std_magnitude", "max_magnitude",
                  "smoothness", "occlusion_ratio"):
            self.assertIn(k, m)

    def test_metrics_non_negative(self):
        flow = self.oa.compute_optical_flow(self.u8, self.u8)
        m    = self.oa.extract_motion_metrics(flow)
        for k, v in m.items():
            self.assertGreaterEqual(v, 0, f"{k} is negative")

    def test_score_range(self):
        flow  = self.oa.compute_optical_flow(self.u8, self.u8)
        score = self.oa.score_optical_flow([flow] * 5)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(   score, 1.0)

    def test_empty_flow_sequence_score(self):
        score = self.oa.score_optical_flow([])
        self.assertEqual(score, 0.5)

    def test_anomaly_empty(self):
        s, a = self.oa.detect_temporal_anomalies([])
        self.assertEqual(s, 0.0)
        self.assertEqual(a, [])

    def test_process_frame_pair(self):
        flow, metrics = self.oa.process_frame_pair(self.u8, self.u8)
        self.assertEqual(flow.shape, (256, 256, 2))
        self.assertIn("smoothness", metrics)


# ══════════════════════════════════════════════════════════════════════════════
# MOD-05  FrequencyAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequencyAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.frequency_analyzer import FrequencyAnalyzer
        cls.fa  = FrequencyAnalyzer()
        cls.f32 = _dummy_face_f32()
        cls.u8  = _dummy_face_u8()

    def test_fft_keys(self):
        feats = self.fa.compute_fft(self.f32)
        for k in ("low_energy_ratio", "high_energy_ratio", "isotropy",
                  "peak_ratio", "fft_anomaly_score"):
            self.assertIn(k, feats)

    def test_fft_energy_in_unit_interval(self):
        feats = self.fa.compute_fft(self.f32)
        self.assertGreaterEqual(feats["low_energy_ratio"],  0)
        self.assertLessEqual(   feats["low_energy_ratio"],  1)
        self.assertGreaterEqual(feats["high_energy_ratio"], 0)
        self.assertLessEqual(   feats["high_energy_ratio"], 1)

    def test_fft_anomaly_range(self):
        score = self.fa.compute_fft(self.f32)["fft_anomaly_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(   score, 1.0)

    def test_dct_keys(self):
        feats = self.fa.compute_dct(self.f32)
        for k in ("high_freq_variance", "high_freq_energy_ratio", "dct_anomaly_score"):
            self.assertIn(k, feats)

    def test_dct_variance_non_negative(self):
        self.assertGreaterEqual(
            self.fa.compute_dct(self.f32)["high_freq_variance"], 0
        )

    def test_score_range_f32(self):
        p = self.fa.score_frequency_domain(self.f32)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(   p, 1.0)

    def test_score_range_u8(self):
        p = self.fa.score_frequency_domain(self.u8)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(   p, 1.0)

    def test_all_features_has_p(self):
        feats = self.fa.get_all_features(self.f32)
        self.assertIn("p_frequency", feats)
        self.assertGreaterEqual(feats["p_frequency"], 0)
        self.assertLessEqual(   feats["p_frequency"], 1)

    def test_noisy_higher_variance_than_blurred(self):
        rng   = np.random.default_rng(7)
        noisy = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        smooth = cv2.GaussianBlur(noisy, (31, 31), 0)
        n_var = self.fa.compute_dct(noisy) ["high_freq_variance"]
        s_var = self.fa.compute_dct(smooth)["high_freq_variance"]
        self.assertGreater(n_var, s_var,
            "Noisy image should have higher high-freq DCT variance than blurred")


# ══════════════════════════════════════════════════════════════════════════════
# MOD-06  LandmarkValidator
# ══════════════════════════════════════════════════════════════════════════════

class TestLandmarkValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.landmark_validator import LandmarkValidator
        cls.lv       = LandmarkValidator()
        cls.f32      = _dummy_face_f32()
        cls.face_u8  = _real_face_u8()

    @classmethod
    def tearDownClass(cls):
        cls.lv.close()

    def test_detect_returns_none_or_array(self):
        result = self.lv.detect_landmarks(self.f32)
        # MediaPipe backend  → (468, 3);  Geometric backend → (20, 2)
        valid_shapes = {(468, 3), (20, 2)}
        self.assertTrue(
            result is None or
            (isinstance(result, np.ndarray) and result.shape in valid_shapes),
            f"Unexpected shape: {None if result is None else result.shape}",
        )

    def test_score_range_noise(self):
        p = self.lv.score_landmark_geometry(self.f32)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(   p, 1.0)

    def test_score_range_face(self):
        p = self.lv.score_landmark_geometry(self.face_u8)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(   p, 1.0)

    def test_full_analysis_structure(self):
        a = self.lv.get_full_analysis(self.face_u8)
        self.assertIn("detected",    a)
        self.assertIn("p_landmarks", a)
        self.assertGreaterEqual(a["p_landmarks"], 0.0)
        self.assertLessEqual(   a["p_landmarks"], 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# MOD-07  EnsembleClassifier
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsembleClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.ensemble_classifier import EnsembleClassifier
        cls.clf = EnsembleClassifier()

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(self.clf.weights.sum(), 1.0, places=6)

    def test_aggregate_range(self):
        for args in [(0.9, 0.8, 0.7, 0.6), (0.1,)*4, (0.5,)*4]:
            p = self.clf.aggregate_predictions(*args)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(   p, 1.0)

    def test_aggregate_high_all_high(self):
        self.assertGreater(self.clf.aggregate_predictions(0.9, 0.9, 0.9, 0.9), 0.5)

    def test_aggregate_all_low(self):
        self.assertLess(self.clf.aggregate_predictions(0.1, 0.1, 0.1, 0.1), 0.5)

    def test_temporal_smooth_length(self):
        preds = [0.1, 0.9, 0.5, 0.3, 0.7]
        self.assertEqual(len(self.clf.temporal_smooth(preds)), len(preds))

    def test_temporal_smooth_range(self):
        for s in self.clf.temporal_smooth([0.0, 1.0, 0.0, 1.0]):
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(   s, 1.0)

    def test_temporal_smooth_empty(self):
        self.assertEqual(self.clf.temporal_smooth([]), [])

    def test_calibrate_range(self):
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            cal = self.clf.calibrate_confidence(p)
            self.assertGreaterEqual(cal, 0.0)
            self.assertLessEqual(   cal, 1.0)

    def test_classify_deepfake(self):
        label, conf = self.clf.classify(0.95, calibrate=False)
        self.assertEqual(label, "Deepfake")
        self.assertGreaterEqual(conf, 0)
        self.assertLessEqual(   conf, 100)

    def test_classify_authentic(self):
        label, conf = self.clf.classify(0.05, calibrate=False)
        self.assertEqual(label, "Authentic")

    def test_process_frame_keys(self):
        self.clf.reset_ema()
        r = self.clf.process_frame(0.7, 0.6, 0.5, 0.4)
        for k in ("ensemble_prob", "smoothed_prob", "calibrated_prob",
                  "label", "confidence"):
            self.assertIn(k, r)

    def test_process_frame_temporal_damping(self):
        self.clf.reset_ema()
        self.clf.process_frame(0.1, 0.1, 0.1, 0.1)      # low frame
        r = self.clf.process_frame(0.9, 0.9, 0.9, 0.9)  # high frame
        self.assertLess(r["smoothed_prob"], 0.9,
                        "EMA should dampen sudden jump to 0.9")

    def test_aggregate_video_empty(self):
        v = self.clf.aggregate_video_predictions([])
        self.assertEqual(v["label"], "Unknown")

    def test_aggregate_video_verdict(self):
        self.clf.reset_ema()
        frames  = [self.clf.process_frame(0.8, 0.7, 0.6, 0.5) for _ in range(10)]
        verdict = self.clf.aggregate_video_predictions(frames)
        self.assertIn(verdict["label"], ("Deepfake", "Authentic"))
        self.assertEqual(verdict["frames_processed"], 10)


# ══════════════════════════════════════════════════════════════════════════════
# Integration  MOD-01 → 04 → 05 → 07
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.video_path = _make_video(20)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.video_path)

    def test_video_to_ensemble_pipeline(self):
        from src.video_handler       import VideoHandler
        from src.optical_flow        import OpticalFlowAnalyzer
        from src.frequency_analyzer  import FrequencyAnalyzer
        from src.ensemble_classifier import EnsembleClassifier
        from src.utils               import normalize_image

        vh  = VideoHandler(self.video_path, frame_skip=3)
        oa  = OpticalFlowAnalyzer()
        fa  = FrequencyAnalyzer()
        clf = EnsembleClassifier()
        clf.reset_ema()

        results    = []
        prev_f32   = None

        for frame, _ in vh.extract_frames():
            face    = cv2.resize(frame, (256, 256))
            face_f32 = normalize_image(face)

            p_freq = fa.score_frequency_domain(face_f32)
            p_flow = 0.5
            if prev_f32 is not None:
                flow, _ = oa.process_frame_pair(prev_f32, face_f32)
                p_flow  = oa.score_optical_flow([flow])
            prev_f32 = face_f32

            results.append(clf.process_frame(0.5, p_flow, p_freq, 0.5))

        verdict = clf.aggregate_video_predictions(results)
        vh.release()

        self.assertIn(verdict["label"], ("Deepfake", "Authentic"))
        self.assertGreaterEqual(verdict["probability"], 0)
        self.assertLessEqual(   verdict["probability"], 1)
        self.assertEqual(verdict["frames_processed"], len(results))


# ══════════════════════════════════════════════════════════════════════════════
# Performance
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance(unittest.TestCase):

    def _bench(self, fn, n=10):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times), sum(times) / n

    def test_frequency_latency(self):
        from src.frequency_analyzer import FrequencyAnalyzer
        fa  = FrequencyAnalyzer()
        f32 = _dummy_face_f32()
        _, avg = self._bench(lambda: fa.score_frequency_domain(f32))
        print(f"\n  FrequencyAnalyzer: {avg*1000:.1f} ms avg")
        self.assertLess(avg, 1.0, f"Too slow: {avg*1000:.1f} ms")

    def test_optical_flow_latency(self):
        from src.optical_flow import OpticalFlowAnalyzer
        oa = OpticalFlowAnalyzer()
        u8 = _dummy_face_u8()
        _, avg = self._bench(lambda: oa.compute_optical_flow(u8, u8))
        print(f"\n  OpticalFlow: {avg*1000:.1f} ms avg")
        self.assertLess(avg, 0.5, f"Too slow: {avg*1000:.1f} ms")

    def test_ensemble_latency(self):
        from src.ensemble_classifier import EnsembleClassifier
        clf = EnsembleClassifier()
        clf.reset_ema()
        _, avg = self._bench(lambda: clf.process_frame(0.7, 0.6, 0.5, 0.4))
        print(f"\n  Ensemble: {avg*1000:.4f} ms avg")
        self.assertLess(avg, 0.01)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# MOD-03  CNNExtractor  (geometric fallback — no torch needed)
# ══════════════════════════════════════════════════════════════════════════════

class TestCNNExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.cnn_extractor import CNNExtractor
        cls.cnn  = CNNExtractor()
        cls.f32  = _dummy_face_f32()

    def test_using_fallback_flag(self):
        # Will be True here (no torch), False on GPU machine
        self.assertIsInstance(self.cnn.using_fallback, bool)

    def test_extract_features_returns_tuple(self):
        feat, prob = self.cnn.extract_features(self.f32)
        self.assertIsInstance(feat, np.ndarray)
        self.assertIsInstance(prob, float)

    def test_feature_dim_positive(self):
        feat, _ = self.cnn.extract_features(self.f32)
        self.assertGreater(feat.shape[0], 0)

    def test_prob_in_unit_interval(self):
        _, prob = self.cnn.extract_features(self.f32)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(   prob, 1.0)

    def test_score_image(self):
        p = self.cnn.score_image(self.f32)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(   p, 1.0)

    def test_batch_extract_features_shape(self):
        faces = [self.f32] * 4
        feats, probs = self.cnn.batch_extract_features(faces)
        self.assertEqual(feats.shape[0], 4)
        self.assertEqual(probs.shape[0], 4)

    def test_batch_probs_in_range(self):
        feats, probs = self.cnn.batch_extract_features([self.f32] * 3)
        for p in probs:
            self.assertGreaterEqual(float(p), 0.0)
            self.assertLessEqual(   float(p), 1.0)

    def test_forward_with_attention(self):
        feat, prob, fm = self.cnn.forward_with_attention(self.f32)
        self.assertIsInstance(feat, np.ndarray)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(   prob, 1.0)
        # fm is None in fallback mode — that is expected


# ══════════════════════════════════════════════════════════════════════════════
# VisualizationModule
# ══════════════════════════════════════════════════════════════════════════════

class TestVisualizationModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        from src.visualization import VisualizationModule
        from src.cnn_extractor import CNNExtractor
        cls.tmpdir = tempfile.mkdtemp()
        cls.viz    = VisualizationModule(
            cnn_model=CNNExtractor(),
            output_dir=cls.tmpdir,
        )
        cls.f32    = _dummy_face_f32()
        cls.frame_results = []
        from src.ensemble_classifier import EnsembleClassifier
        clf = EnsembleClassifier(); clf.reset_ema()
        for i in range(10):
            r = clf.process_frame(0.6+i*0.02, 0.5, 0.5, 0.5)
            r["frame_index"] = i
            cls.frame_results.append(r)

    def test_generate_gradcam_shape(self):
        overlay = self.viz.generate_gradcam(self.f32)
        self.assertEqual(overlay.shape, (256, 256, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_generate_gradcam_saves_file(self):
        path = os.path.join(self.tmpdir, "test_gc.png")
        self.viz.generate_gradcam(self.f32, save_path=path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_temporal_plot_saves_file(self):
        path = os.path.join(self.tmpdir, "test_temporal.png")
        saved = self.viz.visualize_temporal_predictions(self.frame_results, save_path=path)
        self.assertEqual(saved, path)
        self.assertTrue(os.path.exists(path))

    def test_temporal_plot_empty(self):
        result = self.viz.visualize_temporal_predictions([])
        self.assertEqual(result, "")

    def test_radar_chart_saves_file(self):
        path = os.path.join(self.tmpdir, "test_radar.png")
        scores = {"cnn": 0.7, "optical_flow": 0.6, "frequency": 0.5, "landmarks": 0.4}
        saved = self.viz.visualize_pathway_scores(scores, "Deepfake", 75.0, save_path=path)
        self.assertEqual(saved, path)
        self.assertTrue(os.path.exists(path))

    def test_create_result_report(self):
        report = self.viz.create_result_report(
            video_path="test.mp4",
            classification="Deepfake",
            confidence=82.5,
            probability=0.912,
            pathway_scores={"cnn":0.9,"optical_flow":0.8,"frequency":0.7,"landmarks":0.6},
            processing_time=3.14,
            frames_processed=20,
            fps=12.5,
            output_dir=self.tmpdir,
        )
        self.assertEqual(report["classification"], "Deepfake")
        self.assertAlmostEqual(report["confidence"], 82.5)
        self.assertIn("timestamp", report)


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline integration (all modules together)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.video_path = _make_video(n_frames=30, size=256)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.video_path)

    def test_deepfake_detector_e2e(self):
        import tempfile
        from src.inference import DeepfakeDetector
        outdir  = tempfile.mkdtemp()
        det     = DeepfakeDetector()
        report  = det.process_video(self.video_path, output_dir=outdir,
                                    save_gradcam=False,
                                    save_temporal=True,
                                    save_radar=True)
        det.close()

        # Check report structure
        self.assertIn("classification",    report)
        self.assertIn("confidence",        report)
        self.assertIn("probability",       report)
        self.assertIn("frames_processed",  report)
        self.assertIn("pathway_scores",    report)

        # Values in valid ranges
        self.assertIn(report["classification"], ("Deepfake", "Authentic"))
        self.assertGreaterEqual(report["confidence"], 0)
        self.assertLessEqual(   report["confidence"], 100)
        self.assertGreaterEqual(report["probability"], 0)
        self.assertLessEqual(   report["probability"], 1)
        self.assertGreater(report["frames_processed"], 0)

    def test_batch_process(self):
        import tempfile
        from src.inference import DeepfakeDetector
        outdir  = tempfile.mkdtemp()
        det     = DeepfakeDetector()
        results = det.batch_process_videos(
            [self.video_path, self.video_path], output_dir=outdir
        )
        det.close()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertNotIn("error", r)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)