"""Full end-to-end deepfake detection inference pipeline.

Ties together all 8 modules (MOD-01 → MOD-08) into a single
DeepfakeDetector class that processes one video at a time or a batch.
"""
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .video_handler       import VideoHandler
from .face_detection      import FaceDetector
from .cnn_extractor       import CNNExtractor
from .optical_flow        import OpticalFlowAnalyzer
from .frequency_analyzer  import FrequencyAnalyzer
from .landmark_validator  import LandmarkValidator
from .ensemble_classifier import EnsembleClassifier
from .visualization       import VisualizationModule
from .config              import Config, load_config
from .utils               import setup_logger, ensure_dirs, normalize_image

logger = setup_logger(__name__)


class DeepfakeDetector:
    """Full multi-modal deepfake detection pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: str = "config.yaml",
    ):
        self.cfg = config or load_config(config_path)
        self._build_modules()

    # ── Module construction ───────────────────────────────────────────────────

    def _build_modules(self) -> None:
        cfg = self.cfg

        self.face_det = FaceDetector(
            confidence_threshold=cfg.detection.face_confidence_threshold,
            target_size=cfg.detection.target_face_size,
            normalization_mean=tuple(cfg.normalization.mean),
            normalization_std=tuple(cfg.normalization.std),
        )
        self.cnn = CNNExtractor(
            model_name=cfg.model.cnn_model,
            pretrained=cfg.model.pretrained,
            device=cfg.hardware.device,
            feature_dim=cfg.model.feature_dim,
        )
        self.optflow  = OpticalFlowAnalyzer()
        self.freq     = FrequencyAnalyzer()
        self.lm_val   = LandmarkValidator()
        self.ensemble = EnsembleClassifier(
            cnn_weight=cfg.ensemble.cnn_weight,
            optflow_weight=cfg.ensemble.optical_flow_weight,
            freq_weight=cfg.ensemble.frequency_weight,
            lm_weight=cfg.ensemble.landmarks_weight,
            temperature=cfg.ensemble.temperature,
            temporal_alpha=cfg.ensemble.temporal_alpha,
            classification_threshold=cfg.ensemble.classification_threshold,
        )
        self.viz = VisualizationModule(
            cnn_model=self.cnn,
            output_dir=cfg.visualizations,
        )
        logger.info("DeepfakeDetector: all modules ready.")

    # ── Single video ──────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_gradcam:     bool = True,
        save_temporal:    bool = True,
        save_radar:       bool = True,
    ) -> Dict:
        """Run full deepfake detection on one video file.

        Returns a report dict (and saves a JSON + optional visualisations).
        """
        out_dir = output_dir or self.cfg.predictions
        vis_dir = self.cfg.visualizations
        ensure_dirs(out_dir, vis_dir)

        t_start = time.perf_counter()

        # ── MOD-01: load video ────────────────────────────────────────────────
        vh   = VideoHandler(video_path, frame_skip=self.cfg.detection.frame_skip)
        meta = vh.get_metadata()
        logger.info(f"Processing {vh}")

        self.ensemble.reset_ema()

        frame_results: List[Dict] = []
        flow_buffer:   List       = []        # last N flow fields for scoring
        FLOW_BUF_LEN  = 5
        prev_face_f32: Optional[np.ndarray] = None

        # For Grad-CAM we capture the first high-scoring face
        gradcam_path:  Optional[str] = None
        gradcam_saved: bool          = False

        # ── Main frame loop ───────────────────────────────────────────────────
        for frame_rgb, frame_idx in vh.extract_frames():

            # MOD-02: face detection + alignment
            crops = self.face_det.extract_face_crops(
                frame_rgb,
                min_size=self.cfg.detection.face_size_min,
                max_size=self.cfg.detection.face_size_max,
            )
            if not crops:
                # No face found — use whole frame downscaled as fallback
                face_f32 = normalize_image(
                    cv2.resize(frame_rgb, (256, 256)),
                    self.cfg.normalization.mean,
                    self.cfg.normalization.std,
                )
            else:
                face_f32 = crops[0]

            # MOD-03: CNN score
            p_cnn = self.cnn.score_image(face_f32)

            # MOD-04: optical flow (needs previous frame)
            if prev_face_f32 is not None:
                flow, _ = self.optflow.process_frame_pair(prev_face_f32, face_f32)
                flow_buffer.append(flow)
                if len(flow_buffer) > FLOW_BUF_LEN:
                    flow_buffer.pop(0)
                p_optflow = self.optflow.score_optical_flow(flow_buffer)
            else:
                p_optflow = 0.5
            prev_face_f32 = face_f32

            # MOD-05: frequency domain
            p_freq = self.freq.score_frequency_domain(face_f32)

            # MOD-06: landmark geometry
            p_lm = self.lm_val.score_landmark_geometry(face_f32)

            # MOD-07: ensemble + temporal smoothing
            result = self.ensemble.process_frame(p_cnn, p_optflow, p_freq, p_lm)
            result["frame_index"] = frame_idx
            frame_results.append(result)

            # MOD-08: save Grad-CAM for first high-confidence deepfake frame
            if (save_gradcam and not gradcam_saved
                    and result["calibrated_prob"] > 0.65):
                base     = os.path.splitext(os.path.basename(video_path))[0]
                gc_path  = os.path.join(vis_dir, f"{base}_gradcam.png")
                self.viz.generate_gradcam(face_f32, save_path=gc_path)
                gradcam_path  = gc_path
                gradcam_saved = True

        vh.release()
        t_elapsed = time.perf_counter() - t_start

        # ── Video-level verdict ───────────────────────────────────────────────
        verdict   = self.ensemble.aggregate_video_predictions(frame_results)
        actual_fps = len(frame_results) / max(t_elapsed, 1e-6)
        base       = os.path.splitext(os.path.basename(video_path))[0]

        # MOD-08: temporal plot
        temporal_path = None
        if save_temporal and frame_results:
            temporal_path = os.path.join(vis_dir, f"{base}_temporal.png")
            self.viz.visualize_temporal_predictions(
                frame_results, save_path=temporal_path
            )

        # MOD-08: pathway radar chart
        radar_path = None
        if save_radar and verdict.get("pathway_means"):
            radar_path = os.path.join(vis_dir, f"{base}_radar.png")
            self.viz.visualize_pathway_scores(
                verdict["pathway_means"],
                label=verdict["label"],
                confidence=verdict["confidence"],
                save_path=radar_path,
            )

        # MOD-08: JSON report
        report = self.viz.create_result_report(
            video_path=video_path,
            classification=verdict["label"],
            confidence=verdict["confidence"],
            probability=verdict["probability"],
            pathway_scores=verdict.get("pathway_means", {}),
            processing_time=t_elapsed,
            frames_processed=len(frame_results),
            fps=actual_fps,
            gradcam_path=gradcam_path,
            temporal_path=temporal_path,
            radar_path=radar_path,
            output_dir=out_dir,
        )

        logger.info(
            f"[RESULT] {os.path.basename(video_path)} → "
            f"{verdict['label']} ({verdict['confidence']:.1f}% conf)  "
            f"{len(frame_results)} frames  {actual_fps:.1f} FPS"
        )
        return report

    # ── Batch processing ──────────────────────────────────────────────────────

    def batch_process_videos(
        self,
        video_list: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """Process multiple videos and return list of report dicts."""
        results = []
        for i, vp in enumerate(video_list, 1):
            logger.info(f"[{i}/{len(video_list)}] {vp}")
            try:
                r = self.process_video(vp, output_dir=output_dir)
            except Exception as e:
                logger.error(f"Failed on {vp}: {e}")
                r = {"video_path": vp, "error": str(e)}
            results.append(r)

        ok        = [r for r in results if "error" not in r]
        n_fake    = sum(1 for r in ok if r.get("classification") == "Deepfake")
        logger.info(
            f"Batch done: {len(ok)}/{len(video_list)} succeeded  |  "
            f"{n_fake} deepfake(s) detected."
        )
        return results

    # ── Resource management ───────────────────────────────────────────────────

    def close(self) -> None:
        self.face_det.close()
        self.lm_val.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()