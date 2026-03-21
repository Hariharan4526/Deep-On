"""MOD-04: Optical Flow Analyzer.

Farneback dense optical flow for temporal-consistency scoring.
No torch required.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .utils import setup_logger

logger = setup_logger(__name__)


class OpticalFlowAnalyzer:
    """Dense optical flow between aligned face frames."""

    _REAL_SMOOTH_LO = 0.5
    _REAL_SMOOTH_HI = 2.0

    def __init__(self, method: str = "farneback"):
        """
        Args:
            method: 'farneback' (default) or 'raft' (requires torchvision).
        """
        self.method = method
        logger.info(f"OpticalFlowAnalyzer ready (method={method})")

    # ── Core flow computation ─────────────────────────────────────────────────

    def compute_optical_flow(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> np.ndarray:
        """Compute dense optical flow between two face crops.

        Args:
            frame1, frame2 : (H, W, 3) float32 normalised OR uint8 RGB.

        Returns:
            flow : (H, W, 2) float32  — (u, v) displacement per pixel.
        """
        g1 = self._to_gray_u8(frame1)
        g2 = self._to_gray_u8(frame2)

        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        return flow  # (H, W, 2)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def extract_motion_metrics(self, flow: np.ndarray) -> Dict[str, float]:
        """Compute magnitude stats and smoothness (total-variation)."""
        u, v = flow[..., 0], flow[..., 1]
        mag  = np.sqrt(u**2 + v**2)

        tv_u = (np.abs(np.diff(u, axis=0)).sum() +
                np.abs(np.diff(u, axis=1)).sum())
        tv_v = (np.abs(np.diff(v, axis=0)).sum() +
                np.abs(np.diff(v, axis=1)).sum())
        smoothness = float((tv_u + tv_v) / (mag.size + 1e-8))

        return {
            "mean_magnitude": float(mag.mean()),
            "std_magnitude":  float(mag.std()),
            "max_magnitude":  float(mag.max()),
            "smoothness":     smoothness,
            "occlusion_ratio": float(np.mean(mag > 10.0)),
        }

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def detect_temporal_anomalies(
        self, flow_fields: List[np.ndarray]
    ) -> Tuple[float, List[int]]:
        """Flag frame-to-frame magnitude discontinuities.

        Returns:
            anomaly_score : float [0, 1]
            anomaly_frames: list of suspicious frame indices
        """
        if len(flow_fields) < 2:
            return 0.0, []

        mags = [
            float(np.sqrt(f[..., 0]**2 + f[..., 1]**2).mean())
            for f in flow_fields
        ]
        diffs     = np.abs(np.diff(mags))
        threshold = diffs.mean() + 2 * diffs.std()
        anomalies = [i + 1 for i, d in enumerate(diffs) if d > threshold]

        score = float(np.clip(len(anomalies) / max(1, len(flow_fields)), 0, 1))
        return score, anomalies

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score_optical_flow(self, flow_fields: List[np.ndarray]) -> float:
        """Deepfake probability from a sequence of flow fields.

        Real faces  → smoothness in [0.5, 2.0].
        Deepfakes   → smoothness outside this range or high anomaly count.

        Returns float [0, 1].
        """
        if not flow_fields:
            return 0.5

        metrics    = [self.extract_motion_metrics(f) for f in flow_fields]
        mean_smooth = float(np.mean([m["smoothness"] for m in metrics]))
        anom_score, _ = self.detect_temporal_anomalies(flow_fields)

        lo, hi = self._REAL_SMOOTH_LO, self._REAL_SMOOTH_HI
        if lo <= mean_smooth <= hi:
            smooth_score = 0.0
        elif mean_smooth < lo:
            smooth_score = float(np.clip((lo - mean_smooth) / lo, 0, 1))
        else:
            smooth_score = float(np.clip((mean_smooth - hi) / hi, 0, 1))

        p = 0.6 * smooth_score + 0.4 * anom_score
        return float(np.clip(p, 0.0, 1.0))

    # ── Convenience ──────────────────────────────────────────────────────────

    def process_frame_pair(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute flow + metrics in one call."""
        flow    = self.compute_optical_flow(frame1, frame2)
        metrics = self.extract_motion_metrics(flow)
        return flow, metrics

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img