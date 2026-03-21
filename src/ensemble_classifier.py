"""MOD-07: Ensemble Classifier.

Weighted voting + EMA temporal smoothing + temperature calibration.
No torch required.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import setup_logger, sigmoid, logit

logger = setup_logger(__name__)


class EnsembleClassifier:
    """Aggregate pathway probabilities into a calibrated deepfake verdict."""

    def __init__(
        self,
        cnn_weight:          float = 0.45,
        optflow_weight:      float = 0.30,
        freq_weight:         float = 0.15,
        lm_weight:           float = 0.10,
        temperature:         float = 1.2,
        temporal_alpha:      float = 0.7,
        classification_threshold: float = 0.5,
    ):
        raw = np.array([cnn_weight, optflow_weight, freq_weight, lm_weight],
                       dtype=np.float64)
        self.weights   = raw / raw.sum()          # always sums to 1
        self.T         = temperature
        self.alpha     = temporal_alpha
        self.threshold = classification_threshold
        self._ema: Optional[float] = None

        logger.info(
            f"EnsembleClassifier  weights={self.weights.round(3)}  "
            f"T={temperature}  α={temporal_alpha}"
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def aggregate_predictions(
        self,
        p_cnn: float, p_optflow: float,
        p_freq: float, p_lm: float,
    ) -> float:
        """Weighted average → ensemble probability [0, 1]."""
        probs = np.clip([p_cnn, p_optflow, p_freq, p_lm], 0, 1)
        return float(np.clip(np.dot(self.weights, probs), 0, 1))

    # ── Temporal smoothing ────────────────────────────────────────────────────

    def temporal_smooth(
        self, predictions: List[float], alpha: Optional[float] = None
    ) -> List[float]:
        """Offline EMA over a list: s_t = α·s_{t-1} + (1-α)·p_t."""
        a = alpha if alpha is not None else self.alpha
        if not predictions:
            return []
        s, out = predictions[0], []
        for p in predictions:
            s = a * s + (1.0 - a) * p
            out.append(float(s))
        return out

    def update_ema(self, p_t: float) -> float:
        """Online EMA — call once per frame."""
        if self._ema is None:
            self._ema = p_t
        else:
            self._ema = self.alpha * self._ema + (1.0 - self.alpha) * p_t
        return self._ema

    def reset_ema(self) -> None:
        """Call at the start of each new video."""
        self._ema = None

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate_confidence(self, raw_prob: float) -> float:
        """Temperature scaling: p_cal = σ(logit(p) / T)."""
        return float(np.clip(sigmoid(logit(raw_prob) / self.T), 0, 1))

    # ── Classification ────────────────────────────────────────────────────────

    def classify(
        self, ensemble_prob: float, calibrate: bool = True
    ) -> Tuple[str, float]:
        """Return (label, confidence_percent).

        confidence_percent: distance from 0.5 boundary, scaled to [0, 100].
        """
        p     = self.calibrate_confidence(ensemble_prob) if calibrate else float(np.clip(ensemble_prob, 0, 1))
        label = "Deepfake" if p > self.threshold else "Authentic"
        conf  = float(abs(p - 0.5) * 200.0)
        return label, conf

    # ── Full per-frame step ───────────────────────────────────────────────────

    def process_frame(
        self,
        p_cnn: float, p_optflow: float,
        p_freq: float, p_lm: float,
        use_temporal: bool = True,
    ) -> Dict[str, object]:
        """One complete ensemble step for a single frame."""
        raw      = self.aggregate_predictions(p_cnn, p_optflow, p_freq, p_lm)
        smoothed = self.update_ema(raw) if use_temporal else raw
        cal      = self.calibrate_confidence(smoothed)
        label    = "Deepfake" if cal > self.threshold else "Authentic"
        conf     = float(abs(cal - 0.5) * 200.0)
        return {
            "p_cnn":          p_cnn,
            "p_optflow":      p_optflow,
            "p_freq":         p_freq,
            "p_lm":           p_lm,
            "ensemble_prob":  raw,
            "smoothed_prob":  smoothed,
            "calibrated_prob": cal,
            "label":          label,
            "confidence":     conf,
        }

    # ── Video-level aggregation ───────────────────────────────────────────────

    def aggregate_video_predictions(
        self, frame_results: List[Dict]
    ) -> Dict:
        """Summarise per-frame dicts into a final video verdict."""
        if not frame_results:
            return {"label": "Unknown", "confidence": 0.0, "probability": 0.5}

        probs = [r["calibrated_prob"] for r in frame_results]

        mean_cnn = float(np.mean([r["p_cnn"] for r in frame_results]))
        mean_opt = float(np.mean([r["p_optflow"] for r in frame_results]))
        mean_frq = float(np.mean([r["p_freq"] for r in frame_results]))
        mean_lm  = float(np.mean([r["p_lm"] for r in frame_results]))

        mean_prob = float(np.mean(probs))
        # Video-level calibration rules:
        # 1) Low-motion + moderate/high spectral artefacts often indicate AI generation.
        if 0.05 <= mean_opt < 0.15 and 0.72 < mean_frq < 0.80 and mean_cnn > 0.50:
            mean_prob += 0.03
        # 1b) Very high spectral energy with low-but-not-zero motion is another
        # common AI pattern in short clips.
        if 0.04 <= mean_opt < 0.18 and mean_frq > 0.85:
            mean_prob += 0.03
        # 2) Very high motion with weak structure cues tends to be authentic content.
        if mean_opt > 0.35 and mean_cnn < 0.55 and mean_lm < 0.50:
            mean_prob -= 0.07
        # 3) Ultra-static but highly textured natural captures can mimic AI frequency cues.
        if mean_opt < 0.05 and mean_frq > 0.80:
            mean_prob -= 0.03
        mean_prob = float(np.clip(mean_prob, 0, 1))

        label, conf = self.classify(mean_prob, calibrate=False)

        return {
            "label":            label,
            "confidence":       round(conf, 2),
            "probability":      round(mean_prob, 4),
            "frames_processed": len(frame_results),
            "frames_deepfake":  sum(1 for r in frame_results if r["label"] == "Deepfake"),
            "pathway_means": {
                "cnn":          round(mean_cnn, 4),
                "optical_flow": round(mean_opt, 4),
                "frequency":    round(mean_frq, 4),
                "landmarks":    round(mean_lm, 4),
            },
        }

    # ── Optional: learn weights from validation data ──────────────────────────

    def learn_ensemble_weights(
        self,
        val_preds: List[Tuple[float, float, float, float]],
        val_labels: List[int],
    ) -> np.ndarray:
        """Fit logistic regression to learn optimal ensemble weights."""
        from sklearn.linear_model import LogisticRegression
        X = np.array(val_preds)
        y = np.array(val_labels)
        clf = LogisticRegression(fit_intercept=False, max_iter=500)
        clf.fit(X, y)
        raw = np.clip(clf.coef_[0], 0, None)
        self.weights = raw / (raw.sum() + 1e-12)
        logger.info(f"Learned weights: {self.weights.round(4)}")
        return self.weights