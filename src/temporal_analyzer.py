"""ENHANCEMENT-02: Temporal Analysis Module with LSTM.

Detects temporal inconsistencies:
  - Unnatural facial motion (via LSTM)
  - Flickering and inconsistent blinking
  - Lip-sync problems
  - Motion discontinuities

Falls back to statistical analysis if torch unavailable.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from .utils import setup_logger

logger = setup_logger(__name__)

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ════════════════════════════════════════════════════════════════════════════════
#  LSTM Motion Analyzer (torch version)
# ════════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:
    
    class MotionLSTM(nn.Module):
        """Lightweight LSTM for temporal motion analysis."""
        
        def __init__(
            self,
            input_size: int = 10,
            hidden_size: int = 32,
            num_layers: int = 2,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Process sequence and return predictions + embeddings."""
            lstm_out, (h_n, _) = self.lstm(x)
            predictions = torch.sigmoid(self.fc(lstm_out))
            return predictions, h_n[-1]


# ════════════════════════════════════════════════════════════════════════════════
#  Temporal Analyzer
# ════════════════════════════════════════════════════════════════════════════════

class TemporalAnalyzer:
    """Analyze temporal consistency and motion naturalness.
    
    Improves detection of:
    - High-quality deepfakes (inconsistent motion)
    - Talking-face AI (unnatural lip sync)
    - Fast motion videos (motion jitter)
    """

    def __init__(self, use_lstm: bool = True):
        """
        Args:
            use_lstm: If True and torch available, use LSTM. Otherwise fallback.
        """
        self.use_lstm = use_lstm and _TORCH_OK
        self._motion_history: List[Dict[str, float]] = []
        self._flow_history: List[np.ndarray] = []
        self._blink_history: List[float] = []
        
        if self.use_lstm:
            try:
                self.lstm_model = MotionLSTM(input_size=10, hidden_size=32, num_layers=2)
                self.lstm_model.eval()
                logger.info("TemporalAnalyzer: LSTM enabled")
            except Exception as e:
                logger.warning(f"LSTM init failed: {e}, using statistical fallback")
                self.use_lstm = False
        else:
            logger.info("TemporalAnalyzer: Statistical mode (no torch/LSTM)")

    # ── Motion metrics ────────────────────────────────────────────────────────

    def compute_motion_metrics(
        self, flow: np.ndarray
    ) -> Dict[str, float]:
        """Extract motion features from optical flow field.
        
        Returns:
            Dict with motion statistics (mean, std, max, entropy).
        """
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u**2 + v**2)
        
        # Statistics
        mean_mag = float(mag.mean())
        std_mag = float(mag.std())
        max_mag = float(mag.max())
        
        # Directional consistency
        angle = np.arctan2(v, u)
        angle_entropy = float(self._compute_entropy(angle))
        
        # Temporal consistency (smoothness)
        smoothness = float(self._compute_smoothness(flow))
        
        return {
            "mean_magnitude": mean_mag,
            "std_magnitude": std_mag,
            "max_magnitude": max_mag,
            "angle_entropy": angle_entropy,
            "smoothness": smoothness,
        }

    @staticmethod
    def _compute_entropy(data: np.ndarray, bins: int = 16) -> float:
        """Compute Shannon entropy of data distribution."""
        hist, _ = np.histogram(data.ravel(), bins=bins)
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy)

    @staticmethod
    def _compute_smoothness(flow: np.ndarray) -> float:
        """Compute flow field smoothness (total variation)."""
        u, v = flow[..., 0], flow[..., 1]
        du = np.abs(np.diff(u, axis=0)).mean() + np.abs(np.diff(u, axis=1)).mean()
        dv = np.abs(np.diff(v, axis=0)).mean() + np.abs(np.diff(v, axis=1)).mean()
        return float(du + dv)

    # ── Flickering detection ──────────────────────────────────────────────────

    def detect_flickering(
        self, cnn_scores: List[float], window_size: int = 5
    ) -> Dict[str, float]:
        """Detect rapid CNN score oscillations (flickering).
        
        Args:
            cnn_scores: Per-frame deepfake probability scores.
            window_size: Frame window for flickering analysis.
            
        Returns:
            Dict with flickering metrics.
        """
        if len(cnn_scores) < window_size:
            return {"flicker_score": 0.0, "is_flickering": False}
        
        scores = np.array(cnn_scores)
        
        # Check for high-frequency oscillations
        flicker_count = 0
        for i in range(1, len(scores) - 1):
            # Local max or min followed by opposite
            if (scores[i] > scores[i-1] and scores[i] > scores[i+1]) or \
               (scores[i] < scores[i-1] and scores[i] < scores[i+1]):
                flicker_count += 1
        
        flicker_ratio = flicker_count / max(1, len(scores) - 2)
        is_flickering = flicker_ratio > 0.4  # >40% local extrema = flickering
        
        return {
            "flicker_score": float(np.clip(flicker_ratio, 0, 1)),
            "is_flickering": is_flickering,
            "flicker_frames": flicker_count,
        }

    # ── Blink detection ───────────────────────────────────────────────────────

    def detect_blinking_patterns(
        self, eye_landmarks: List[Tuple[float, float, float, float]]
    ) -> Dict[str, float]:
        """Detect unnatural blinking patterns.
        
        Args:
            eye_landmarks: List of (left_eye_height, right_eye_height, ...).
            
        Returns:
            Dict with blink statistics.
        """
        if len(eye_landmarks) < 5:
            return {"blink_frequency": 0.0, "blink_regularity": 1.0, "is_unnatural": False}
        
        eyes = np.array(eye_landmarks)
        diffs = np.abs(np.diff(eyes, axis=0))
        
        # Blink: sudden eye height drop
        blinks = np.sum(diffs > np.std(eyes) * 1.5)
        blink_freq = float(blinks / max(1, len(eyes)))
        
        # Normal: 15-30 blinks/min ≈ 0.3-0.5 blinks/sec
        natural_blink_range = (0.25, 0.8)
        is_unnatural = not (natural_blink_range[0] <= blink_freq <= natural_blink_range[1])
        
        # Regularity: variance of blink intervals
        blink_indices = np.where(diffs > np.std(eyes) * 1.5)[0]
        if len(blink_indices) > 1:
            intervals = np.diff(blink_indices)
            regularity = float(1.0 / (1.0 + np.std(intervals)))
        else:
            regularity = 1.0
        
        return {
            "blink_frequency": float(np.clip(blink_freq, 0, 1)),
            "blink_regularity": float(np.clip(regularity, 0, 1)),
            "is_unnatural": is_unnatural,
            "blink_count": int(blinks),
        }

    # ── Lip-sync analysis ─────────────────────────────────────────────────────

    def analyze_lip_sync(
        self, jaw_motion: List[float], audio_energy: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Detect lip-sync inconsistencies (requires audio or jaw landmarks).
        
        Args:
            jaw_motion: Per-frame jaw height changes.
            audio_energy: Per-frame audio energy (optional).
            
        Returns:
            Dict with lip-sync metrics.
        """
        if len(jaw_motion) < 5:
            return {"lip_sync_score": 0.5, "is_suspicious": False}
        
        jaw = np.array(jaw_motion)
        
        # If audio available, cross-correlate
        if audio_energy is not None:
            audio = np.array(audio_energy)
            if len(audio) == len(jaw):
                # Normalize
                jaw_norm = (jaw - jaw.mean()) / (jaw.std() + 1e-8)
                audio_norm = (audio - audio.mean()) / (audio.std() + 1e-8)
                correlation = float(np.corrcoef(jaw_norm, audio_norm)[0, 1])
                lip_sync_score = float(np.clip((correlation + 1) / 2, 0, 1))
            else:
                lip_sync_score = 0.5
        else:
            # Check jaw motion consistency
            diffs = np.abs(np.diff(jaw))
            jaw_entropy = self._compute_entropy(diffs)
            # Natural: moderate entropy; Fake: too smooth or too noisy
            lip_sync_score = 1.0 - float(np.clip(abs(jaw_entropy - 1.5) / 2.0, 0, 1))
        
        is_suspicious = lip_sync_score < 0.4
        
        return {
            "lip_sync_score": float(np.clip(lip_sync_score, 0, 1)),
            "is_suspicious": is_suspicious,
        }

    # ── LSTM-based temporal analysis ──────────────────────────────────────────

    def analyze_motion_sequence_lstm(
        self, motion_features: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Analyze motion sequence with LSTM (if available).
        
        Args:
            motion_features: List of motion metric dicts.
            
        Returns:
            Dict with anomaly scores.
        """
        if not self.use_lstm or len(motion_features) < 5:
            return self._analyze_motion_sequence_statistical(motion_features)
        
        # Convert to tensor
        try:
            features_array = np.array([
                [
                    f.get("mean_magnitude", 0),
                    f.get("std_magnitude", 0),
                    f.get("max_magnitude", 0),
                    f.get("angle_entropy", 0),
                    f.get("smoothness", 0),
                    f.get("mean_magnitude", 0) ** 2,
                    f.get("std_magnitude", 0) ** 2,
                    f.get("angle_entropy", 0) ** 2,
                    1.0,  # Bias
                    0.0,  # Reserve
                ]
                for f in motion_features[-32:]  # Last 32 frames
            ])
            
            features_tensor = torch.from_numpy(features_array).float().unsqueeze(0)
            
            with torch.no_grad():
                predictions, embeddings = self.lstm_model(features_tensor)
            
            anomaly_scores = predictions.squeeze().cpu().numpy()
            final_anomaly = float(anomaly_scores.mean())
            
            return {
                "temporal_anomaly_score": float(np.clip(final_anomaly, 0, 1)),
                "sequence_length": len(motion_features),
                "lstm_available": True,
            }
        except Exception as e:
            logger.warning(f"LSTM analysis failed: {e}, falling back")
            return self._analyze_motion_sequence_statistical(motion_features)

    def _analyze_motion_sequence_statistical(
        self, motion_features: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Statistical fallback for temporal anomaly detection."""
        if not motion_features:
            return {"temporal_anomaly_score": 0.5, "sequence_length": 0}
        
        magnitudes = np.array([f.get("mean_magnitude", 0) for f in motion_features])
        smoothness = np.array([f.get("smoothness", 0) for f in motion_features])
        
        # Anomalies: sudden spikes or drops
        mag_diffs = np.abs(np.diff(magnitudes))
        mag_outliers = np.sum(mag_diffs > (magnitudes.std() * 2.5)) / max(1, len(magnitudes) - 1)
        
        # Inconsistent smoothness
        smooth_var = smoothness.std() / (smoothness.mean() + 1e-8)
        anomaly = float(np.clip((mag_outliers + np.clip(smooth_var / 5, 0, 1)) / 2, 0, 1))
        
        return {
            "temporal_anomaly_score": anomaly,
            "sequence_length": len(motion_features),
            "lstm_available": False,
        }

    # ── Update history ────────────────────────────────────────────────────────

    def update_motion_history(self, metrics: Dict[str, float]) -> None:
        """Add motion metrics to history."""
        self._motion_history.append(metrics)
        if len(self._motion_history) > 256:  # Limit memory
            self._motion_history.pop(0)

    def update_flow_history(self, flow: np.ndarray) -> None:
        """Add optical flow to history."""
        self._flow_history.append(flow.copy())
        if len(self._flow_history) > 64:
            self._flow_history.pop(0)

    def reset(self) -> None:
        """Reset all history (call at video start)."""
        self._motion_history.clear()
        self._flow_history.clear()
        self._blink_history.clear()

    def get_temporal_score(self) -> float:
        """Compute overall temporal consistency score [0, 1]."""
        if not self._motion_history:
            return 0.5
        
        # Analyze stored history
        result = self.analyze_motion_sequence_lstm(self._motion_history)
        return result.get("temporal_anomaly_score", 0.5)
