"""ENHANCEMENT-01: Adaptive Frame Processing Module.

Replaces fixed frame-skip with intelligent frame selection:
  - Blur detection (skip blurry frames)
  - Duplicate detection (skip redundant frames)
  - Motion sensitivity (extract high-motion frames)
  - Compression artifact detection

Maintains backward compatibility with existing VideoHandler.
"""
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from .utils import setup_logger

logger = setup_logger(__name__)


class AdaptiveFrameProcessor:
    """Intelligent frame sampling for deepfake detection.
    
    Improves robustness to:
    - High-quality deepfakes (motion analysis)
    - Compressed videos (artifact detection)
    - Low-light videos (brightness normalization)
    - Side profiles (frame quality filtering)
    - Fast motion (motion-sensitive extraction)
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        duplicate_threshold: float = 0.95,
        motion_threshold: float = 0.05,
        min_frame_interval: int = 2,
        compression_threshold: float = 0.3,
    ):
        """
        Args:
            blur_threshold: Laplacian variance threshold for blurry frames.
            duplicate_threshold: Cosine similarity for duplicate detection.
            motion_threshold: Minimum optical flow magnitude to be "motion".
            min_frame_interval: Minimum frames between extracted samples.
            compression_threshold: Artifact energy ratio threshold.
        """
        self.blur_threshold = blur_threshold
        self.duplicate_threshold = duplicate_threshold
        self.motion_threshold = motion_threshold
        self.min_frame_interval = min_frame_interval
        self.compression_threshold = compression_threshold
        self._prev_frame_hash: Optional[np.ndarray] = None
        self._frame_count = 0
        logger.info(
            f"AdaptiveFrameProcessor init: blur={blur_threshold}, "
            f"duplicate={duplicate_threshold}, motion={motion_threshold}"
        )

    # ── Blur detection ────────────────────────────────────────────────────────

    def is_blurry(self, frame: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Detect blurry frames using Laplacian variance.
        
        Args:
            frame: RGB image (any size, uint8).
            threshold: Override self.blur_threshold.
            
        Returns:
            True if frame is blurry (should be skipped).
        """
        th = threshold if threshold is not None else self.blur_threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        is_blur = variance < th
        return is_blur

    # ── Duplicate detection ───────────────────────────────────────────────────

    def is_duplicate(
        self, frame: np.ndarray, threshold: Optional[float] = None
    ) -> bool:
        """Detect near-duplicate frames via histogram comparison.
        
        Args:
            frame: RGB image.
            threshold: Override self.duplicate_threshold.
            
        Returns:
            True if frame is duplicate of previous.
        """
        if self._prev_frame_hash is None:
            self._prev_frame_hash = self._compute_hash(frame)
            return False

        th = threshold if threshold is not None else self.duplicate_threshold
        curr_hash = self._compute_hash(frame)
        similarity = float(np.dot(curr_hash, self._prev_frame_hash))
        is_dup = similarity > th
        
        if not is_dup:
            self._prev_frame_hash = curr_hash
        
        return is_dup

    @staticmethod
    def _compute_hash(frame: np.ndarray, bins: int = 32) -> np.ndarray:
        """Compute normalized histogram hash for frame."""
        hist_b = cv2.calcHist([frame], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [bins], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [bins], [0, 256])
        hist = np.concatenate([hist_b.ravel(), hist_g.ravel(), hist_r.ravel()])
        hist = hist / (hist.sum() + 1e-8)
        return hist.astype(np.float32)

    # ── Motion detection ──────────────────────────────────────────────────────

    def compute_motion_magnitude(
        self, frame1: np.ndarray, frame2: np.ndarray
    ) -> float:
        """Compute optical flow magnitude between frames.
        
        Returns:
            Mean optical flow magnitude [0, inf).
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.0, flags=0
        )
        
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(mag.mean())

    # ── Compression artifact detection ─────────────────────────────────────

    def detect_compression_artifacts(
        self, frame: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Detect compression artifacts (JPEG blocking, loss patterns).
        
        Returns dict with:
            - blocking_score: [0, 1] — higher = more JPEG blocking
            - artifact_ratio: [0, 1] — fraction of artifact pixels
            - is_compressed: bool — threshold exceeded
        """
        th = threshold if threshold is not None else self.compression_threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # JPEG blocking detection via DCT block boundary analysis
        h, w = gray.shape
        block_size = 8
        block_edges = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                tile = gray[i:i+block_size, j:j+block_size]
                # Gradient magnitude at block boundaries
                if i + block_size < h:
                    edge_h = np.abs(gray[i+block_size, j:j+block_size] - tile[-1, :]).mean()
                    block_edges.append(edge_h)
                if j + block_size < w:
                    edge_v = np.abs(gray[i:i+block_size, j+block_size] - tile[:, -1]).mean()
                    block_edges.append(edge_v)
        
        blocking_score = float(np.mean(block_edges) / 255.0) if block_edges else 0.0
        artifact_ratio = float(np.clip(blocking_score, 0, 1))
        is_compressed = artifact_ratio > th
        
        return {
            "blocking_score": blocking_score,
            "artifact_ratio": artifact_ratio,
            "is_compressed": is_compressed,
        }

    # ── Low-light adjustment ──────────────────────────────────────────────────

    def estimate_brightness(self, frame: np.ndarray) -> float:
        """Estimate frame brightness [0, 1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return float(gray.mean() / 255.0)

    def is_low_light(self, frame: np.ndarray, threshold: float = 0.3) -> bool:
        """Check if frame is low-light (brightness < threshold)."""
        return self.estimate_brightness(frame) < threshold

    def normalize_brightness(
        self, frame: np.ndarray, target_brightness: float = 0.5
    ) -> np.ndarray:
        """Adaptive brightness normalization for low-light frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        current = gray.mean() / 255.0
        
        if current < 0.01:
            return frame  # Too dark, don't adjust
        
        scale = target_brightness / (current + 1e-8)
        scale = np.clip(scale, 0.5, 2.0)  # Limit adjustment
        
        adjusted = (frame.astype(np.float32) * scale).astype(np.uint8)
        adjusted = cv2.cvtColor(adjusted, cv2.COLOR_RGB2RGB)  # Ensure RGB
        return adjusted

    # ── Frame quality scoring ─────────────────────────────────────────────────

    def score_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Compute overall frame quality score."""
        blur_score = self.is_blurry(frame)
        brightness = self.estimate_brightness(frame)
        artifacts = self.detect_compression_artifacts(frame)
        
        # Quality = (1 - blur) * brightness_ok * (1 - artifacts)
        blur_penalty = 1.0 if blur_score else 0.0
        brightness_ok = 1.0 if 0.2 < brightness < 0.95 else 0.5
        artifact_penalty = artifacts["artifact_ratio"]
        
        quality = (1.0 - blur_penalty) * brightness_ok * (1.0 - artifact_penalty)
        
        return {
            "quality_score": float(np.clip(quality, 0, 1)),
            "blur": blur_score,
            "brightness": brightness,
            "artifact_ratio": artifacts["artifact_ratio"],
        }

    # ── Main API ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset internal state (call at video start)."""
        self._prev_frame_hash = None
        self._frame_count = 0

    def should_process_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        frames_since_last: int = 0,
    ) -> Tuple[bool, Dict[str, float]]:
        """Decide whether to process frame based on quality heuristics.
        
        Args:
            frame: Current frame (RGB uint8).
            prev_frame: Previous frame for motion calculation.
            frames_since_last: Frames since last processed frame.
            
        Returns:
            (should_process, metadata_dict)
        """
        self._frame_count += 1
        
        # Skip if minimum interval not met (unless high motion)
        if frames_since_last < self.min_frame_interval and prev_frame is not None:
            motion = self.compute_motion_magnitude(prev_frame, frame)
            if motion < self.motion_threshold:
                return False, {"skip_reason": "low_motion", "motion": motion}
        
        # Skip if blurry
        if self.is_blurry(frame):
            return False, {"skip_reason": "blurry"}
        
        # Skip if duplicate
        if self.is_duplicate(frame):
            return False, {"skip_reason": "duplicate"}
        
        # Accept frame with quality metadata
        quality = self.score_frame_quality(frame)
        return True, {
            "reason": "accepted",
            "frame_idx": self._frame_count,
            **quality,
        }
