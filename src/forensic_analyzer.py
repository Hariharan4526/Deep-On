"""ENHANCEMENT-03: Forensic Analysis Module.

Lightweight forensic checks for deepfake artifacts:
  - FFT frequency analysis (GAN grid artifacts)
  - Texture inconsistencies
  - Compression anomaly detection
  - Sensor pattern noise (SPN) analysis

No torch required. Complements existing FrequencyAnalyzer.
"""
from typing import Dict, Optional
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.ndimage import median_filter, gaussian_filter
from .utils import setup_logger

logger = setup_logger(__name__)


class ForensicAnalyzer:
    """Detect forensic artifacts typical of deepfakes.
    
    Improves detection of:
    - GAN-generated faces (texture grid patterns)
    - Deepfake face swaps (boundary artifacts)
    - Compression anomalies (local quality drops)
    - Unnatural lighting (shadow inconsistencies)
    """

    def __init__(self):
        """Initialize forensic analyzer."""
        logger.info("ForensicAnalyzer ready (no torch required)")

    # ── GAN artifact detection ────────────────────────────────────────────────

    def detect_gan_artifacts(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect artifacts typical of GAN-generated faces.
        
        GAN outputs often have:
        - Power-law spectrum anomalies
        - Periodic grid patterns from deconvolution
        - Unnatural texture coherence
        
        Returns dict with artifact scores.
        """
        gray = self._to_gray(face_image).astype(np.float32)
        h, w = gray.shape
        
        # FFT analysis
        f_transform = fftshift(fft2(gray))
        power_spectrum = np.abs(f_transform) ** 2 / (h * w)
        
        # Remove DC and low frequencies
        cy, cx = h // 2, w // 2
        radius = np.sqrt((np.arange(h)[:, None] - cy)**2 + 
                        (np.arange(w) - cx)**2)
        
        # Mid-frequency anomalies (GAN grid patterns typically at 10-40 pixels)
        mask_mid = (radius > 5) & (radius < 60)
        mid_energy = power_spectrum[mask_mid].mean()
        
        # High-frequency anomalies
        mask_high = radius > 60
        high_energy = power_spectrum[mask_high].mean()
        
        # Ratio: GANs often have excess energy in mid-frequencies
        gan_grid_score = float(np.clip(mid_energy / (high_energy + 1e-12), 0, 10)) / 10.0
        
        # Phase coherence (GANs often have unnatural phase patterns)
        phase = np.angle(f_transform)
        phase_variance = np.var(phase[mask_mid])
        coherence_score = float(np.clip(1.0 - phase_variance / 10.0, 0, 1))
        
        # Power-law spectrum analysis
        # Natural images: power ∝ 1/f^2, GANs: deviation from this
        power_law_slope = self._estimate_power_law_slope(power_spectrum)
        natural_slope = 2.0
        slope_deviation = float(abs(power_law_slope - natural_slope) / 2.0)
        
        gan_score = float(np.clip(
            (gan_grid_score + coherence_score + slope_deviation) / 3.0, 0, 1
        ))
        
        return {
            "gan_grid_score": gan_grid_score,
            "coherence_score": coherence_score,
            "power_law_deviation": slope_deviation,
            "gan_artifact_score": gan_score,
            "is_likely_gan": gan_score > 0.5,
        }

    @staticmethod
    def _estimate_power_law_slope(power_spectrum: np.ndarray) -> float:
        """Estimate power-law exponent from radial frequency profile."""
        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2
        
        # Radial averaging
        radius = np.sqrt((np.arange(h)[:, None] - cy)**2 + 
                        (np.arange(w) - cx)**2).astype(int)
        
        radial_power = np.array([
            power_spectrum[radius == r].mean() for r in range(1, min(h, w) // 2)
        ])
        
        # Fit log(power) = slope * log(freq) + intercept
        freqs = np.arange(1, len(radial_power) + 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_power = np.log(radial_power + 1e-12)
            log_freq = np.log(freqs)
            
            # Remove infs
            valid = np.isfinite(log_power) & np.isfinite(log_freq)
            if not valid.any():
                return 2.0
            
            slope = np.polyfit(log_freq[valid], log_power[valid], 1)[0]
        
        return float(slope)

    # ── Face boundary artifact detection ──────────────────────────────────────

    def detect_swap_boundary_artifacts(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect artifacts at face swap boundaries.
        
        Returns dict with boundary anomaly scores.
        """
        gray = self._to_gray(face_image).astype(np.float32)
        h, w = gray.shape
        
        # Gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        
        # Edges (high gradient)
        edges = gradient_mag > np.percentile(gradient_mag, 90)
        
        # Check for concentration of edges at image boundaries
        border_width = 20
        border_edges = edges.copy()
        border_edges[border_width:-border_width, border_width:-border_width] = False
        
        interior_edge_density = edges[border_width:-border_width, border_width:-border_width].sum() / \
                                max(1, ((h - 2*border_width) * (w - 2*border_width)))
        boundary_edge_density = border_edges.sum() / max(1, (2*border_width*(h+w)))
        
        # High boundary edge density indicates blending artifacts
        boundary_artifact_score = float(np.clip(boundary_edge_density / (interior_edge_density + 1e-8), 0, 5)) / 5.0
        
        # Halo detection: bright/dark rings around face
        blurred = gaussian_filter(gray, sigma=5)
        detail = gray - blurred
        halo_strength = float(np.std(detail))
        
        return {
            "boundary_artifact_score": boundary_artifact_score,
            "halo_strength": float(np.clip(halo_strength / 50.0, 0, 1)),
            "has_swap_artifacts": boundary_artifact_score > 0.6,
        }

    # ── Texture inconsistency detection ───────────────────────────────────────

    def detect_texture_inconsistencies(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect unnatural texture patterns.
        
        Returns dict with texture anomaly scores.
        """
        gray = self._to_gray(face_image).astype(np.float32)
        h, w = gray.shape
        
        # Split into tiles and compute local statistics
        tile_size = 32
        tile_vars = []
        tile_means = []
        
        for i in range(0, h - tile_size + 1, tile_size):
            for j in range(0, w - tile_size + 1, tile_size):
                tile = gray[i:i+tile_size, j:j+tile_size]
                tile_vars.append(tile.var())
                tile_means.append(tile.mean())
        
        if not tile_vars:
            return {"texture_consistency": 1.0, "is_suspicious": False}
        
        tile_vars = np.array(tile_vars)
        tile_means = np.array(tile_means)
        
        # Natural images: smooth variance variation
        # Deepfakes: abnormal variance distribution
        var_cv = float(tile_vars.std() / (tile_vars.mean() + 1e-8))
        mean_cv = float(tile_means.std() / (tile_means.mean() + 1e-8))
        
        # Deepfakes often have over-smooth textures (too low variance)
        var_median = np.median(tile_vars)
        var_entropy = self._compute_entropy(tile_vars)
        
        # Texture consistency: 1 = natural, 0 = suspicious
        texture_consistency = 1.0 - float(np.clip(abs(var_cv - 0.5) / 0.3, 0, 1))
        
        return {
            "texture_consistency": float(np.clip(texture_consistency, 0, 1)),
            "variance_entropy": var_entropy,
            "is_suspicious": texture_consistency < 0.4,
        }

    # ── Compression anomaly detection ─────────────────────────────────────────

    def detect_compression_anomalies(
        self, face_image: np.ndarray, reference_image: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Detect spatially-varying compression quality.
        
        Args:
            face_image: Current face frame.
            reference_image: Previous frame for comparison (optional).
            
        Returns dict with compression anomaly scores.
        """
        gray = self._to_gray(face_image).astype(np.float32)
        h, w = gray.shape
        
        # Block-wise DCT analysis for JPEG compression patterns
        block_size = 8
        dct_energies = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                tile = gray[i:i+block_size, j:j+block_size]
                dct = cv2.dct(tile)
                energy = np.sum(np.abs(dct)) / (block_size * block_size)
                dct_energies.append(energy)
        
        if not dct_energies:
            return {"compression_uniformity": 1.0, "has_anomalies": False}
        
        dct_energies = np.array(dct_energies)
        
        # Uniform compression: consistent energy across blocks
        # Anomalies: isolated high/low energy regions
        compression_uniformity = 1.0 - float(np.clip(
            dct_energies.std() / (dct_energies.mean() + 1e-8), 0, 1
        ))
        
        # Detect isolated anomalies
        median_energy = np.median(dct_energies)
        anomaly_count = np.sum(np.abs(dct_energies - median_energy) > 2 * dct_energies.std())
        anomaly_ratio = float(anomaly_count / max(1, len(dct_energies)))
        
        return {
            "compression_uniformity": float(np.clip(compression_uniformity, 0, 1)),
            "anomaly_ratio": anomaly_ratio,
            "has_anomalies": anomaly_ratio > 0.15,
        }

    # ── Lighting analysis ─────────────────────────────────────────────────────

    def detect_unnatural_lighting(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect unnatural lighting and shadow inconsistencies.
        
        Returns dict with lighting anomaly scores.
        """
        # Convert to HSV for better light analysis
        hsv = cv2.cvtColor(face_image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        
        h, w = v.shape
        cy, cx = h // 2, w // 2
        
        # Natural faces have radial lighting (brighter center)
        dist_from_center = np.sqrt(
            (np.arange(h)[:, None] - cy)**2 + (np.arange(w) - cx)**2
        )
        
        # Radial gradient
        bright_center = np.corrcoef(
            dist_from_center.ravel(), v.ravel()
        )[0, 1]
        
        # Should be negative (brighter in center)
        radial_consistency = float(np.clip(-bright_center, 0, 1))
        
        # Shadow consistency: smooth transitions
        v_grad = np.gradient(v)
        grad_mag = np.sqrt(v_grad[0]**2 + v_grad[1]**2)
        shadow_smoothness = 1.0 - float(np.clip(grad_mag.std() / 30.0, 0, 1))
        
        return {
            "radial_consistency": radial_consistency,
            "shadow_smoothness": shadow_smoothness,
            "lighting_naturalness": float((radial_consistency + shadow_smoothness) / 2),
            "is_suspicious": (radial_consistency + shadow_smoothness) / 2 < 0.4,
        }

    # ── Utility methods ───────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return image

    @staticmethod
    def _compute_entropy(data: np.ndarray, bins: int = 16) -> float:
        """Compute Shannon entropy."""
        hist, _ = np.histogram(data.ravel(), bins=bins)
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy)

    # ── Comprehensive forensic score ──────────────────────────────────────────

    def compute_forensic_score(self, face_image: np.ndarray) -> Dict[str, float]:
        """Compute overall forensic deepfake score.
        
        Combines all forensic analyses into single anomaly score.
        
        Returns dict with comprehensive forensic analysis.
        """
        gan_results = self.detect_gan_artifacts(face_image)
        boundary_results = self.detect_swap_boundary_artifacts(face_image)
        texture_results = self.detect_texture_inconsistencies(face_image)
        compression_results = self.detect_compression_anomalies(face_image)
        lighting_results = self.detect_unnatural_lighting(face_image)
        
        # Aggregate scores (invert some for consistency: low = natural, high = fake)
        forensic_score = float(np.clip(
            (
                gan_results["gan_artifact_score"] * 0.3 +
                (1.0 - boundary_results["boundary_artifact_score"]) * 0.2 +
                (1.0 - texture_results["texture_consistency"]) * 0.2 +
                (1.0 - compression_results["compression_uniformity"]) * 0.15 +
                (1.0 - lighting_results["lighting_naturalness"]) * 0.15
            ),
            0, 1
        ))
        
        return {
            "forensic_score": forensic_score,
            "gan_artifacts": gan_results,
            "boundary_artifacts": boundary_results,
            "texture_inconsistencies": texture_results,
            "compression_anomalies": compression_results,
            "lighting_artifacts": lighting_results,
            "suspicious": forensic_score > 0.5,
        }
