"""MOD-05: Frequency Domain Analyzer.

FFT power-spectral analysis and block-wise DCT for deepfake artefact detection.
No torch, no GPU required.
"""
from typing import Dict

import cv2
import numpy as np
from scipy.fft import dct as scipy_dct

from .utils import setup_logger

logger = setup_logger(__name__)


class FrequencyAnalyzer:
    """Analyse frequency-domain artefacts in aligned face crops."""

    DCT_BLOCK = 8  # JPEG-standard 8×8 blocks

    def __init__(self):
        logger.info("FrequencyAnalyzer ready (NumPy FFT + SciPy DCT, CPU-only).")

    # ── FFT ──────────────────────────────────────────────────────────────────

    def compute_fft(self, face_image: np.ndarray) -> Dict[str, float]:
        """2-D FFT power-spectral features.

        Args:
            face_image : (H, W, 3) float32 normalised or uint8 RGB.

        Returns dict:
            low_energy_ratio   – fraction of energy in low spatial freqs (<0.3)
            high_energy_ratio  – fraction of energy in high spatial freqs (>0.7)
            isotropy           – angular energy uniformity [0, 1]
            peak_ratio         – max-PSD / median-PSD (high → GAN grid artefact)
            fft_anomaly_score  – derived deepfake score [0, 1]
        """
        gray = self._to_gray(face_image).astype(np.float32)
        h, w = gray.shape

        fft_shift = np.fft.fftshift(np.fft.fft2(gray))
        psd       = np.abs(fft_shift) ** 2 / (h * w)

        cy, cx = h // 2, w // 2
        Y, X   = np.ogrid[:h, :w]
        R      = np.sqrt((X - cx)**2 + (Y - cy)**2) / max(cx, cy)

        total     = psd.sum() + 1e-12
        low_ratio = float(psd[R < 0.3].sum() / total)
        hi_ratio  = float(psd[R > 0.7].sum() / total)

        # Isotropy: 1 = perfectly isotropic, 0 = strongly directional
        angles = np.degrees(np.arctan2(Y - cy, X - cx))
        sector_e = [psd[(angles >= a) & (angles < a + 45)].sum()
                    for a in range(0, 360, 45)]
        isotropy = float(np.clip(
            1.0 - np.std(sector_e) / (np.mean(sector_e) + 1e-12), 0, 1
        ))

        peak_ratio   = float(psd.max() / (np.median(psd) + 1e-12))
        # GAN grid artefacts push peak_ratio >> 5
        fft_anomaly  = float(np.clip((peak_ratio - 5.0) / 50.0, 0, 1))

        return {
            "low_energy_ratio":  low_ratio,
            "high_energy_ratio": hi_ratio,
            "isotropy":          isotropy,
            "peak_ratio":        peak_ratio,
            "fft_anomaly_score": fft_anomaly,
        }

    # ── DCT ──────────────────────────────────────────────────────────────────

    def compute_dct(self, face_image: np.ndarray) -> Dict[str, float]:
        """Block-wise DCT analysis.

        Returns dict:
            high_freq_variance      – variance of high-freq DCT coefficients
            high_freq_energy_ratio  – fraction of energy in high-freq bands
            dct_anomaly_score       – derived deepfake score [0, 1]
        """
        gray  = self._to_gray(face_image).astype(np.float32)
        h, w  = gray.shape
        B     = self.DCT_BLOCK

        hf_coeffs    = []
        hf_energy    = 0.0
        total_energy = 0.0

        for i in range(0, h - B + 1, B):
            for j in range(0, w - B + 1, B):
                tile = gray[i:i + B, j:j + B]
                # 2-D DCT-II (scipy: row-wise then transpose)
                d = scipy_dct(scipy_dct(tile, norm="ortho").T, norm="ortho").T

                # High-freq mask: coefficient (u, v) with u+v > 4
                for u in range(B):
                    for v in range(B):
                        val = d[u, v] ** 2
                        total_energy += val
                        if u + v > 4:
                            hf_coeffs.append(d[u, v])
                            hf_energy += val

        hf_arr = np.array(hf_coeffs)
        hf_var = float(np.var(hf_arr)) if len(hf_arr) > 0 else 0.0
        hf_ratio = hf_energy / (total_energy + 1e-12)

        # Deepfakes tend to have low high-freq variance (over-smoothed textures)
        dct_anomaly = float(np.clip(1.0 - hf_var / (hf_var + 5.0), 0, 1))

        return {
            "high_freq_variance":     hf_var,
            "high_freq_energy_ratio": float(hf_ratio),
            "dct_anomaly_score":      dct_anomaly,
        }

    # ── Combined score ────────────────────────────────────────────────────────

    def score_frequency_domain(self, face_image: np.ndarray) -> float:
        """p_frequency = 0.6 × fft_anomaly + 0.4 × dct_anomaly → [0, 1]."""
        fft_feats = self.compute_fft(face_image)
        dct_feats = self.compute_dct(face_image)
        p = 0.6 * fft_feats["fft_anomaly_score"] + 0.4 * dct_feats["dct_anomaly_score"]
        return float(np.clip(p, 0.0, 1.0))

    def get_all_features(self, face_image: np.ndarray) -> Dict[str, float]:
        """All FFT + DCT features merged into one dict."""
        feats = {}
        feats.update(self.compute_fft(face_image))
        feats.update(self.compute_dct(face_image))
        feats["p_frequency"] = self.score_frequency_domain(face_image)
        return feats

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img