from __future__ import annotations

import json
import math
import mimetypes
import os
import statistics
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}


@dataclass
class Evidence:
    name: str
    score: float
    details: Dict[str, Any]


@dataclass
class DetectionResult:
    input_path: str
    media_type: str
    ai_generated_confidence: float
    authenticity_confidence: float
    predicted_source: str
    evidences: List[Evidence]
    metadata: Dict[str, Any]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidences"] = [asdict(item) for item in self.evidences]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DetectorConfig:
    video_sample_limit: int = 12
    ai_threshold: float = 0.55


class DeepfakeDetector:
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def analyze(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        metadata = self._extract_metadata(path)
        media_type = self._infer_media_type(path)
        notes: List[str] = []

        if media_type == "image":
            score, evidences = self._analyze_image(path)
        elif media_type == "video":
            score, evidences = self._analyze_video(path)
        elif media_type == "audio":
            score, evidences = self._analyze_audio(path)
        else:
            score, evidences = self._analyze_unknown(path)
            notes.append("Unknown media type: falling back to metadata-only analysis.")

        ai_score = self._clamp(score)
        authenticity = self._clamp(1.0 - ai_score)
        predicted_source = self._predict_source(evidences, media_type)

        if ai_score >= self.config.ai_threshold:
            notes.append("Likely AI-generated or manipulated content.")
        else:
            notes.append("Likely authentic or insufficient AI artifacts.")

        notes.append(
            "Model/source prediction is heuristic and not forensic proof."
        )

        return DetectionResult(
            input_path=str(path),
            media_type=media_type,
            ai_generated_confidence=round(ai_score, 4),
            authenticity_confidence=round(authenticity, 4),
            predicted_source=predicted_source,
            evidences=evidences,
            metadata=metadata,
            notes=notes,
        )

    def _extract_metadata(self, path: Path) -> Dict[str, Any]:
        stat = path.stat()
        guessed_mime, _ = mimetypes.guess_type(str(path))
        return {
            "name": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "mime_type": guessed_mime or "unknown",
        }

    def _infer_media_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        guessed_mime, _ = mimetypes.guess_type(str(path))
        if guessed_mime:
            if guessed_mime.startswith("image/"):
                return "image"
            if guessed_mime.startswith("video/"):
                return "video"
            if guessed_mime.startswith("audio/"):
                return "audio"
        return "unknown"

    def _analyze_image(self, path: Path) -> Tuple[float, List[Evidence]]:
        if cv2 is None:
            return 0.45, [
                Evidence(
                    name="opencv_missing",
                    score=0.45,
                    details={"reason": "opencv-python is required for image analysis"},
                )
            ]

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return 0.5, [
                Evidence(
                    name="image_load_error",
                    score=0.5,
                    details={"reason": "could not load image"},
                )
            ]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        freq_features = self._frequency_features(gray)
        high_freq_ratio = freq_features["high_frequency_ratio"]
        radial_spike_score = freq_features["radial_spike_score"]

        score = self._weighted_sum(
            [
                (self._normalize(laplacian_var, 40, 600), 0.25),
                (self._normalize(edge_density, 0.05, 0.28), 0.2),
                (self._normalize(high_freq_ratio, 0.08, 0.30), 0.35),
                (self._normalize(radial_spike_score, 0.01, 0.09), 0.2),
            ]
        )

        evidences = [
            Evidence(
                name="sharpness_pattern",
                score=round(self._normalize(laplacian_var, 40, 600), 4),
                details={"laplacian_variance": round(laplacian_var, 3)},
            ),
            Evidence(
                name="edge_distribution",
                score=round(self._normalize(edge_density, 0.05, 0.28), 4),
                details={"edge_density": round(edge_density, 5)},
            ),
            Evidence(
                name="frequency_signature",
                score=round(self._normalize(high_freq_ratio, 0.08, 0.30), 4),
                details={
                    "high_frequency_ratio": round(high_freq_ratio, 5),
                    "radial_spike_score": round(radial_spike_score, 5),
                },
            ),
        ]

        return score, evidences

    def _analyze_video(self, path: Path) -> Tuple[float, List[Evidence]]:
        if cv2 is None:
            return 0.5, [
                Evidence(
                    name="opencv_missing",
                    score=0.5,
                    details={"reason": "opencv-python is required for video analysis"},
                )
            ]

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return 0.5, [
                Evidence(
                    name="video_load_error",
                    score=0.5,
                    details={"reason": "could not open video"},
                )
            ]

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_count = min(max(frame_count, 1), self.config.video_sample_limit)
        step = max(frame_count // max(sample_count, 1), 1)

        sampled_scores: List[float] = []
        frame_noise: List[float] = []
        temporal_diffs: List[float] = []
        previous_gray = None

        for idx in range(sample_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx * step)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_score, _ = self._analyze_image_frame(gray)
            sampled_scores.append(frame_score)

            noise_estimate = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            frame_noise.append(noise_estimate)

            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                temporal_diffs.append(float(np.mean(diff)))
            previous_gray = gray

        capture.release()

        if not sampled_scores:
            return 0.5, [
                Evidence(
                    name="video_sample_error",
                    score=0.5,
                    details={"reason": "no readable frames"},
                )
            ]

        avg_frame_score = float(np.mean(sampled_scores))
        temporal_var = statistics.pvariance(temporal_diffs) if len(temporal_diffs) > 1 else 0.0
        noise_var = statistics.pvariance(frame_noise) if len(frame_noise) > 1 else 0.0

        temporal_score = self._normalize(temporal_var, 3.0, 140.0)
        noise_score = self._normalize(noise_var, 100.0, 15000.0)

        score = self._weighted_sum(
            [
                (avg_frame_score, 0.65),
                (temporal_score, 0.2),
                (noise_score, 0.15),
            ]
        )

        evidences = [
            Evidence(
                name="frame_level_artifacts",
                score=round(avg_frame_score, 4),
                details={"sampled_frames": len(sampled_scores)},
            ),
            Evidence(
                name="temporal_inconsistency",
                score=round(temporal_score, 4),
                details={"temporal_variance": round(temporal_var, 5)},
            ),
            Evidence(
                name="inter_frame_noise_pattern",
                score=round(noise_score, 4),
                details={"noise_variance": round(noise_var, 5)},
            ),
        ]

        return score, evidences

    def _analyze_audio(self, path: Path) -> Tuple[float, List[Evidence]]:
        suffix = path.suffix.lower()
        if suffix != ".wav":
            return 0.45, [
                Evidence(
                    name="audio_limited_support",
                    score=0.45,
                    details={"reason": "prototype currently performs waveform analysis on .wav"},
                )
            ]

        try:
            with wave.open(str(path), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                raw_audio = wav_file.readframes(n_frames)
        except Exception:
            return 0.5, [
                Evidence(
                    name="audio_read_error",
                    score=0.5,
                    details={"reason": "could not parse wav stream"},
                )
            ]

        if sample_width not in (1, 2, 4):
            return 0.5, [
                Evidence(
                    name="audio_format_unsupported",
                    score=0.5,
                    details={"sample_width": sample_width},
                )
            ]

        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        signal = np.frombuffer(raw_audio, dtype=dtype)
        if n_channels > 1:
            signal = signal.reshape(-1, n_channels).mean(axis=1)

        signal = signal.astype(np.float32)
        if signal.size == 0:
            return 0.5, [
                Evidence(
                    name="audio_empty",
                    score=0.5,
                    details={"reason": "empty audio stream"},
                )
            ]

        normalized = signal / (np.max(np.abs(signal)) + 1e-8)
        zero_crossings = np.where(np.diff(np.signbit(normalized)))[0].size
        zcr = zero_crossings / max(normalized.size, 1)

        fft_mag = np.abs(np.fft.rfft(normalized))
        spectral_flatness = float(np.exp(np.mean(np.log(fft_mag + 1e-8))) / (np.mean(fft_mag) + 1e-8))

        score = self._weighted_sum(
            [
                (self._normalize(zcr, 0.02, 0.25), 0.5),
                (self._normalize(spectral_flatness, 0.1, 0.8), 0.5),
            ]
        )

        evidences = [
            Evidence(
                name="voice_waveform_pattern",
                score=round(self._normalize(zcr, 0.02, 0.25), 4),
                details={
                    "zero_crossing_rate": round(zcr, 6),
                    "sample_rate": frame_rate,
                },
            ),
            Evidence(
                name="spectral_characteristics",
                score=round(self._normalize(spectral_flatness, 0.1, 0.8), 4),
                details={"spectral_flatness": round(spectral_flatness, 6)},
            ),
        ]

        return score, evidences

    def _analyze_unknown(self, path: Path) -> Tuple[float, List[Evidence]]:
        size_score = self._normalize(math.log(max(path.stat().st_size, 1), 10), 2.0, 8.0)
        evidence = Evidence(
            name="metadata_only",
            score=round(size_score, 4),
            details={"message": "no dedicated analyzer for this file type"},
        )
        return size_score, [evidence]

    def _analyze_image_frame(self, gray_frame: np.ndarray) -> Tuple[float, Dict[str, float]]:
        laplacian_var = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())
        edges = cv2.Canny(gray_frame, threshold1=100, threshold2=200)
        edge_density = float(np.count_nonzero(edges) / edges.size)

        freq_features = self._frequency_features(gray_frame)
        high_freq_ratio = freq_features["high_frequency_ratio"]

        frame_score = self._weighted_sum(
            [
                (self._normalize(laplacian_var, 40, 600), 0.35),
                (self._normalize(edge_density, 0.05, 0.28), 0.2),
                (self._normalize(high_freq_ratio, 0.08, 0.30), 0.45),
            ]
        )

        details = {
            "laplacian_var": laplacian_var,
            "edge_density": edge_density,
            "high_freq_ratio": high_freq_ratio,
        }
        return frame_score, details

    def _frequency_features(self, gray: np.ndarray) -> Dict[str, float]:
        normalized = gray.astype(np.float32) / 255.0
        fft = np.fft.fft2(normalized)
        shifted = np.fft.fftshift(fft)
        magnitude = np.abs(shifted)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        radius = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        radius_norm = radius / (np.max(radius) + 1e-8)

        high_band = magnitude[radius_norm > 0.6]
        low_band = magnitude[radius_norm <= 0.6]

        high_energy = float(np.mean(high_band)) if high_band.size else 0.0
        low_energy = float(np.mean(low_band)) if low_band.size else 1e-8
        high_frequency_ratio = high_energy / (low_energy + 1e-8)

        radial_profile = self._radial_profile(magnitude, radius.astype(np.int32))
        spike_score = float(np.std(radial_profile[-10:])) if radial_profile.size >= 10 else 0.0

        return {
            "high_frequency_ratio": high_frequency_ratio,
            "radial_spike_score": spike_score,
        }

    def _radial_profile(self, magnitude: np.ndarray, radius: np.ndarray) -> np.ndarray:
        r_max = int(np.max(radius))
        profile = np.zeros(r_max + 1, dtype=np.float64)
        counts = np.zeros(r_max + 1, dtype=np.int64)

        flat_r = radius.ravel()
        flat_mag = magnitude.ravel()

        for i in range(flat_r.size):
            r = flat_r[i]
            profile[r] += flat_mag[i]
            counts[r] += 1

        valid = counts > 0
        profile[valid] = profile[valid] / counts[valid]
        return profile

    def _predict_source(self, evidences: List[Evidence], media_type: str) -> str:
        if media_type == "audio":
            return "Likely neural voice synthesis (prototype guess)"

        freq = next((e for e in evidences if e.name == "frequency_signature"), None)
        temporal = next((e for e in evidences if e.name == "temporal_inconsistency"), None)

        if freq and freq.score > 0.72:
            return "Likely diffusion-style image generation (prototype guess)"
        if temporal and temporal.score > 0.7:
            return "Likely face-swap/manipulation pipeline (prototype guess)"
        return "Source model uncertain"

    @staticmethod
    def _normalize(value: float, min_v: float, max_v: float) -> float:
        if max_v <= min_v:
            return 0.0
        return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.0))

    @staticmethod
    def _clamp(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _weighted_sum(components: List[Tuple[float, float]]) -> float:
        total_weight = sum(weight for _, weight in components)
        if total_weight == 0:
            return 0.0
        weighted_total = sum(value * weight for value, weight in components)
        return weighted_total / total_weight
