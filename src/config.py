"""Configuration management for the deepfake detection system."""
import os
import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    cnn_model: str  = "efficientnet_b4"
    pretrained: bool = True
    feature_dim: int = 1024
    weights_dir: str = "models/weights"


@dataclass
class DetectionConfig:
    face_confidence_threshold: float = 0.95
    face_size_min: int = 50
    face_size_max: int = 500
    target_face_size: int = 256
    frame_skip: int = 5


@dataclass
class EnsembleConfig:
    cnn_weight: float          = 0.45
    optical_flow_weight: float = 0.30
    frequency_weight: float    = 0.15
    landmarks_weight: float    = 0.10
    temperature: float         = 1.2
    temporal_alpha: float      = 0.7
    classification_threshold: float = 0.5


@dataclass
class NormalizationConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class HardwareConfig:
    device: str      = "cuda"
    batch_size: int  = 8
    num_workers: int = 4


@dataclass
class Config:
    model:         ModelConfig         = field(default_factory=ModelConfig)
    detection:     DetectionConfig     = field(default_factory=DetectionConfig)
    ensemble:      EnsembleConfig      = field(default_factory=EnsembleConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    hardware:      HardwareConfig      = field(default_factory=HardwareConfig)
    # Paths
    raw_videos:     str = "data/raw_videos"
    processed:      str = "data/processed"
    datasets:       str = "data/datasets"
    predictions:    str = "results/predictions"
    visualizations: str = "results/visualizations"
    metrics:        str = "results/metrics"


def load_config(config_path: str = "config.yaml") -> Config:
    """Load a Config from YAML; falls back to defaults if file is missing."""
    cfg = Config()
    if not os.path.exists(config_path):
        return cfg

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    if "model" in data:
        m = data["model"]
        cfg.model = ModelConfig(
            cnn_model   = m.get("cnn_model",   cfg.model.cnn_model),
            pretrained  = m.get("pretrained",  cfg.model.pretrained),
            feature_dim = m.get("feature_dim", cfg.model.feature_dim),
            weights_dir = m.get("weights_dir", cfg.model.weights_dir),
        )

    if "detection" in data:
        d = data["detection"]
        cfg.detection = DetectionConfig(
            face_confidence_threshold = d.get("face_confidence_threshold", cfg.detection.face_confidence_threshold),
            face_size_min  = d.get("face_size_min",  cfg.detection.face_size_min),
            face_size_max  = d.get("face_size_max",  cfg.detection.face_size_max),
            target_face_size = d.get("target_face_size", cfg.detection.target_face_size),
            frame_skip     = d.get("frame_skip",     cfg.detection.frame_skip),
        )

    if "ensemble" in data:
        e = data["ensemble"]
        w = e.get("weights", {})
        cfg.ensemble = EnsembleConfig(
            cnn_weight           = w.get("cnn",          cfg.ensemble.cnn_weight),
            optical_flow_weight  = w.get("optical_flow", cfg.ensemble.optical_flow_weight),
            frequency_weight     = w.get("frequency",    cfg.ensemble.frequency_weight),
            landmarks_weight     = w.get("landmarks",    cfg.ensemble.landmarks_weight),
            temperature          = e.get("temperature",  cfg.ensemble.temperature),
            temporal_alpha       = e.get("temporal_alpha", cfg.ensemble.temporal_alpha),
            classification_threshold = e.get("classification_threshold", cfg.ensemble.classification_threshold),
        )

    if "normalization" in data:
        n = data["normalization"]
        cfg.normalization = NormalizationConfig(
            mean = n.get("mean", cfg.normalization.mean),
            std  = n.get("std",  cfg.normalization.std),
        )

    if "hardware" in data:
        h = data["hardware"]
        cfg.hardware = HardwareConfig(
            device      = h.get("device",      cfg.hardware.device),
            batch_size  = h.get("batch_size",  cfg.hardware.batch_size),
            num_workers = h.get("num_workers", cfg.hardware.num_workers),
        )

    if "paths" in data:
        p = data["paths"]
        cfg.raw_videos     = p.get("raw_videos",     cfg.raw_videos)
        cfg.processed      = p.get("processed",      cfg.processed)
        cfg.datasets       = p.get("datasets",       cfg.datasets)
        cfg.predictions    = p.get("predictions",    cfg.predictions)
        cfg.visualizations = p.get("visualizations", cfg.visualizations)
        cfg.metrics        = p.get("metrics",        cfg.metrics)

    device = (cfg.hardware.device or "cpu").strip().lower()
    if device == "auto":
        try:
            import torch
            cfg.hardware.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.hardware.device = "cpu"
    elif device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                cfg.hardware.device = "cpu"
        except Exception:
            cfg.hardware.device = "cpu"
    else:
        cfg.hardware.device = "cpu"

    return cfg