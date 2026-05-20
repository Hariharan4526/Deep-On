"""Configuration management for the deepfake detection system."""
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict


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
    # ENHANCEMENT: Adaptive frame processing
    adaptive_frame_processing: bool = True
    blur_threshold: float = 50.0
    duplicate_threshold: float = 0.95
    motion_threshold: float = 0.05
    min_frame_interval: int = 2


@dataclass
class EnsembleConfig:
    cnn_weight: float          = 0.40
    optical_flow_weight: float = 0.15
    frequency_weight: float    = 0.15
    landmarks_weight: float    = 0.10
    temporal_weight: float     = 0.10  # NEW
    forensic_weight: float     = 0.10  # NEW
    temperature: float         = 1.2
    temporal_alpha: float      = 0.7
    classification_threshold: float = 0.5


@dataclass
class TemporalConfig:
    enabled: bool = True
    use_lstm: bool = True
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 2
    sequence_length: int = 32


@dataclass
class ForensicConfig:
    enabled: bool = True
    gan_sensitivity: float = 0.5
    boundary_sensitivity: float = 0.6
    texture_sensitivity: float = 0.4
    compression_sensitivity: float = 0.5


@dataclass
class RobustnessConfig:
    enabled: bool = True
    test_time_augmentation: bool = True
    tta_num_augmentations: int = 4
    training_augmentation: bool = True


@dataclass
class TrainingConfig:
    mixed_dataset: bool = True
    class_balancing: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 5
    learning_rate_schedule: bool = True
    initial_learning_rate: float = 0.001
    lr_decay_factor: float = 0.1
    lr_decay_steps: int = 10
    validation_metrics: bool = True


@dataclass
class EvaluationConfig:
    compute_roc_auc: bool = True
    compute_f1_score: bool = True
    compute_confusion_matrix: bool = True
    analyze_false_positives: bool = True


@dataclass
class NormalizationConfig:
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class HardwareConfig:
    device: str      = "cuda"
    batch_size: int  = 8
    num_workers: int = 4
    use_batch_inference: bool = True
    max_batch_size: int = 32
    mixed_precision: bool = False


@dataclass
class OutputConfig:
    save_gradcam: bool = True
    save_attention: bool = True
    save_report: bool = True
    report_format: str = "json"
    save_confidence_timeline: bool = True
    save_suspicious_frames: bool = True
    save_forensic_report: bool = True
    save_temporal_analysis: bool = True


@dataclass
class Config:
    model:         ModelConfig         = field(default_factory=ModelConfig)
    detection:     DetectionConfig     = field(default_factory=DetectionConfig)
    ensemble:      EnsembleConfig      = field(default_factory=EnsembleConfig)
    temporal:      TemporalConfig      = field(default_factory=TemporalConfig)
    forensic:      ForensicConfig      = field(default_factory=ForensicConfig)
    robustness:    RobustnessConfig    = field(default_factory=RobustnessConfig)
    training:      TrainingConfig      = field(default_factory=TrainingConfig)
    evaluation:    EvaluationConfig    = field(default_factory=EvaluationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    hardware:      HardwareConfig      = field(default_factory=HardwareConfig)
    output:        OutputConfig        = field(default_factory=OutputConfig)
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
            adaptive_frame_processing = d.get("adaptive_frame_processing", cfg.detection.adaptive_frame_processing),
            blur_threshold = d.get("blur_threshold", cfg.detection.blur_threshold),
            duplicate_threshold = d.get("duplicate_threshold", cfg.detection.duplicate_threshold),
            motion_threshold = d.get("motion_threshold", cfg.detection.motion_threshold),
            min_frame_interval = d.get("min_frame_interval", cfg.detection.min_frame_interval),
        )

    if "ensemble" in data:
        e = data["ensemble"]
        w = e.get("weights", {})
        cfg.ensemble = EnsembleConfig(
            cnn_weight           = w.get("cnn",          cfg.ensemble.cnn_weight),
            optical_flow_weight  = w.get("optical_flow", cfg.ensemble.optical_flow_weight),
            frequency_weight     = w.get("frequency",    cfg.ensemble.frequency_weight),
            landmarks_weight     = w.get("landmarks",    cfg.ensemble.landmarks_weight),
            temporal_weight      = w.get("temporal",     cfg.ensemble.temporal_weight),
            forensic_weight      = w.get("forensic",     cfg.ensemble.forensic_weight),
            temperature          = e.get("temperature",  cfg.ensemble.temperature),
            temporal_alpha       = e.get("temporal_alpha", cfg.ensemble.temporal_alpha),
            classification_threshold = e.get("classification_threshold", cfg.ensemble.classification_threshold),
        )

    if "temporal" in data:
        t = data["temporal"]
        cfg.temporal = TemporalConfig(
            enabled = t.get("enabled", cfg.temporal.enabled),
            use_lstm = t.get("use_lstm", cfg.temporal.use_lstm),
            lstm_hidden_size = t.get("lstm_hidden_size", cfg.temporal.lstm_hidden_size),
            lstm_num_layers = t.get("lstm_num_layers", cfg.temporal.lstm_num_layers),
            sequence_length = t.get("sequence_length", cfg.temporal.sequence_length),
        )

    if "forensic" in data:
        f = data["forensic"]
        cfg.forensic = ForensicConfig(
            enabled = f.get("enabled", cfg.forensic.enabled),
            gan_sensitivity = f.get("gan_sensitivity", cfg.forensic.gan_sensitivity),
            boundary_sensitivity = f.get("boundary_sensitivity", cfg.forensic.boundary_sensitivity),
            texture_sensitivity = f.get("texture_sensitivity", cfg.forensic.texture_sensitivity),
            compression_sensitivity = f.get("compression_sensitivity", cfg.forensic.compression_sensitivity),
        )

    if "robustness" in data:
        r = data["robustness"]
        cfg.robustness = RobustnessConfig(
            enabled = r.get("enabled", cfg.robustness.enabled),
            test_time_augmentation = r.get("test_time_augmentation", cfg.robustness.test_time_augmentation),
            tta_num_augmentations = r.get("tta_num_augmentations", cfg.robustness.tta_num_augmentations),
            training_augmentation = r.get("training_augmentation", cfg.robustness.training_augmentation),
        )

    if "training" in data:
        tr = data["training"]
        cfg.training = TrainingConfig(
            mixed_dataset = tr.get("mixed_dataset", cfg.training.mixed_dataset),
            class_balancing = tr.get("class_balancing", cfg.training.class_balancing),
            early_stopping = tr.get("early_stopping", cfg.training.early_stopping),
            early_stopping_patience = tr.get("early_stopping_patience", cfg.training.early_stopping_patience),
            learning_rate_schedule = tr.get("learning_rate_schedule", cfg.training.learning_rate_schedule),
            initial_learning_rate = tr.get("initial_learning_rate", cfg.training.initial_learning_rate),
            lr_decay_factor = tr.get("lr_decay_factor", cfg.training.lr_decay_factor),
            lr_decay_steps = tr.get("lr_decay_steps", cfg.training.lr_decay_steps),
            validation_metrics = tr.get("validation_metrics", cfg.training.validation_metrics),
        )

    if "evaluation" in data:
        ev = data["evaluation"]
        cfg.evaluation = EvaluationConfig(
            compute_roc_auc = ev.get("compute_roc_auc", cfg.evaluation.compute_roc_auc),
            compute_f1_score = ev.get("compute_f1_score", cfg.evaluation.compute_f1_score),
            compute_confusion_matrix = ev.get("compute_confusion_matrix", cfg.evaluation.compute_confusion_matrix),
            analyze_false_positives = ev.get("analyze_false_positives", cfg.evaluation.analyze_false_positives),
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
            use_batch_inference = h.get("use_batch_inference", cfg.hardware.use_batch_inference),
            max_batch_size = h.get("max_batch_size", cfg.hardware.max_batch_size),
            mixed_precision = h.get("mixed_precision", cfg.hardware.mixed_precision),
        )

    if "output" in data:
        o = data["output"]
        cfg.output = OutputConfig(
            save_gradcam = o.get("save_gradcam", cfg.output.save_gradcam),
            save_attention = o.get("save_attention", cfg.output.save_attention),
            save_report = o.get("save_report", cfg.output.save_report),
            report_format = o.get("report_format", cfg.output.report_format),
            save_confidence_timeline = o.get("save_confidence_timeline", cfg.output.save_confidence_timeline),
            save_suspicious_frames = o.get("save_suspicious_frames", cfg.output.save_suspicious_frames),
            save_forensic_report = o.get("save_forensic_report", cfg.output.save_forensic_report),
            save_temporal_analysis = o.get("save_temporal_analysis", cfg.output.save_temporal_analysis),
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