"""Utility functions for the deepfake detection system."""
import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import cv2

# ── torch is optional (only needed for CNN module) ──────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────── Logging ────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────── File/Dir Helpers ───────────────────────────────

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def _json_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────── Image Helpers ──────────────────────────────────

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def normalize_image(
    image: np.ndarray,
    mean: List[float] = (0.485, 0.456, 0.406),
    std: List[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize HxWxC uint8 image → float32 using ImageNet stats."""
    img = image.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    return (img - mean_arr) / std_arr


def denormalize_image(
    image: np.ndarray,
    mean: List[float] = (0.485, 0.456, 0.406),
    std: List[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse ImageNet normalization → uint8."""
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    img = image * std_arr + mean_arr
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def image_to_tensor(image: np.ndarray):
    """Convert HxWxC float32 → 1xCxHxW tensor (requires torch)."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed")
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


# ─────────────────────────── Timing ─────────────────────────────────────────

class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ─────────────────────────── Device ─────────────────────────────────────────

def get_device(prefer_cuda: bool = True):
    if not _TORCH_AVAILABLE:
        return "cpu"
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────── Metrics ────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auc_roc"] = float("nan")
    return metrics


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def logit(p: float, eps: float = 1e-7) -> float:
    p = float(np.clip(p, eps, 1 - eps))
    return float(np.log(p / (1 - p)))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")