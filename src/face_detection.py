"""MOD-02: Face Detection and Alignment.

Backend priority:
  1. MediaPipe Tasks API  (mediapipe >= 0.10 + face_detector.task model file)
  2. OpenCV Haar cascade  (always available, zero extra files)

Alignment uses an affine warp to a canonical 256x256 crop in both cases.
"""
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import setup_logger, normalize_image

logger = setup_logger(__name__)

# Canonical 5-point reference in a 256x256 face:
# (left-eye, right-eye, nose-tip, left-mouth, right-mouth)
_CANONICAL_256 = np.array(
    [[ 72.0, 100.0], [184.0, 100.0], [128.0, 148.0],
     [ 84.0, 196.0], [172.0, 196.0]], dtype=np.float32)

# Try mediapipe Tasks API (mp >= 0.10)
try:
    import mediapipe as _mp_module
    from mediapipe.tasks.python.vision import (
        FaceDetector as _MpFD,
        FaceDetectorOptions as _MpFDOpts,
        RunningMode as _RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
    _MP_TASKS_OK = True
except ImportError:
    _MP_TASKS_OK = False


class FaceDetectionError(Exception):
    pass


class FaceDetector:
    """Detect and align faces (OpenCV Haar by default; MediaPipe optional)."""

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        target_size: int = 256,
        normalization_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalization_std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
        mediapipe_model_path: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.norm_mean   = list(normalization_mean)
        self.norm_std    = list(normalization_std)
        self._mp_det     = None
        self._haar       = None
        self._backend    = "haar"
        self._init_backend(mediapipe_model_path)

    # ── Backend init ─────────────────────────────────────────────────────────

    def _init_backend(self, mp_model_path: Optional[str]) -> None:
        if mp_model_path and _MP_TASKS_OK and os.path.exists(mp_model_path):
            try:
                opts = _MpFDOpts(
                    base_options=_BaseOptions(model_asset_path=mp_model_path),
                    running_mode=_RunningMode.IMAGE,
                    min_detection_confidence=self.confidence_threshold,
                )
                self._mp_det  = _MpFD.create_from_options(opts)
                self._backend = "mediapipe"
                logger.info(f"FaceDetector: MediaPipe Tasks API ({mp_model_path})")
                return
            except Exception as e:
                logger.warning(f"MediaPipe init failed: {e}; falling back to Haar.")

        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade)
        if self._haar.empty():
            raise FaceDetectionError(f"Cannot load Haar cascade: {cascade}")
        self._backend = "haar"
        logger.info("FaceDetector: OpenCV Haar cascade.")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in RGB uint8 frame.  Returns list of dicts with
        keys: bbox [x1,y1,x2,y2], confidence float, landmarks (5,2)."""
        if self._backend == "mediapipe":
            return self._detect_mp(frame)
        return self._detect_haar(frame)

    # ── MediaPipe backend ─────────────────────────────────────────────────────

    def _detect_mp(self, frame: np.ndarray) -> List[Dict]:
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(frame))
        res    = self._mp_det.detect(mp_img)
        h, w   = frame.shape[:2]
        out    = []
        for d in res.detections:
            score = d.categories[0].score
            bb    = d.bounding_box
            x1, y1 = max(0, bb.origin_x), max(0, bb.origin_y)
            x2, y2 = min(w, bb.origin_x + bb.width), min(h, bb.origin_y + bb.height)
            pts    = self._mp_kps_to_5pt(d.keypoints, w, h)
            out.append({"bbox": [x1, y1, x2, y2], "confidence": float(score),
                        "landmarks": pts})
        return out

    @staticmethod
    def _mp_kps_to_5pt(kps, w, h) -> np.ndarray:
        def p(k): return [k.x * w, k.y * h]
        if len(kps) >= 4:
            le = p(kps[1]); re = p(kps[0]); ns = p(kps[2]); mc = p(kps[3])
            dx = abs(re[0] - le[0]) * 0.25
            lm = [mc[0] - dx, mc[1]]; rm = [mc[0] + dx, mc[1]]
        else:
            le = re = ns = lm = rm = [w / 2, h / 2]
        return np.array([le, re, ns, lm, rm], dtype=np.float32)

    # ── Haar backend ──────────────────────────────────────────────────────────

    def _detect_haar(self, frame: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._haar.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        out = []
        if len(rects) == 0:
            return out
        hi, wi = frame.shape[:2]
        for (x, y, bw, bh) in rects:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(wi, x + bw), min(hi, y + bh)
            pts = self._bbox_to_5pt(x1, y1, x2, y2)
            out.append({"bbox": [x1, y1, x2, y2], "confidence": 0.90,
                        "landmarks": pts})
        return out

    @staticmethod
    def _bbox_to_5pt(x1, y1, x2, y2) -> np.ndarray:
        """Geometric 5-point estimate from bbox using average face proportions."""
        w, h = x2 - x1, y2 - y1
        cx   = x1 + w / 2.0
        return np.array([
            [x1 + w * 0.30, y1 + h * 0.35],   # left eye
            [x1 + w * 0.70, y1 + h * 0.35],   # right eye
            [cx,             y1 + h * 0.55],   # nose
            [x1 + w * 0.35, y1 + h * 0.75],   # left mouth
            [x1 + w * 0.65, y1 + h * 0.75],   # right mouth
        ], dtype=np.float32)

    # ── Alignment ─────────────────────────────────────────────────────────────

    def align_face(self, frame: np.ndarray, landmarks: np.ndarray,
                   target_size: Optional[int] = None) -> np.ndarray:
        """Affine-warp to canonical crop. Returns float32 normalised (H,W,3)."""
        size = target_size or self.target_size
        dst  = _CANONICAL_256 * (size / 256.0)
        M, _ = cv2.estimateAffinePartial2D(
            landmarks.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), method=cv2.LMEDS)
        if M is None:
            return self._crop_fallback(frame, landmarks, size)
        aligned = cv2.warpAffine(frame, M, (size, size),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        return normalize_image(aligned, self.norm_mean, self.norm_std)

    def _crop_fallback(self, frame: np.ndarray,
                       landmarks: np.ndarray, size: int) -> np.ndarray:
        fh, fw = frame.shape[:2]
        x1 = int(max(0, landmarks[:, 0].min()))
        x2 = int(min(fw, landmarks[:, 0].max()))
        y1 = int(max(0, landmarks[:, 1].min()))
        y2 = int(min(fh, landmarks[:, 1].max()))
        crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame
        return normalize_image(cv2.resize(crop, (size, size)), self.norm_mean, self.norm_std)

    def extract_face_crops(self, frame: np.ndarray,
                           min_size: int = 50, max_size: int = 500) -> List[np.ndarray]:
        """Detect → size-filter → align → return list of normalised crops."""
        crops = []
        for d in self.detect_faces(frame):
            x1, y1, x2, y2 = d["bbox"]
            fw, fh = x2 - x1, y2 - y1
            if min_size <= fw <= max_size and min_size <= fh <= max_size:
                crops.append(self.align_face(frame, d["landmarks"]))
        return crops

    @property
    def backend(self) -> str:
        return self._backend

    def close(self) -> None:
        if self._mp_det:
            self._mp_det.close()
            self._mp_det = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __repr__(self):
        return f"FaceDetector(backend={self._backend}, threshold={self.confidence_threshold})"