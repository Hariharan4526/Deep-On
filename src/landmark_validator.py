"""MOD-06: Facial Landmark Validator.

Backend priority:
  1. MediaPipe Tasks FaceLandmarker  (requires face_landmarker.task model file)
  2. Geometric estimation from face bbox (always available, zero extra files)

Geometric validation checks EAR, MAR, symmetry, and aspect ratio regardless
of which backend detected the landmarks.
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import setup_logger

logger = setup_logger(__name__)

# Try mediapipe Tasks FaceLandmarker
try:
    import mediapipe as _mp_module
    from mediapipe.tasks.python.vision import (
        FaceLandmarker as _MpFL,
        FaceLandmarkerOptions as _MpFLOpts,
        RunningMode as _RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
    _MP_LANDMARKER_OK = True
except ImportError:
    _MP_LANDMARKER_OK = False

# MediaPipe 468-point index constants (used when MediaPipe backend is active)
_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_MOUTH_PTS = [61, 39, 0, 269, 291, 405, 17, 181]
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
              361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
              176, 149, 150, 136, 172,  58, 132,  93, 234, 127,
              162,  21,  54, 103,  67, 109]
_NOSE_TIP    = 1
_NOSE_BRIDGE = 6


class LandmarkValidator:
    """Validate facial geometry. Works with or without MediaPipe model files."""

    EAR_LO, EAR_HI = 0.10, 0.50
    MAR_LO, MAR_HI = 0.0,  0.80

    def __init__(self, landmarker_model_path: Optional[str] = None):
        """
        Args:
            landmarker_model_path: Path to face_landmarker.task.
                If provided and valid, use MediaPipe 468-point mesh.
                Otherwise, use OpenCV Haar + geometric estimation.
        """
        self._mp_lm    = None
        self._haar     = None
        self._backend  = "geometric"
        self._init_backend(landmarker_model_path)

    # ── Backend init ──────────────────────────────────────────────────────────

    def _init_backend(self, model_path: Optional[str]) -> None:
        import os
        if model_path and _MP_LANDMARKER_OK and os.path.exists(model_path):
            try:
                opts = _MpFLOpts(
                    base_options=_BaseOptions(model_asset_path=model_path),
                    running_mode=_RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                )
                self._mp_lm   = _MpFL.create_from_options(opts)
                self._backend = "mediapipe"
                logger.info(f"LandmarkValidator: MediaPipe FaceLandmarker ({model_path})")
                return
            except Exception as e:
                logger.warning(f"MediaPipe Landmarker init failed: {e}; using geometric.")

        # Geometric backend: Haar for detection, estimated landmark positions
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade)
        self._backend = "geometric"
        logger.info("LandmarkValidator: geometric estimation (OpenCV Haar).")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Detect facial landmarks.

        Returns:
          MediaPipe backend : (468, 3) array  (x, y, z in [0,1])
          Geometric backend : (20, 2) array   (x, y in [0,1])
          None if detection fails.
        """
        if self._backend == "mediapipe":
            return self._detect_mp(face_image)
        return self._detect_geometric(face_image)

    # ── MediaPipe backend ─────────────────────────────────────────────────────

    def _detect_mp(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        import mediapipe as mp
        img_u8  = self._to_u8(face_image)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(img_u8))
        result  = self._mp_lm.detect(mp_img)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    # ── Geometric backend ─────────────────────────────────────────────────────

    def _detect_geometric(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Estimate key landmark positions from face bbox using fixed proportions.

        Returns a (20, 2) array of estimated x/y in [0,1].
        Rows: [l_eye_inner, l_eye_outer, l_eye_top, l_eye_bot,
               r_eye_inner, r_eye_outer, r_eye_top, r_eye_bot,
               nose_tip, nose_base, mouth_left, mouth_right,
               mouth_top, mouth_bot, l_ear, r_ear, chin, forehead,
               l_cheek, r_cheek]
        """
        img_u8  = self._to_u8(face_image)
        gray    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        gray    = cv2.equalizeHist(gray)
        h, w    = img_u8.shape[:2]

        rects = self._haar.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        if len(rects) == 0:
            # No face found — synthesise centred estimate for a 256x256 input
            return self._centred_estimate(w, h)

        x, y, bw, bh = rects[0]
        return self._bbox_to_landmarks(x, y, bw, bh, w, h)

    def _centred_estimate(self, w: int, h: int) -> np.ndarray:
        """Fallback: assume face fills the whole crop."""
        return self._bbox_to_landmarks(0, 0, w, h, w, h)

    @staticmethod
    def _bbox_to_landmarks(
        bx: int, by: int, bw: int, bh: int,
        iw: int, ih: int,
    ) -> np.ndarray:
        """Map bbox proportions to 20 landmark positions (x/y in [0,1])."""
        def p(rx, ry): return [(bx + bw * rx) / iw, (by + bh * ry) / ih]

        # Eyes (4 pts each, simplified to 2 rows: inner/outer, top/bot)
        le_inner = p(0.35, 0.35);  le_outer = p(0.20, 0.35)
        le_top   = p(0.28, 0.30);  le_bot   = p(0.28, 0.40)
        re_inner = p(0.65, 0.35);  re_outer = p(0.80, 0.35)
        re_top   = p(0.72, 0.30);  re_bot   = p(0.72, 0.40)
        # Nose
        nose_tip  = p(0.50, 0.58); nose_base = p(0.50, 0.65)
        # Mouth
        m_left  = p(0.35, 0.75); m_right = p(0.65, 0.75)
        m_top   = p(0.50, 0.70); m_bot   = p(0.50, 0.82)
        # Ears / chin / forehead / cheeks
        l_ear   = p(0.05, 0.45); r_ear    = p(0.95, 0.45)
        chin    = p(0.50, 0.95); forehead = p(0.50, 0.10)
        l_chk   = p(0.15, 0.60); r_chk    = p(0.85, 0.60)

        return np.array([
            le_inner, le_outer, le_top, le_bot,
            re_inner, re_outer, re_top, re_bot,
            nose_tip, nose_base,
            m_left, m_right, m_top, m_bot,
            l_ear, r_ear, chin, forehead,
            l_chk, r_chk,
        ], dtype=np.float32)

    # ── EAR / MAR (work on both backends via index mapping) ──────────────────

    def compute_eye_aspect_ratio(
        self, lm: np.ndarray, left: bool = True
    ) -> Tuple[float, bool]:
        """EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)."""
        if self._backend == "mediapipe" and lm.shape[0] == 468:
            idx = _LEFT_EYE if left else _RIGHT_EYE
            p   = lm[idx, :2]
            ear = (np.linalg.norm(p[1] - p[5]) +
                   np.linalg.norm(p[2] - p[4])) / \
                  (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-8)
        else:
            # Geometric: rows 0-3 = left eye, 4-7 = right eye
            offset = 0 if left else 4
            p = lm[offset:offset + 4]   # inner, outer, top, bot
            ear = (np.linalg.norm(p[2] - p[3])) / \
                  (np.linalg.norm(p[0] - p[1]) + 1e-8)
        return float(ear), self.EAR_LO <= ear <= self.EAR_HI

    def compute_mouth_aspect_ratio(
        self, lm: np.ndarray
    ) -> Tuple[float, bool]:
        """Simplified MAR from outer mouth points."""
        if self._backend == "mediapipe" and lm.shape[0] == 468:
            p   = lm[_MOUTH_PTS, :2]
            v1  = np.linalg.norm(p[1] - p[6])
            v2  = np.linalg.norm(p[2] - p[5])
            v3  = np.linalg.norm(p[3] - p[7])
            h   = np.linalg.norm(p[0] - p[4]) + 1e-8
            mar = (v1 + v2 + v3) / (3.0 * h)
        else:
            # Geometric: rows 10-13 = m_left, m_right, m_top, m_bot
            p   = lm[10:14]
            vert = np.linalg.norm(p[2] - p[3])
            horiz = np.linalg.norm(p[0] - p[1]) + 1e-8
            mar = vert / horiz
        return float(mar), self.MAR_LO <= mar <= self.MAR_HI

    # ── Geometry checks ───────────────────────────────────────────────────────

    def check_facial_geometry(self, lm: np.ndarray) -> Dict[str, float]:
        """Constraint violation scores in [0, 1]."""
        scores: Dict[str, float] = {}

        if self._backend == "mediapipe" and lm.shape[0] == 468:
            le_c   = lm[_LEFT_EYE,  :2].mean(0)
            re_c   = lm[_RIGHT_EYE, :2].mean(0)
            nose_x = lm[_NOSE_TIP,   0]
            oval   = lm[_FACE_OVAL,  :2]
            nb_x   = lm[_NOSE_BRIDGE, 0]
        else:
            le_c   = lm[0:4,  :2].mean(0)       # left eye centre
            re_c   = lm[4:8,  :2].mean(0)       # right eye centre
            nose_x = float(lm[8,  0])
            nb_x   = float(lm[9,  0])
            oval   = np.vstack([lm[0:2], lm[4:6], lm[14:20]])  # rough oval

        mid_x = (le_c[0] + re_c[0]) / 2.0

        # 1. Symmetry
        sym_err = abs(nose_x - mid_x) * 2.0
        scores["symmetry_violation"] = float(np.clip(sym_err / 0.05, 0, 1))

        # 2. EAR L/R consistency
        ear_l, _ = self.compute_eye_aspect_ratio(lm, True)
        ear_r, _ = self.compute_eye_aspect_ratio(lm, False)
        ear_diff  = abs(ear_l - ear_r) / (max(ear_l, ear_r) + 1e-8)
        scores["ear_asymmetry"] = float(np.clip(ear_diff / 0.3, 0, 1))

        # 3. Face aspect ratio (width/height)
        fw = oval[:, 0].max() - oval[:, 0].min()
        fh = oval[:, 1].max() - oval[:, 1].min() + 1e-8
        whr = fw / fh
        scores["aspect_violation"] = float(
            0.0 if (0.55 <= whr <= 0.90)
            else np.clip(abs(whr - 0.72) / 0.3, 0, 1)
        )

        # 4. Iris deviation
        eye_span = abs(le_c[0] - re_c[0]) + 1e-8
        scores["iris_deviation"] = float(
            np.clip(abs(nb_x - mid_x) / eye_span / 0.2, 0, 1)
        )
        return scores

    # ── Combined score ────────────────────────────────────────────────────────

    def score_landmark_geometry(self, face_image: np.ndarray) -> float:
        """Deepfake probability from geometry validation → [0, 1]."""
        lm = self.detect_landmarks(face_image)
        if lm is None:
            return 0.5

        ear_l, ok_el = self.compute_eye_aspect_ratio(lm, True)
        ear_r, ok_er = self.compute_eye_aspect_ratio(lm, False)
        mar,   ok_m  = self.compute_mouth_aspect_ratio(lm)

        invalidity = float(not ok_el) * 0.3 + float(not ok_er) * 0.3 + float(not ok_m) * 0.2
        geo_score  = float(np.mean(list(self.check_facial_geometry(lm).values())))
        return float(np.clip(0.5 * invalidity + 0.5 * geo_score, 0, 1))

    def get_full_analysis(self, face_image: np.ndarray) -> Dict:
        """Full diagnostic dict for debugging."""
        lm = self.detect_landmarks(face_image)
        if lm is None:
            return {"detected": False, "p_landmarks": 0.5}
        ear_l, ok_el = self.compute_eye_aspect_ratio(lm, True)
        ear_r, ok_er = self.compute_eye_aspect_ratio(lm, False)
        mar,   ok_m  = self.compute_mouth_aspect_ratio(lm)
        return {
            "detected":     True,
            "backend":      self._backend,
            "ear_left":     round(ear_l, 4), "ear_left_ok":  ok_el,
            "ear_right":    round(ear_r, 4), "ear_right_ok": ok_er,
            "mar":          round(mar,   4), "mar_ok":       ok_m,
            "geometry":     self.check_facial_geometry(lm),
            "p_landmarks":  round(self.score_landmark_geometry(face_image), 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def close(self):
        if self._mp_lm:
            self._mp_lm.close()
            self._mp_lm = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __repr__(self):
        return f"LandmarkValidator(backend={self._backend})"