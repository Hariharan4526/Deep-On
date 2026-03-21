"""MOD-01: Video Input Handler.

Handles video loading, metadata extraction, and frame-by-frame iteration.
No GPU or torch required.
"""
import os
from typing import Dict, Any, Generator, Optional, Tuple

import cv2
import numpy as np

from .utils import setup_logger, bgr_to_rgb

logger = setup_logger(__name__)


class VideoHandlerError(Exception):
    pass


class VideoHandler:
    """Load a video file and iterate over frames at a configurable skip rate."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv"}

    def __init__(self, video_path: str, frame_skip: int = 5):
        """
        Args:
            video_path : Path to video file.
            frame_skip : Yield every nth frame (1 = every frame).
        """
        self.video_path = video_path
        self.frame_skip = max(1, frame_skip)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[Dict[str, Any]] = None

        self._validate()
        self._open()

    # ── Validation & open ────────────────────────────────────────────────────

    def _validate(self) -> None:
        if not os.path.exists(self.video_path):
            raise VideoHandlerError(f"File not found: {self.video_path}")
        ext = os.path.splitext(self.video_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(
                f"Extension '{ext}' is not in the known-supported list "
                f"{self.SUPPORTED_EXTENSIONS}. Attempting to open anyway."
            )

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise VideoHandlerError(
                f"OpenCV could not open: {self.video_path}. "
                "Check that the codec is installed (e.g. ffmpeg)."
            )
        logger.info(f"Opened video: {self.video_path}")

    # ── Metadata ─────────────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        """Return a dict with fps, frame_count, width, height, duration, codec."""
        if self._metadata is not None:
            return self._metadata

        cap = self._cap
        fps         = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration    = (frame_count / fps) if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ]).strip()

        self._metadata = {
            "fps":         fps,
            "frame_count": frame_count,
            "width":       width,
            "height":      height,
            "duration":    duration,
            "codec":       codec,
        }
        return self._metadata

    def get_frame_count(self) -> int:
        return self.get_metadata()["frame_count"]

    def get_fps(self) -> float:
        return self.get_metadata()["fps"]

    # ── Frame extraction ─────────────────────────────────────────────────────

    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Yield (rgb_frame, frame_index) tuples, every frame_skip frames.

        Args:
            start_frame : First frame index (inclusive).
            end_frame   : Last frame index (exclusive). None → all frames.

        Yields:
            (H×W×3 uint8 RGB ndarray, frame_index)
        """
        total = self.get_frame_count()
        if end_frame is None:
            end_frame = total
        end_frame = min(end_frame, total)

        if start_frame >= end_frame:
            logger.warning("start_frame >= end_frame — no frames to extract.")
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx = start_frame

        while idx < end_frame:
            ret, frame_bgr = self._cap.read()
            if not ret:
                logger.debug(f"Could not read frame {idx}; stopping.")
                break

            if frame_bgr is None or frame_bgr.size == 0:
                logger.warning(f"Empty / corrupted frame at index {idx}; skipping.")
            else:
                yield bgr_to_rgb(frame_bgr), idx

            idx += self.frame_skip
            if self.frame_skip > 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def read_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Read a single frame by index. Returns RGB uint8 ndarray or None."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            return None
        return bgr_to_rgb(frame_bgr)

    # ── Resource management ──────────────────────────────────────────────────

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self) -> str:
        m = self.get_metadata()
        return (
            f"VideoHandler('{os.path.basename(self.video_path)}' | "
            f"{m['frame_count']} frames @ {m['fps']:.1f} fps | "
            f"{m['width']}×{m['height']} | codec={m['codec']})"
        )