"""MOD-08: Visualization Module.

Grad-CAM overlays, SE-attention bar charts, temporal confidence plots,
and structured JSON result reports. Torch is optional — falls back to
plain matplotlib overlays when not available.
"""
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .utils import setup_logger, ensure_dirs, save_json, timestamp, denormalize_image

logger = setup_logger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class VisualizationModule:
    """Generate Grad-CAM maps, attention charts, temporal plots and reports."""

    def __init__(
        self,
        cnn_model=None,
        output_dir: str = "results/visualizations",
    ):
        self.model      = cnn_model
        self.output_dir = output_dir
        ensure_dirs(output_dir)
        logger.info(f"VisualizationModule ready  output_dir={output_dir}")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────

    def generate_gradcam(
        self,
        face_image: np.ndarray,
        target_class: int = 1,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Grad-CAM heatmap overlaid on the face crop.

        Args:
            face_image   : (256,256,3) float32 normalised array.
            target_class : 1=deepfake, 0=authentic.
            save_path    : Optional path to save PNG.

        Returns:
            (256,256,3) uint8 RGB overlay.
        """
        if not _TORCH_OK or self.model is None or self.model.using_fallback:
            return self._plain_overlay(face_image, save_path)

        self.model._model.eval()
        device = next(self.model._model.parameters()).device

        from .utils import image_to_tensor
        t = image_to_tensor(face_image).to(device).requires_grad_(True)

        feat, logit, fm = self.model._model.forward_with_featmap(t)
        if fm is None:
            return self._plain_overlay(face_image, save_path)

        fm.retain_grad()
        self.model._model.zero_grad()
        score = logit if target_class == 1 else -logit
        score.mean().backward()

        if fm.grad is None:
            return self._plain_overlay(face_image, save_path)

        # α_c = GAP(∂score / ∂A_c)
        alpha = fm.grad.mean(dim=(2, 3), keepdim=True)
        cam   = F.relu((alpha * fm).sum(dim=1, keepdim=True))
        cam   = F.interpolate(cam, size=(256, 256), mode="bilinear",
                              align_corners=False)
        cam   = cam.squeeze().detach().cpu().numpy()
        cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        base     = denormalize_image(face_image)
        heatmap  = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay  = cv2.addWeighted(base, 0.60, heatmap, 0.40, 0)

        if save_path:
            self._save_image(overlay, save_path)
        return overlay

    def _plain_overlay(
        self, face_image: np.ndarray, save_path: Optional[str]
    ) -> np.ndarray:
        """Fallback: return denormalised face (no gradient available)."""
        overlay = denormalize_image(face_image)
        if save_path:
            self._save_image(overlay, save_path)
        return overlay

    # ── SE Attention ──────────────────────────────────────────────────────────

    def generate_attention_maps(
        self,
        face_image: np.ndarray,
        save_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Bar chart of SE channel attention weights + original face side-by-side."""
        if not _TORCH_OK or self.model is None or self.model.using_fallback:
            logger.debug("Attention map skipped (no torch / fallback mode).")
            return None

        device = next(self.model._model.parameters()).device
        from .utils import image_to_tensor
        with torch.no_grad():
            t  = image_to_tensor(face_image).to(device)
            fm = self.model._model.backbone(t)
            se_weights = (
                self.model._model.se.fc(
                    self.model._model.se.pool(fm).view(1, -1)
                ).squeeze().cpu().numpy()
            )

        n_show = min(64, len(se_weights))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(range(n_show), se_weights[:n_show],
                    color="steelblue", alpha=0.8)
        axes[0].set_title(f"SE Channel Attention (first {n_show} channels)")
        axes[0].set_xlabel("Channel index")
        axes[0].set_ylabel("Attention weight")
        axes[0].axhline(se_weights[:n_show].mean(), color="red",
                        linestyle="--", linewidth=1, label="mean")
        axes[0].legend()

        axes[1].imshow(denormalize_image(face_image))
        axes[1].axis("off")
        axes[1].set_title("Input face")

        plt.tight_layout()
        arr = self._fig_to_array(fig)

        if save_path:
            ensure_dirs(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.debug(f"Attention map saved → {save_path}")
        plt.close(fig)
        return arr

    # ── Temporal plot ─────────────────────────────────────────────────────────

    def visualize_temporal_predictions(
        self,
        frame_results: List[Dict],
        save_path: Optional[str] = None,
    ) -> str:
        """Line chart of raw + smoothed ensemble probability across frames.

        Returns the path where the PNG was saved.
        """
        if not frame_results:
            return ""

        frames   = [r.get("frame_index", i) for i, r in enumerate(frame_results)]
        raw      = [r["ensemble_prob"]  for r in frame_results]
        smoothed = [r["smoothed_prob"]  for r in frame_results]
        n_fake   = sum(1 for r in frame_results if r["label"] == "Deepfake")
        pct_fake = 100.0 * n_fake / len(frame_results)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(frames, raw,      alpha=0.45, linewidth=1.2,
                color="royalblue", label="Raw ensemble")
        ax.plot(frames, smoothed, linewidth=2.0,
                color="navy",       label="EMA smoothed")
        ax.axhline(0.5, color="crimson", linestyle="--",
                   linewidth=1, alpha=0.7, label="Decision boundary (0.5)")
        ax.fill_between(frames, smoothed, 0.5,
                        where=[s > 0.5 for s in smoothed],
                        alpha=0.15, color="red",  label="Deepfake region")
        ax.fill_between(frames, smoothed, 0.5,
                        where=[s <= 0.5 for s in smoothed],
                        alpha=0.10, color="green", label="Authentic region")

        ax.set_xlabel("Frame index", fontsize=11)
        ax.set_ylabel("Deepfake probability", fontsize=11)
        ax.set_title(
            f"Temporal prediction  |  {len(frame_results)} frames  |  "
            f"{pct_fake:.1f}% flagged deepfake",
            fontsize=12,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = save_path or os.path.join(
            self.output_dir, f"temporal_{timestamp()}.png"
        )
        ensure_dirs(os.path.dirname(path))
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Temporal plot saved → {path}")
        return path

    # ── Pathway radar chart ───────────────────────────────────────────────────

    def visualize_pathway_scores(
        self,
        pathway_scores: Dict[str, float],
        label: str,
        confidence: float,
        save_path: Optional[str] = None,
    ) -> str:
        """Radar / spider chart of the four pathway scores."""
        names  = ["CNN", "Optical\nFlow", "Frequency", "Landmarks"]
        values = [
            pathway_scores.get("cnn",          0.5),
            pathway_scores.get("optical_flow",  0.5),
            pathway_scores.get("frequency",     0.5),
            pathway_scores.get("landmarks",     0.5),
        ]

        # Close the radar polygon
        angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
        angles += angles[:1]
        values_plot = values + values[:1]

        fig, ax = plt.subplots(figsize=(5, 5),
                               subplot_kw={"polar": True})
        color = "crimson" if label == "Deepfake" else "seagreen"
        ax.fill(angles, values_plot, alpha=0.25, color=color)
        ax.plot(angles, values_plot, linewidth=2, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"{label}  ({confidence:.1f}% confidence)",
            fontsize=12, pad=15,
            color=color, fontweight="bold",
        )
        plt.tight_layout()

        path = save_path or os.path.join(
            self.output_dir, f"radar_{timestamp()}.png"
        )
        ensure_dirs(os.path.dirname(path))
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Radar chart saved → {path}")
        return path

    # ── Result report ─────────────────────────────────────────────────────────

    def create_result_report(
        self,
        video_path:       str,
        classification:   str,
        confidence:       float,
        probability:      float,
        pathway_scores:   Dict[str, float],
        processing_time:  float,
        frames_processed: int,
        fps:              float,
        gradcam_path:     Optional[str] = None,
        temporal_path:    Optional[str] = None,
        radar_path:       Optional[str] = None,
        output_dir:       Optional[str] = None,
    ) -> Dict:
        """Write a structured JSON report and return it as a dict."""
        report = {
            "video_path":       video_path,
            "classification":   classification,
            "confidence":       round(confidence, 2),
            "probability":      round(probability, 4),
            "pathway_scores":   {k: round(v, 4) for k, v in pathway_scores.items()},
            "processing_time_s": round(processing_time, 2),
            "frames_processed": frames_processed,
            "fps":              round(fps, 2),
            "timestamp":        timestamp(),
            "visualizations":   {},
        }
        if gradcam_path:  report["visualizations"]["gradcam"]   = gradcam_path
        if temporal_path: report["visualizations"]["temporal"]  = temporal_path
        if radar_path:    report["visualizations"]["radar"]     = radar_path

        out_dir  = output_dir or "results/predictions"
        basename = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(out_dir, f"{basename}_{timestamp()}.json")
        save_json(report, json_path)
        logger.info(f"Report saved → {json_path}")
        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_array(fig) -> np.ndarray:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    @staticmethod
    def _save_image(img_rgb: np.ndarray, path: str) -> None:
        ensure_dirs(os.path.dirname(path))
        cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        logger.debug(f"Image saved → {path}")