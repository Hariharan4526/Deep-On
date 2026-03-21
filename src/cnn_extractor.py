"""MOD-03: CNN Feature Extractor.

EfficientNet-B4 backbone with Squeeze-Excitation attention for
deepfake probability scoring and Grad-CAM support.

When torch is not installed, a GeometricScorer fallback is used that
estimates a probability from frequency + landmark features without any
neural network — useful for testing the rest of the pipeline.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import setup_logger, get_device

logger = setup_logger(__name__)

# ── Optional torch imports ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    logger.warning("torch not found — CNNExtractor will use GeometricScorer fallback.")


# ════════════════════════════════════════════════════════════════════════════════
#  Sub-modules (only defined when torch is available)
# ════════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:

    class SEBlock(nn.Module):
        """Channel Squeeze-and-Excitation block."""

        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            mid = max(1, channels // reduction)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc   = nn.Sequential(
                nn.Linear(channels, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.shape
            w = self.pool(x).view(b, c)
            w = self.fc(w).view(b, c, 1, 1)
            return x * w

    class _DeepfakeHead(nn.Module):
        def __init__(self, in_features: int, hidden: int = 512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x)


# ════════════════════════════════════════════════════════════════════════════════
#  Fallback scorer (no torch needed)
# ════════════════════════════════════════════════════════════════════════════════

class _GeometricScorer:
    """Lightweight CPU scorer used when torch is not available.

    Estimates a deepfake probability from simple pixel statistics
    (texture variance, colour spread) that proxy for CNN texture features.
    This is NOT a substitute for a real CNN — accuracy is much lower —
    but it keeps the rest of the pipeline functional for integration tests.
    """

    def __init__(self):
        logger.info("GeometricScorer active (torch not available).")

    def score(self, face_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (feature_vector, probability)."""
        import cv2
        img = self._to_u8(face_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Local binary pattern variance proxy — deepfakes are often smoother
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        detail = gray - blur
        var    = float(detail.var())

        # Colour channel correlation (deepfakes can have unnatural correlations)
        r, g, b = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
        rg_corr = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])

        # Laplacian sharpness
        lap    = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        sharp  = float(lap.var())

        # Heuristic: real faces have higher variance and sharpness
        # These thresholds are rough empirical values
        var_score   = float(np.clip(1.0 - var / (var + 500.0),  0, 1))
        sharp_score = float(np.clip(1.0 - sharp / (sharp + 200.0), 0, 1))
        corr_score  = float(np.clip((rg_corr - 0.95) / 0.05,    0, 1))  # high corr → fake

        prob = float(np.clip(0.4 * var_score + 0.3 * sharp_score + 0.3 * corr_score, 0, 1))

        # Return a 16-D feature vector (arbitrary but consistent)
        features = np.array([
            var, sharp, rg_corr, var_score, sharp_score, corr_score,
            float(gray.mean()), float(gray.std()),
            float(r.mean()), float(g.mean()), float(b.mean()),
            float(r.std()),  float(g.std()),  float(b.std()),
            prob, 1.0 - prob,
        ], dtype=np.float32)
        return features, prob

    @staticmethod
    def _to_u8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════════
#  Main class
# ════════════════════════════════════════════════════════════════════════════════

class CNNExtractor:
    """EfficientNet-B4 feature extractor for deepfake detection.

    Falls back to GeometricScorer when torch is not installed.
    """

    SUPPORTED = {"efficientnet_b4", "resnet50", "densenet169"}

    def __init__(
        self,
        model_name:   str = "efficientnet_b4",
        pretrained:   bool = True,
        device:       Optional[str] = None,
        weights_path: Optional[str] = None,
        feature_dim:  int = 1024,
    ):
        self.model_name  = model_name
        self.feature_dim = feature_dim
        self._fallback   = None
        self._model      = None
        self.device      = None

        if not _TORCH_OK:
            self._fallback = _GeometricScorer()
            logger.warning("CNNExtractor: using GeometricScorer fallback (no torch).")
            return

        import torch
        self.device = torch.device(device) if device else get_device()

        self._backbone, backbone_dim = self._build_backbone(model_name, pretrained)
        self._se   = SEBlock(backbone_dim)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._proj = nn.Linear(backbone_dim, feature_dim)
        self._head = _DeepfakeHead(feature_dim)

        # Assemble into a single nn.Module for clean state_dict handling
        self._model = self._assemble_model()
        self._model.to(self.device).eval()

        if weights_path:
            self._load_weights(weights_path)

        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"CNNExtractor ready  model={model_name}  "
            f"params={total/1e6:.1f}M  device={self.device}"
        )

    # ── Model construction ────────────────────────────────────────────────────

    def _build_backbone(self, name: str, pretrained: bool):
        weights = "IMAGENET1K_V1" if pretrained else None
        if name == "efficientnet_b4":
            m = tv_models.efficientnet_b4(weights=weights)
            dim = m.features[-1][0].out_channels       # 1792
            return m.features, dim
        if name == "resnet50":
            m = tv_models.resnet50(weights=weights)
            return nn.Sequential(*list(m.children())[:-2]), 2048
        if name == "densenet169":
            m = tv_models.densenet169(weights=weights)
            return m.features, m.features.norm5.num_features  # 1664
        raise ValueError(f"Unsupported model '{name}'. Choose from {self.SUPPORTED}.")

    def _assemble_model(self):
        """Wrap sub-modules into a single nn.Module."""
        backbone, se, pool, proj, head = (
            self._backbone, self._se, self._pool, self._proj, self._head
        )
        feature_dim = self.feature_dim

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.se       = se
                self.pool     = pool
                self.proj     = proj
                self.head     = head

            def forward(self, x):
                fm  = self.backbone(x)
                fm  = self.se(fm)
                vec = self.pool(fm).flatten(1)
                feat = self.proj(vec)
                logit = self.head(feat)
                return feat, logit

            def forward_with_featmap(self, x):
                fm   = self.backbone(x)
                fm_s = self.se(fm)
                vec  = self.pool(fm_s).flatten(1)
                feat  = self.proj(vec)
                logit = self.head(feat)
                return feat, logit, fm   # raw (pre-SE) for Grad-CAM

        return _Net()

    def _load_weights(self, path: str) -> None:
        import torch, os
        if not os.path.exists(path):
            logger.warning(f"Weights file not found: {path}")
            return
        try:
            state = torch.load(path, map_location=self.device)
            self._model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {path}")
        except Exception as e:
            logger.warning(f"Could not load weights from {path}: {e}")

    # ── Public inference API ──────────────────────────────────────────────────

    def extract_features(
        self, face_image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Extract feature vector + deepfake probability from one face crop.

        Args:
            face_image: (256, 256, 3) float32 ImageNet-normalised array.

        Returns:
            (feature_vector ndarray shape (feature_dim,), p_cnn float [0,1])
        """
        if self._fallback:
            return self._fallback.score(face_image)

        import torch
        from .utils import image_to_tensor
        with torch.no_grad():
            t = image_to_tensor(face_image).to(self.device)
            feat, logit = self._model(t)
            prob = torch.sigmoid(logit).item()
        return feat.squeeze(0).cpu().numpy(), float(prob)

    def batch_extract_features(
        self, face_images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch feature extraction.

        Returns:
            features: (N, feature_dim) ndarray
            probs:    (N,) ndarray
        """
        if self._fallback:
            results = [self._fallback.score(img) for img in face_images]
            feats   = np.stack([r[0] for r in results])
            probs   = np.array([r[1] for r in results])
            return feats, probs

        import torch
        from .utils import image_to_tensor
        with torch.no_grad():
            batch = torch.stack(
                [image_to_tensor(img).squeeze(0) for img in face_images]
            ).to(self.device)
            feats, logits = self._model(batch)
            probs = torch.sigmoid(logits).squeeze(1)
        return feats.cpu().numpy(), probs.cpu().numpy()

    def score_image(self, face_image: np.ndarray) -> float:
        """Return p_cnn for a single face crop."""
        _, prob = self.extract_features(face_image)
        return prob

    def forward_with_attention(
        self, face_image: np.ndarray
    ) -> Tuple[np.ndarray, float, Optional[object]]:
        """Forward pass returning (features, prob, feature_map_tensor).

        feature_map_tensor is needed for Grad-CAM. Returns None when using
        the geometric fallback.
        """
        if self._fallback:
            feat, prob = self._fallback.score(face_image)
            return feat, prob, None

        import torch
        from .utils import image_to_tensor
        t = image_to_tensor(face_image).to(self.device).requires_grad_(True)
        feat, logit, fm = self._model.forward_with_featmap(t)
        prob = float(torch.sigmoid(logit).item())
        return feat.squeeze(0).detach().cpu().numpy(), prob, fm

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def fine_tune(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        lr: float = 1e-4,
        save_path: Optional[str] = None,
    ) -> List[Dict]:
        """Fine-tune on a binary deepfake classification task.

        Args:
            train_loader: DataLoader yielding (face_tensor, label) batches.
            val_loader:   Optional validation DataLoader.
            epochs:       Number of training epochs.
            lr:           Learning rate.
            save_path:    Where to save the best checkpoint.

        Returns:
            List of per-epoch metric dicts.
        """
        if not _TORCH_OK or self._model is None:
            raise RuntimeError("torch required for fine-tuning.")

        import torch
        from sklearn.metrics import roc_auc_score

        self._model.train()
        opt   = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        history  = []

        for epoch in range(1, epochs + 1):
            # ── Train ──
            self._model.train()
            train_loss = 0.0
            for faces, labels in train_loader:
                faces  = faces.to(self.device)
                labels = labels.float().to(self.device)
                opt.zero_grad()
                _, logits = self._model(faces)
                loss = crit(logits.squeeze(1), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item()
            sched.step()
            avg_loss = train_loss / max(1, len(train_loader))

            # ── Validate ──
            val_auc = float("nan")
            if val_loader is not None:
                self._model.eval()
                all_probs, all_labels = [], []
                with torch.no_grad():
                    for faces, labels in val_loader:
                        _, logits = self._model(faces.to(self.device))
                        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                        all_probs.extend(probs.tolist())
                        all_labels.extend(labels.tolist())
                if len(set(all_labels)) > 1:
                    val_auc = roc_auc_score(all_labels, all_probs)

                if val_auc > best_auc and save_path:
                    best_auc = val_auc
                    import torch, os
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    torch.save(self._model.state_dict(), save_path)
                    logger.info(f"  Saved best model (AUC={val_auc:.4f}) → {save_path}")

            metrics = {"epoch": epoch, "train_loss": round(avg_loss, 4),
                       "val_auc": round(val_auc, 4) if val_auc == val_auc else None}
            history.append(metrics)
            logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

        self._model.eval()
        return history

    @property
    def using_fallback(self) -> bool:
        return self._fallback is not None