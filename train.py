"""
train.py — Fine-tune the CNN extractor on FaceForensics++ (or any
binary deepfake dataset organised as two folders: real/ and fake/).

Usage
-----
# Train from scratch on FF++ extracted frames
python train.py --real data/datasets/real --fake data/datasets/fake

# Resume training from a checkpoint
python train.py --real data/datasets/real --fake data/datasets/fake \
                --resume models/weights/best.pth --epochs 5

# Evaluate a saved checkpoint on a held-out test split
python train.py --real data/datasets/real --fake data/datasets/fake \
                --eval-only --resume models/weights/best.pth
"""
import argparse
import os
import random
import time
from typing import List, Tuple

import cv2
import numpy as np

# ── Optional torch ────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ════════════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(folder: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(root, f))
    return paths


if _TORCH_OK:

    class DeepfakeDataset(Dataset):
        """Binary classification dataset.

        Args:
            real_dir  : directory of real face images (label=0)
            fake_dir  : directory of deepfake face images (label=1)
            split     : 'train', 'val', or 'test'
            split_ratio: (train, val, test) fractions — must sum to 1
            seed      : random seed for reproducibility
            augment   : whether to apply training augmentations
        """

        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        def __init__(
            self,
            real_dir: str,
            fake_dir: str,
            split: str = "train",
            split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
            seed: int = 42,
            augment: bool = True,
        ):
            assert split in ("train", "val", "test")
            real_paths = _collect_images(real_dir)
            fake_paths = _collect_images(fake_dir)

            rng = random.Random(seed)
            rng.shuffle(real_paths)
            rng.shuffle(fake_paths)

            def _split(paths):
                n = len(paths)
                a = int(n * split_ratio[0])
                b = int(n * (split_ratio[0] + split_ratio[1]))
                return {"train": paths[:a], "val": paths[a:b], "test": paths[b:]}[split]

            real_split = _split(real_paths)
            fake_split = _split(fake_paths)

            self.samples = (
                [(p, 0) for p in real_split] +
                [(p, 1) for p in fake_split]
            )
            rng.shuffle(self.samples)
            self.augment = augment and (split == "train")

            print(f"[Dataset:{split}]  real={len(real_split)}  "
                  f"fake={len(fake_split)}  total={len(self.samples)}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = cv2.imread(path)
            if img is None:
                # Return a zero tensor on broken files
                return torch.zeros(3, 256, 256), torch.tensor(float(label))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

            if self.augment:
                img = self._augment(img)

            # ImageNet normalisation → (C, H, W) float32 tensor
            img = img.astype(np.float32) / 255.0
            mean = np.array(self.MEAN, dtype=np.float32)
            std  = np.array(self.STD,  dtype=np.float32)
            img  = (img - mean) / std
            tensor = torch.from_numpy(img.transpose(2, 0, 1))
            return tensor, torch.tensor(float(label))

        @staticmethod
        def _augment(img: np.ndarray) -> np.ndarray:
            """Random horizontal flip + colour jitter."""
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            # Brightness / contrast jitter
            alpha = random.uniform(0.85, 1.15)
            beta  = random.randint(-15, 15)
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            # Random JPEG compression to simulate FF++ artefacts
            if random.random() < 0.3:
                quality = random.randint(40, 95)
                _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                img    = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img


# ════════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ════════════════════════════════════════════════════════════════════════════════

def _evaluate(model, loader, device, criterion):
    """Compute loss, accuracy, and AUC-ROC on a DataLoader."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for faces, labels in loader:
            faces  = faces.to(device)
            labels = labels.to(device)
            _, logits = model._model(faces)
            loss   = criterion(logits.squeeze(1), labels)
            probs  = torch.sigmoid(logits).squeeze(1)
            preds  = (probs > 0.5).float()
            total_loss += loss.item() * len(labels)
            n_correct  += (preds == labels).sum().item()
            n_total    += len(labels)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(n_total, 1)
    accuracy = n_correct  / max(n_total, 1)
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, accuracy, auc


def train(args):
    if not _TORCH_OK:
        print("ERROR: torch is required for training.  Install with:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return

    from src.cnn_extractor import CNNExtractor
    from src.config import load_config

    cfg    = load_config(args.config) if os.path.exists(args.config) else None
    device = cfg.hardware.device if cfg else ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = DeepfakeDataset(args.real, args.fake, split="train", augment=True)
    val_ds   = DeepfakeDataset(args.real, args.fake, split="val",   augment=False)
    test_ds  = DeepfakeDataset(args.real, args.fake, split="test",  augment=False)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding {args.model} on {device}...")
    model = CNNExtractor(
        model_name=args.model,
        pretrained=True,
        device=device,
        weights_path=args.resume if args.resume else None,
    )

    if args.eval_only:
        print("\n── Evaluation only ──")
        loss, acc, auc = _evaluate(model, test_loader, device,
                                   nn.BCEWithLogitsLoss())
        print(f"Test  loss={loss:.4f}  acc={acc:.4f}  AUC={auc:.4f}")
        return

    # ── Training ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    opt       = torch.optim.AdamW(model._model.parameters(),
                                  lr=args.lr, weight_decay=1e-2)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc  = 0.0
    history   = []

    print(f"\nTraining for {args.epochs} epochs  lr={args.lr}  batch={args.batch_size}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model._model.train()
        train_loss = 0.0

        for step, (faces, labels) in enumerate(train_loader, 1):
            faces  = faces.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            _, logits = model._model(faces)
            loss = criterion(logits.squeeze(1), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model._model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

            if step % max(1, len(train_loader) // 5) == 0:
                print(f"  epoch {epoch}  step {step}/{len(train_loader)}"
                      f"  loss={train_loss/step:.4f}", end="\r")

        sched.step()
        avg_train = train_loss / max(len(train_loader), 1)

        val_loss, val_acc, val_auc = _evaluate(model, val_loader, device, criterion)
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_train:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_auc={val_auc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train_loss": avg_train,
                        "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model._model.state_dict(), args.save)
            print(f"  ✓ Saved best model → {args.save}  (val_auc={val_auc:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n── Final test evaluation ──")
    model._load_weights(args.save)
    test_loss, test_acc, test_auc = _evaluate(model, test_loader, device, criterion)
    print(f"Test  loss={test_loss:.4f}  acc={test_acc:.4f}  AUC={test_auc:.4f}")

    # Save history
    from src.utils import save_json
    save_json({"history": history, "test": {"loss": test_loss, "acc": test_acc, "auc": test_auc}},
              os.path.join(os.path.dirname(args.save), "train_history.json"))
    print("Training complete.")


# ════════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune CNN extractor on a deepfake dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real",       required=True,  help="Directory of real face images")
    p.add_argument("--fake",       required=True,  help="Directory of deepfake face images")
    p.add_argument("--model",      default="efficientnet_b4",
                   choices=["efficientnet_b4", "resnet50", "densenet169"])
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--save",       default="models/weights/best.pth",
                   help="Path to save best checkpoint")
    p.add_argument("--resume",     default=None, help="Path to resume from checkpoint")
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training; evaluate --resume checkpoint on test set")
    p.add_argument("--config",     default="config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())