#!/usr/bin/env python3
"""
main.py — Deepfake Detection System CLI

Commands
--------
  detect    Analyse one video or a directory of videos
  evaluate  Score accuracy against a labelled CSV
  extract   Extract and align face crops from videos (for training data prep)

Examples
--------
  # Single video
  python main.py detect --video path/to/video.mp4

  # Whole directory, custom output folder
  python main.py detect --dir path/to/videos/ --output results/

  # Evaluate on a labelled CSV (columns: path, label)
  python main.py evaluate --csv labels.csv

  # Prepare training data from raw videos
  python main.py extract --real-dir data/raw_videos/real \
                          --fake-dir data/raw_videos/fake \
                          --out-dir  data/datasets
"""
import argparse
import csv
import os
import sys
import json


# ════════════════════════════════════════════════════════════════════════════════
#  detect
# ════════════════════════════════════════════════════════════════════════════════

def cmd_detect(args):
    from src.inference import DeepfakeDetector
    from src.config    import load_config

    cfg      = load_config(args.config) if os.path.exists(args.config) else None
    detector = DeepfakeDetector(config=cfg)

    videos = _collect_videos(args)
    if not videos:
        print("No video files found.", file=sys.stderr)
        sys.exit(1)

    results = detector.batch_process_videos(videos, output_dir=args.output)

    _print_table(results)
    detector.close()


def _collect_videos(args) -> list:
    exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    if hasattr(args, "video") and args.video:
        return [args.video]
    if hasattr(args, "dir") and args.dir:
        return [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
    return []


def _print_table(results: list):
    print()
    print(f"{'VIDEO':<44} {'RESULT':<12} {'CONF%':>6}  {'PROB':>6}")
    print("─" * 74)
    for r in results:
        name   = os.path.basename(r.get("video_path", "?"))[:42]
        label  = r.get("classification", r.get("error", "ERROR"))
        conf   = f"{r['confidence']:.1f}"   if "confidence"   in r else "—"
        prob   = f"{r['probability']:.4f}"  if "probability"  in r else "—"
        print(f"{name:<44} {label:<12} {conf:>6}  {prob:>6}")
    print("─" * 74)
    ok     = [r for r in results if "error" not in r]
    fakes  = sum(1 for r in ok if r.get("classification") == "Deepfake")
    print(f"  {len(ok)}/{len(results)} processed  |  {fakes} deepfake(s)  |  "
          f"{len(ok)-fakes} authentic")
    print()


# ════════════════════════════════════════════════════════════════════════════════
#  evaluate
# ════════════════════════════════════════════════════════════════════════════════

def cmd_evaluate(args):
    from src.inference import DeepfakeDetector
    from src.utils     import compute_metrics, save_json
    from src.config    import load_config

    cfg      = load_config(args.config) if os.path.exists(args.config) else None
    detector = DeepfakeDetector(config=cfg)

    rows = _load_csv(args.csv)
    if not rows:
        print("Empty CSV.", file=sys.stderr)
        sys.exit(1)

    y_true, y_pred, y_prob = [], [], []
    for i, row in enumerate(rows, 1):
        vpath = row["path"]
        gt    = int(row["label"])
        print(f"[{i}/{len(rows)}] {os.path.basename(vpath)}", end=" … ")
        try:
            r    = detector.process_video(vpath,
                                          save_gradcam=False,
                                          save_temporal=False,
                                          save_radar=False)
            prob = r["probability"]
            pred = 1 if r["classification"] == "Deepfake" else 0
            y_true.append(gt); y_pred.append(pred); y_prob.append(prob)
            print(f"{'Deepfake' if pred else 'Authentic'}  ({r['confidence']:.1f}%)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not y_true:
        print("No videos processed.")
        return

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print()
    print("═" * 40)
    print("  EVALUATION RESULTS")
    print("═" * 40)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("═" * 40)

    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, "eval_metrics.json")
    save_json(metrics, out)
    print(f"\n  Saved to {out}")
    detector.close()


def _load_csv(path: str) -> list:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ════════════════════════════════════════════════════════════════════════════════
#  extract  (data preparation helper)
# ════════════════════════════════════════════════════════════════════════════════

def cmd_extract(args):
    """Extract aligned face crops from raw videos into image folders."""
    import cv2
    from src.video_handler  import VideoHandler
    from src.face_detection import FaceDetector
    from src.utils          import ensure_dirs, denormalize_image

    fd = FaceDetector(confidence_threshold=0.75)

    for label_name, src_dir in [("real", args.real_dir), ("fake", args.fake_dir)]:
        if not src_dir or not os.path.isdir(src_dir):
            print(f"  Skipping {label_name} (dir not found: {src_dir})")
            continue

        out_dir = os.path.join(args.out_dir, label_name)
        ensure_dirs(out_dir)

        videos = [
            os.path.join(src_dir, f)
            for f in sorted(os.listdir(src_dir))
            if os.path.splitext(f)[1].lower() in {".mp4",".avi",".mov",".webm",".mkv"}
        ]
        print(f"Extracting [{label_name}]  {len(videos)} videos → {out_dir}")

        total_saved = 0
        for vpath in videos:
            base = os.path.splitext(os.path.basename(vpath))[0]
            try:
                with VideoHandler(vpath, frame_skip=args.frame_skip) as vh:
                    for frame, fidx in vh.extract_frames():
                        crops = fd.extract_face_crops(frame)
                        for ci, crop_f32 in enumerate(crops):
                            crop_u8  = denormalize_image(crop_f32)
                            out_path = os.path.join(out_dir,
                                                    f"{base}_f{fidx:05d}_c{ci}.jpg")
                            cv2.imwrite(out_path,
                                        cv2.cvtColor(crop_u8, cv2.COLOR_RGB2BGR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 95])
                            total_saved += 1
            except Exception as e:
                print(f"  Warning: {vpath}: {e}")

        print(f"  → {total_saved} face crops saved.")

    fd.close()
    print("Extraction complete.")


# ════════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ════════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deepfake",
        description="Multi-Modal Deepfake Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (default: config.yaml)")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # ── detect ────────────────────────────────────────────────────────────────
    d = sub.add_parser("detect", help="Detect deepfakes in video(s)",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    grp = d.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", metavar="FILE",  help="Single video file")
    grp.add_argument("--dir",   metavar="DIR",   help="Directory of video files")
    d.add_argument("--output",  default="results/predictions", metavar="DIR")
    d.add_argument("--no-gradcam",  dest="gradcam",  action="store_false", default=True)
    d.add_argument("--no-temporal", dest="temporal", action="store_false", default=True)
    d.add_argument("--no-radar",    dest="radar",    action="store_false", default=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    e = sub.add_parser("evaluate", help="Evaluate on a labelled CSV")
    e.add_argument("--csv",    required=True, metavar="FILE",
                   help="CSV with columns: path, label  (1=deepfake, 0=authentic)")
    e.add_argument("--output", default="results/metrics", metavar="DIR")

    # ── extract ───────────────────────────────────────────────────────────────
    x = sub.add_parser("extract",
                       help="Extract aligned face crops from raw videos")
    x.add_argument("--real-dir",    metavar="DIR",  help="Directory of real videos")
    x.add_argument("--fake-dir",    metavar="DIR",  help="Directory of deepfake videos")
    x.add_argument("--out-dir",     metavar="DIR",  default="data/datasets")
    x.add_argument("--frame-skip",  type=int, default=5)

    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "extract":
        cmd_extract(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()