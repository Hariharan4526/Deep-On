from __future__ import annotations

import argparse
import json
from pathlib import Path

from detect import DeepfakeDetector, DetectorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deepfake & AI content detection prototype"
    )
    parser.add_argument("input", help="Path to media file (image/video/audio)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON output",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save JSON report",
    )
    parser.add_argument(
        "--video-samples",
        type=int,
        default=12,
        help="Maximum number of frames to sample for video",
    )
    parser.add_argument(
        "--ai-threshold",
        type=float,
        default=0.55,
        help="Threshold for labeling media as likely AI-generated",
    )
    return parser


def format_console(result_dict: dict) -> str:
    lines = [
        "=== Deepfake & AI Content Detection Report ===",
        f"Input: {result_dict['input_path']}",
        f"Media Type: {result_dict['media_type']}",
        f"AI-Generated Confidence: {result_dict['ai_generated_confidence']:.4f}",
        f"Authenticity Confidence: {result_dict['authenticity_confidence']:.4f}",
        f"Predicted Source: {result_dict['predicted_source']}",
        "",
        "Evidence:",
    ]

    for evidence in result_dict["evidences"]:
        lines.append(
            f"- {evidence['name']}: score={evidence['score']:.4f} details={evidence['details']}"
        )

    lines.append("")
    lines.append("Notes:")
    for note in result_dict["notes"]:
        lines.append(f"- {note}")

    lines.append("")
    lines.append(f"Metadata: {result_dict['metadata']}")
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        return 1

    config = DetectorConfig(
        video_sample_limit=max(args.video_samples, 1),
        ai_threshold=min(max(args.ai_threshold, 0.0), 1.0),
    )
    detector = DeepfakeDetector(config=config)

    try:
        result = detector.analyze(str(input_path))
    except Exception as exc:
        print(f"Analysis failed: {exc}")
        return 2

    result_dict = result.to_dict()

    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print(format_console(result_dict))

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
        print(f"\nSaved report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
