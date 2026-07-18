#!/usr/bin/env python3
"""Evaluate generated TTS audio with WER, CER, similarity and audio metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_studio.evaluation import (  # noqa: E402
    SpeakerEncoder,
    WhisperTranscriber,
    evaluate_cases,
    load_manifest,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá TTS bằng WER, CER và speaker similarity")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON manifest chứa các test case")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument(
        "--ignore-case-punctuation",
        action="store_true",
        help="Đánh giá bổ sung theo chính sách không phân biệt hoa/thường và dấu câu",
    )
    parser.add_argument("--skip-speaker-similarity", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    try:
        cases = load_manifest(args.manifest.resolve())
    except ValueError as exc:
        print(f"Manifest không hợp lệ: {exc}", file=sys.stderr)
        return 2
    transcriber = WhisperTranscriber(args.whisper_model, device)
    encoder = None if args.skip_speaker_similarity else SpeakerEncoder(device)
    results = evaluate_cases(cases, transcriber, encoder, relaxed=args.ignore_case_punctuation)
    write_reports(results, args.output_dir, args.manifest)
    successes = sum(row["status"] == "ok" for row in results)
    print(f"Đã đánh giá {successes}/{len(results)} mẫu. Report: {args.output_dir / 'report.md'}")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
