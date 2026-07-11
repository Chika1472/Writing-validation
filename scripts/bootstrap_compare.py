from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_jsonl
from src.evaluation.bootstrap import paired_stratified_bootstrap
from src.evaluation.predictions import read_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired prompt-stratified bootstrap comparison.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold = load_jsonl(Path(args.gold).resolve())
    candidate = read_predictions(Path(args.candidate).resolve())
    baseline = read_predictions(Path(args.baseline).resolve())
    report = paired_stratified_bootstrap(
        gold,
        candidate,
        baseline,
        n_resamples=args.n_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

