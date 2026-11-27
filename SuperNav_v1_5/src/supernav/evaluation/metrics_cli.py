#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from supernav.evaluation.metrics import compute_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute metrics for one canonical harness stream.")
    parser.add_argument("canonical_jsonl")
    parser.add_argument("--arm", default="?")
    parser.add_argument("--slug", default="?")
    parser.add_argument("--rep", default=None)
    args = parser.parse_args()
    print(json.dumps(compute_metrics(args.canonical_jsonl, arm=args.arm, slug=args.slug, rep=args.rep), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
