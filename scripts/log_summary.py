#!/usr/bin/env python
"""CLI: summarize one NoraIVF experiment run.

Usage:
    python scripts/log_summary.py --run-dir results/logs/<run_id>
    python scripts/log_summary.py --latest [--root results/logs] [--out events.summary.json]

Prints a human-readable summary; with --out also writes the JSON summary
into the run directory (events.summary.json).
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from noraivf.logging import format_summary, summarize_run, write_summary  # noqa: E402


def _latest_run_dir(root: Path) -> Path:
    runs = sorted(
        (d for d in root.iterdir() if d.is_dir() and (d / "run.jsonl").exists()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise SystemExit(f"no runs found under {root}")
    return runs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a NoraIVF experiment run")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="path to a run directory")
    src.add_argument("--latest", action="store_true", help="use most recent run")
    parser.add_argument("--root", type=Path, default=Path("results/logs"),
                        help="logs root for --latest (default: results/logs)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write JSON summary to this path (also written to run dir)")
    parser.add_argument("--json", action="store_true", help="print raw JSON summary")
    args = parser.parse_args()

    run_dir = _latest_run_dir(args.root) if args.latest else args.run_dir
    if not (run_dir / "run.jsonl").exists():
        raise SystemExit(f"not a run directory (no run.jsonl): {run_dir}")

    summary = summarize_run(run_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_summary(run_dir, summary)  # also drop events.summary.json into the run dir

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
