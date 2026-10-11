"""Aggregate a run's JSONL logs into a human/machine readable summary.

Output of ``summarize_run``:
- per-phase duration stats (paired phase_start/phase_end events)
- per-metric aggregates (mean/min/max/std)
- error / warning inventory with stack traces
- bottleneck ranking (phases by total wall time)
- meta echo: config, environment, data provenance
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_ts(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def _aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    if len(values) == 1:
        return {"count": 1, "value": values[0]}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _metric_values(events: list[dict]) -> dict[str, list[float]]:
    """Collect numeric metric values by metric name across events."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        metrics = ev.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                buckets[name].append(float(value))
    return buckets


def _phase_durations(events: list[dict]) -> dict[str, list[float]]:
    """Pair phase_start/phase_end events (in order) per phase, return durations in s."""
    open_events: dict[str, list[float]] = defaultdict(list)
    durations: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        phase = ev.get("phase")
        if not phase:
            continue
        ts = _parse_ts(ev.get("ts", ""))
        if ev["event_type"] == "phase_start":
            open_events[phase].append(ts)
        elif ev["event_type"] == "phase_end":
            starts = open_events[phase]
            if starts:
                durations[phase].append(ts - starts.pop(0))
    return durations


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize one run directory. Raises FileNotFoundError if not a run dir."""
    d = Path(run_dir)
    meta_path = d / "meta.json"
    run_path = d / "run.jsonl"
    errors_path = d / "errors.jsonl"

    meta: dict = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

    events: list[dict] = []
    error_events: list[dict] = []
    if run_path.exists():
        with run_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                events.append(ev)
                if ev.get("event_type") == "error":
                    error_events.append(ev)

    if errors_path.exists():
        with errors_path.open("r", encoding="utf-8") as fh:
            external_errors = [json.loads(l) for l in fh if l.strip()]

    type_counts = Counter(ev["event_type"] for ev in events)
    phase_durations = _phase_durations(events)
    phase_stats = {
        phase: _aggregate(durs) for phase, durs in sorted(phase_durations.items())
    }
    metric_buckets = _metric_values(events)
    metric_stats = {
        name: _aggregate(values) for name, values in sorted(metric_buckets.items())
    }

    phases_by_total = sorted(
        ((p, s.get("mean", 0) * s.get("count", 0)) for p, s in phase_stats.items()),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )

    return {
        "run_id": meta.get("run_id", d.name),
        "experiment_id": meta.get("experiment_id"),
        "meta": {
            "started_at": meta.get("started_at"),
            "ended_at": meta.get("ended_at"),
            "config": meta.get("config"),
            "data_provenance": meta.get("data_provenance", []),
            "env": {
                "hostname": meta.get("env", {}).get("hostname"),
                "cpu_cores": meta.get("env", {}).get("cpu_cores"),
                "ram_bytes": meta.get("env", {}).get("ram_bytes"),
                "gpu": meta.get("env", {}).get("gpu"),
                "git": meta.get("env", {}).get("git"),
                "python": meta.get("env", {}).get("python"),
            },
        },
        "counts": {
            "total_events": len(events),
            "by_type": dict(type_counts),
            "errors_jsonl_lines": len(external_errors) if errors_path.exists() else 0,
        },
        "phases": phase_stats,
        "metrics": metric_stats,
        "errors": error_events,
        "warnings": [ev for ev in events if ev.get("event_type") == "warn"],
        "bottleneck": [
            {"phase": phase, "total_s": total} for phase, total in phases_by_total
        ],
        "files": {
            "run_jsonl": str(run_path),
            "meta_json": str(meta_path),
            "errors_jsonl": str(errors_path),
        },
    }


def write_summary(run_dir: str | Path, summary: dict[str, Any] | None = None) -> Path:
    """Write events.summary.json into the run dir; returns its path."""
    d = Path(run_dir)
    summary = summary or summarize_run(d)
    out = d / "events.summary.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return out


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable rendering of a summary dict (used by scripts/log_summary.py)."""
    lines: list[str] = []
    lines.append(f"run_id:        {summary.get('run_id')}")
    lines.append(f"experiment_id: {summary.get('experiment_id')}")
    meta = summary.get("meta", {})
    lines.append(f"started_at:    {meta.get('started_at')}")
    lines.append(f"ended_at:      {meta.get('ended_at')}")
    env = meta.get("env", {})
    git = env.get("git") or {}
    lines.append(
        f"env:           host={env.get('hostname')} cpu={env.get('cpu_cores')} "
        f"ram={env.get('ram_bytes')} py={env.get('python')} "
        f"git={git.get('commit', '-')}{' (dirty)' if git.get('dirty') else ''}"
    )
    lines.append("")
    counts = summary.get("counts", {})
    lines.append(f"events:        {counts.get('total_events')} total / {counts.get('by_type')}")
    lines.append(f"errors:        {len(summary.get('errors', []))} "
                 f"(errors.jsonl lines: {counts.get('errors_jsonl_lines')})")
    lines.append(f"warnings:      {len(summary.get('warnings', []))}")

    phases = summary.get("phases", {})
    if phases:
        lines.append("")
        lines.append("phases (s):")
        for name, stats in phases.items():
            lines.append(
                f"  {name:<20} n={stats.get('count'):<4} "
                f"mean={stats.get('mean', 0):>10.3f} "
                f"total={stats.get('mean', 0) * stats.get('count', 0):>10.3f}"
            )
    bottleneck = summary.get("bottleneck", [])
    if bottleneck:
        lines.append("")
        lines.append("bottleneck (by total phase time):")
        for item in bottleneck[:5]:
            lines.append(f"  {item['phase']:<20} {item['total_s']:>10.3f} s")

    metrics = summary.get("metrics", {})
    if metrics:
        lines.append("")
        lines.append("metrics:")
        for name, stats in metrics.items():
            if stats.get("count", 0) > 1:
                lines.append(
                    f"  {name:<24} n={stats['count']:<5} mean={stats['mean']:>12.4f} "
                    f"min={stats['min']:>12.4f} max={stats['max']:>12.4f}"
                )
            else:
                lines.append(f"  {name:<24} n=1 value={stats.get('value', stats.get('mean'))}")

    provenance = meta.get("data_provenance", [])
    if provenance:
        lines.append("")
        lines.append("data provenance:")
        for fp in provenance:
            lines.append(
                f"  {fp.get('path')}  size={fp.get('size_bytes')} "
                f"mode={fp.get('mode')} digest={fp.get('digest', '')[:16]}..."
            )

    errors = summary.get("errors", [])
    if errors:
        lines.append("")
        lines.append(f"errors ({len(errors)}):")
        for ev in errors:
            err = ev.get("error", {})
            lines.append(f"  [{ev.get('ts')}] {err.get('type')}: {err.get('message')}")
            if err.get("stack"):
                lines.append("".join(f"    {l}" for l in err["stack"].splitlines(True)))
    return "\n".join(lines)
