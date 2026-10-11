#!/usr/bin/env python3
"""Plot claim-evidence results (QPS-recall frontiers, LIRE L-sweep, summary CSV).

Reads claim-evidence JSON files (output of
experiments/run_claim_evidence_annbench.py).  ``--input`` accepts globs and/or
multiple files; results from all files are merged by dataset name (later files
override earlier ones), so one invocation can plot the full 10-dataset table.

Outputs:
  * figures/<dataset>_qps_recall.pdf / .png    — one QPS-recall frontier per
    dataset (one curve per method, fixed color+marker; AdaIVF is crimson
    dashed).  A per-dataset caveat (e.g. bounded-window / coverage) can be
    added as a second title line via ``--note DATASET=TEXT``.
  * figures/<dataset>_lire_l_sweep.pdf / .png  — 4-panel L-sweep per dataset
    (reassigns / maintenance / NPA violation rate / recall vs L, from the
    JSON ``lire_l_sweep`` section).
  * results/claim_evidence_summary_nprobe8.csv — fixed-point table:
    dataset x method at nprobe=8 (recall / QPS / p99 / maintenance time).

Style follows the existing figures/ plots: serif, ~8pt, pdf.fonttype=42,
single-column ~3.3in wide, minimal decoration.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# method styling: fixed color + marker per method across every figure.
# AdaIVF is deliberately eye-catching (crimson, dashed).
# --------------------------------------------------------------------------- #
METHOD_ORDER = ["Frozen", "Rebuild", "DeDrift", "SPFresh", "NovaIVF", "AdaIVF"]

METHOD_STYLE = {
    "Frozen": ("#d95f02", "^", ":"),
    "Rebuild": ("#e6ab02", "P", "--"),
    "DeDrift": ("#cc79a7", "v", "-."),
    "SPFresh": ("#1f77b4", "o", "-"),
    "NovaIVF": ("#009e73", "D", "-"),
    "AdaIVF": ("crimson", "*", (0, (4, 2))),
}

_FALLBACK_PALETTE = ["#333333", "#888888", "#a6761d", "#7570b3", "#66a61e", "#e7298a"]

# Result JSONs still key the methods as "NovaIVF" and "AdaIVF" (historical
# names); every rendered legend/label must use the paper spellings "NoraIVF"
# and "Ada-IVF".  Look-ups keep using the JSON keys, only the displayed text
# goes through this map.
METHOD_DISPLAY = {"NovaIVF": "NoraIVF", "AdaIVF": "Ada-IVF"}


def display_name(method: str) -> str:
    """Paper spelling for a method key (JSON keys are left untouched)."""
    return METHOD_DISPLAY.get(method, method)

RC_PARAMS = {
    "font.family": "serif",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def style_for(method: str) -> tuple[str, str, str]:
    """Return (color, marker, linestyle) for a method (unknown -> palette)."""
    if method in METHOD_STYLE:
        return METHOD_STYLE[method]
    idx = hash(method) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[idx], "o", "-"


def load_results(inputs: list[str]) -> dict:
    """Merge claim-evidence JSON files (globs supported) by dataset name."""
    merged: dict = {}
    for pattern in inputs:
        if any(ch in pattern for ch in "*?["):
            paths = sorted(glob.glob(pattern))
            if not paths:
                raise FileNotFoundError(f"no files matched {pattern}")
        else:
            paths = [pattern]
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for ds_name, ds_result in data.items():
                if ds_name in merged:
                    print(f"note: {ds_name} in multiple inputs; later file overrides",
                          file=sys.stderr)
                merged[ds_name] = ds_result
    return merged


def method_result(ds_result: dict, method: str) -> dict | None:
    """Result dict for a main method, or None when missing/errored."""
    if "methods" not in ds_result:
        return None
    result = ds_result["methods"].get(method)
    if result is None or "frontier" not in result:
        return None
    return result


def present_methods(ds_result: dict) -> list[str]:
    return [m for m in METHOD_ORDER if method_result(ds_result, m) is not None]


def parse_l(method: str) -> int:
    match = re.search(r"L(\d+)$", method)
    if not match:
        raise ValueError(f"Cannot parse L from method name: {method}")
    return int(match.group(1))


def value_at_nprobe(frontier: dict, nprobe: int) -> tuple[int, dict]:
    """Index into a frontier at ``nprobe`` (nearest when absent)."""
    nprobes = frontier["nprobe"]
    if nprobe in nprobes:
        i = nprobes.index(nprobe)
    else:
        i = min(range(len(nprobes)), key=lambda j: abs(nprobes[j] - nprobe))
    return nprobes[i], {key: frontier[key][i] for key in
                        ("recall", "qps", "p50_ms", "p99_ms")}


# --------------------------------------------------------------------------- #
# QPS-recall frontiers (one figure per dataset)
# --------------------------------------------------------------------------- #
def plot_qps_recall(dataset: str, ds_result: dict, out_dir: Path,
                    note: str | None = None) -> None:
    """One QPS-recall frontier per dataset; ``note`` (e.g. a bounded-window /
    coverage / gt-scope caveat) is drawn as a second title line."""
    methods = present_methods(ds_result)
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    handles = []
    for method in methods:
        frontier = ds_result["methods"][method]["frontier"]
        color, marker, ls = style_for(method)
        line, = ax.plot(frontier["recall"], frontier["qps"],
                        color=color, marker=marker, linestyle=ls,
                        linewidth=1.2, markersize=3.2, label=display_name(method))
        handles.append(line)
    ax.set_xlabel("Recall")
    ax.set_ylabel("QPS")
    if note:
        ax.set_title(f"{dataset}\n{note}", pad=2, fontsize=6.6)
    else:
        ax.set_title(dataset, pad=2)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.set_xlim(left=0.0, right=1.01)
    if handles:
        fig.legend(handles, [display_name(m) for m in methods], loc="lower center",
                   ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03),
                   columnspacing=0.8, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{dataset}_qps_recall.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{dataset}_qps_recall.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# LIRE L-sweep: 4-panel (reassigns / maintenance / NPA / recall vs L)
# --------------------------------------------------------------------------- #
def plot_lire_l_sweep(dataset: str, ds_result: dict, out_dir: Path,
                      nprobe: int = 8) -> None:
    sweep = ds_result.get("lire_l_sweep", {})
    sweep = {m: r for m, r in sweep.items() if "frontier" in r}
    if not sweep:
        return

    panels = [
        ("n_reassigns", "Reassignments", "Vectors"),
        ("maintenance_time_s", "Maintenance", "Time (s)"),
        ("npa_violation_rate", "NPA violation", "Rate"),
        ("recall", "Quality", "Recall@100"),
    ]
    # drop panels whose data is entirely absent (e.g. NPA skipped)
    panels = [
        (key, title, ylabel) for key, title, ylabel in panels
        if any(r.get(key) is not None or key == "recall" for r in sweep.values())
    ]
    n_panels = len(panels)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3, 0.95 * n_rows + 1.05),
                             sharex=True)
    axes = [axes] if n_panels == 1 else axes.ravel()

    l_values = sorted({parse_l(m) for m in sweep})
    for ax, (key, title, ylabel) in zip(axes, panels):
        for method in sorted(sweep):
            result = sweep[method]
            if key == "recall":
                _, point = value_at_nprobe(result["frontier"], nprobe)
                ys = [point["recall"]] * len(l_values)
            else:
                raw = result.get(key)
                if raw is None:
                    continue
                ys = [raw] * len(l_values)
            color, marker, ls = style_for(method.split("-")[0])
            ax.plot(l_values, ys, color=color, marker=marker, linestyle=ls,
                    linewidth=1.1, markersize=3.0, label=method)
        ax.set_title(title, pad=1.5)
        ax.set_xlabel("L")
        ax.set_ylabel(ylabel)
        if key == "maintenance_time_s":
            ax.set_yscale("log")
        ax.grid(True, linewidth=0.28, alpha=0.3)
        ax.set_xticks(l_values)
        ax.tick_params(axis="both", which="major", pad=1.5, length=2.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, 0.99), columnspacing=0.6,
               handlelength=1.2, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.7, h_pad=0.7)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{dataset}_lire_l_sweep.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{dataset}_lire_l_sweep.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# summary CSV: dataset x method at nprobe=8 (paper fixed-point table)
# --------------------------------------------------------------------------- #
def collect_summary_rows(results: dict, nprobe: int,
                         include_lire_sweep: bool) -> list[dict]:
    rows = []
    for ds_name, ds_result in results.items():
        for method in METHOD_ORDER:
            result = method_result(ds_result, method)
            if result is None:
                continue
            rows.append(_summary_row(ds_name, method, "methods", result, nprobe))
        if include_lire_sweep:
            for method, result in ds_result.get("lire_l_sweep", {}).items():
                if "frontier" not in result:
                    continue
                rows.append(_summary_row(ds_name, method, "lire_l_sweep",
                                         result, nprobe))
    return rows


def _summary_row(dataset: str, method: str, source: str, result: dict,
                 nprobe: int) -> dict:
    used_nprobe, point = value_at_nprobe(result["frontier"], nprobe)
    return {
        "dataset": dataset,
        "method": method,
        "source": source,
        "nprobe": used_nprobe,
        "recall": round(float(point["recall"]), 6),
        "qps": round(float(point["qps"]), 3),
        "p99_ms": round(float(point["p99_ms"]), 4),
        "maintenance_time_s": round(float(result.get("maintenance_time_s", 0.0)), 3),
        "n_reassigns": int(result.get("n_reassigns", 0)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot claim-evidence results (QPS-recall, LIRE L-sweep, summary CSV)."
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="claim-evidence JSON file(s); globs supported")
    parser.add_argument("--note", action="append", default=[], metavar="DATASET=TEXT",
                        help="extra title note for one dataset (repeatable; e.g. "
                             "--note 'deep-image-96-angular=bounded window, "
                             "coverage 51%, gt-scope indexed')")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--csv", type=Path,
                        default=Path("results/claim_evidence_summary_nprobe8.csv"))
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--include-lire-sweep", action="store_true",
                        help="also append lire_l_sweep (SPFresh-L*) rows to the CSV")
    args = parser.parse_args()

    plt.rcParams.update(RC_PARAMS)
    results = load_results(args.input)
    if not results:
        raise SystemExit("no datasets found in inputs")

    notes: dict[str, str] = {}
    for item in args.note:
        if "=" not in item:
            raise SystemExit(f"--note expects DATASET=TEXT, got: {item!r}")
        ds_name, text = item.split("=", 1)
        if ds_name not in results:
            print(f"note: --note target {ds_name!r} not among datasets; ignored",
                  file=sys.stderr)
        notes[ds_name] = text

    n_frontier = 0
    n_sweep = 0
    for dataset, ds_result in results.items():
        if present_methods(ds_result):
            plot_qps_recall(dataset, ds_result, args.out_dir, notes.get(dataset))
            n_frontier += 1
        if ds_result.get("lire_l_sweep"):
            plot_lire_l_sweep(dataset, ds_result, args.out_dir, args.nprobe)
            n_sweep += 1

    rows = collect_summary_rows(results, args.nprobe, args.include_lire_sweep)
    write_csv(rows, args.csv)

    print(f"Wrote {n_frontier} qps-recall figures, {n_sweep} L-sweep figures -> {args.out_dir}")
    print(f"Wrote {len(rows)} summary rows -> {args.csv}")


if __name__ == "__main__":
    main()
