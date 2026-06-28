"""
Run the Figure 10 insert/delete microbenchmark for three maintenance paths.

The experiment measures per-operation latency on five 30k Euclidean workloads:
Frozen, LIRE, and Nora.  Results are saved as JSON/CSV and plotted as
dataset-mean P99 curves with min--max bands.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from noraivf.datasets import load_ann_benchmark_hdf5, load_npy_vector_dataset_l2
from experiments.run_claim_evidence_annbench import L2FrozenIVF, L2NovaIVF, L2SPFreshIVF


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

DATASETS = [
    ("mnist-784-euclidean", Path("/Users/bruce/Downloads/mnist-784-euclidean.hdf5"), "hdf5"),
    ("fashion-mnist-784-euclidean", Path("/Users/bruce/Downloads/fashion-mnist-784-euclidean.hdf5"), "hdf5"),
    ("core50", ROOT / "data" / "core50", "npy"),
    ("ScanNetCLIP", ROOT / "data" / "ScanNetCLIP", "npy"),
    ("Ego4D", ROOT / "data" / "Ego4D", "npy"),
]

METHODS = [
    ("Frozen", L2FrozenIVF, {}),
    ("LIRE", L2SPFreshIVF, {"do_reassign": True, "L": 5}),
    ("Nora", L2NovaIVF, {"use_scdkm": True}),
]

COLORS = {
    "Frozen": "#7f7f7f",
    "LIRE": "#1f77b4",
    "Nora": "#d62728",
}

MARKERS = {
    "Frozen": "s",
    "LIRE": "o",
    "Nora": "^",
}


def load_dataset(name, path, kind, max_train):
    if kind == "hdf5":
        train, queries, _ = load_ann_benchmark_hdf5(path, max_train=max_train, max_queries=100)
    else:
        train, queries, _ = load_npy_vector_dataset_l2(path, max_train=max_train, max_queries=100)
    return np.asarray(train, dtype=np.float32), np.asarray(queries, dtype=np.float32)


def run_microbenchmark_p99(index_cls, kwargs, data, queries, nlist, sizes, repeats=100, nprobe=8, k=100):
    out = {
        "size": [],
        "query_p99_ms": [],
        "insert_us": [],
        "delete_us": [],
        "insert_p99_us": [],
        "delete_p99_us": [],
    }
    for size in sizes:
        init = data[: min(size, len(data))]
        idx = index_cls(init, nlist, **kwargs)

        query_times = []
        for r in range(repeats):
            query = queries[r % len(queries)]
            t0 = time.perf_counter()
            idx.search(query, k=k, nprobe=nprobe)
            query_times.append((time.perf_counter() - t0) * 1000)

        insert_times = []
        for r in range(repeats):
            vec = data[(len(init) + r) % len(data)]
            t0 = time.perf_counter()
            idx.insert(vec)
            insert_times.append((time.perf_counter() - t0) * 1e6)

        delete_times = []
        ids = list(idx.id_to_vec.keys())[-repeats:]
        for vid in ids:
            t0 = time.perf_counter()
            idx.delete(vid)
            delete_times.append((time.perf_counter() - t0) * 1e6)

        out["size"].append(int(size))
        out["query_p99_ms"].append(float(np.percentile(query_times, 99)))
        out["insert_us"].append(float(np.mean(insert_times)))
        out["delete_us"].append(float(np.mean(delete_times)))
        out["insert_p99_us"].append(float(np.percentile(insert_times, 99)))
        out["delete_p99_us"].append(float(np.percentile(delete_times, 99)))
    return out


def run_all(max_train=30000, nlist=128, repeats=100, sizes=(5000, 10000, 20000, 30000)):
    results = {
        "config": {
            "max_train": max_train,
            "nlist": nlist,
            "repeats": repeats,
            "sizes": list(sizes),
            "methods": [name for name, _, _ in METHODS],
        },
        "datasets": {},
    }

    for dataset_name, path, kind in DATASETS:
        print(f"[{dataset_name}] loading {path}", flush=True)
        data, queries = load_dataset(dataset_name, path, kind, max_train=max_train)
        results["datasets"][dataset_name] = {}
        for method_name, index_cls, kwargs in METHODS:
            print(f"[{dataset_name}] {method_name}: microbenchmark", flush=True)
            results["datasets"][dataset_name][method_name] = run_microbenchmark_p99(
                index_cls,
                dict(kwargs),
                data,
                queries,
                nlist=nlist,
                sizes=list(sizes),
                repeats=repeats,
            )
    return results


def flatten_to_rows(results):
    rows = []
    for dataset, methods in results["datasets"].items():
        for method, values in methods.items():
            for i, size in enumerate(values["size"]):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "size": int(size),
                        "query_p99_ms": values["query_p99_ms"][i],
                        "insert_us": values["insert_us"][i],
                        "delete_us": values["delete_us"][i],
                        "insert_p99_us": values["insert_p99_us"][i],
                        "delete_p99_us": values["delete_p99_us"][i],
                    }
                )
    return rows


def write_outputs(results):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "microbenchmark_four_methods_30k.json"
    csv_path = RESULT_DIR / "microbenchmark_four_methods_30k.csv"

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    rows = flatten_to_rows(results)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "size",
                "query_p99_ms",
                "insert_us",
                "delete_us",
                "insert_p99_us",
                "delete_p99_us",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def aggregate(results, metric):
    sizes = results["config"]["sizes"]
    out = {}
    for method in results["config"]["methods"]:
        values = []
        for dataset in results["datasets"].values():
            values.append(dataset[method][metric])
        arr = np.asarray(values, dtype=np.float64)
        out[method] = {
            "sizes": np.asarray(sizes),
            "mean": arr.mean(axis=0),
            "min": arr.min(axis=0),
            "max": arr.max(axis=0),
        }
    return out


def plot(results):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15), sharex=True)
    for ax, metric, title in [
        (axes[0], "insert_us", "Insert latency"),
        (axes[1], "delete_us", "Delete latency"),
    ]:
        stats = aggregate(results, metric)
        for method in results["config"]["methods"]:
            vals = stats[method]
            ax.plot(
                vals["sizes"] / 1000.0,
                vals["mean"],
                label=method,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.5,
                markersize=4.0,
            )
            ax.fill_between(
                vals["sizes"] / 1000.0,
                vals["min"],
                vals["max"],
                color=COLORS[method],
                alpha=0.11,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Index size (K vectors)")
        ax.set_ylabel("Latency (us)")
        ax.set_yscale("log")
        ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.94])

    pdf_path = FIGURE_DIR / "exp4_microbench.pdf"
    png_path = FIGURE_DIR / "exp4_microbench.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf_path, png_path


def main():
    results = run_all()
    json_path, csv_path = write_outputs(results)
    pdf_path, png_path = plot(results)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
