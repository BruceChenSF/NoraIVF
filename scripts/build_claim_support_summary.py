"""Build compact claim-support evidence from experiment JSON files.

The paper uses this summary for claim-level evidence tables. The script keeps
the source-of-truth data in JSON form and derives only simple aggregate values
that are reported in the LaTeX text.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_json(name):
    with (RESULTS / name).open() as f:
        return json.load(f)


def fixed_point(method_result, nprobe=8):
    frontier = method_result["frontier"]
    i = frontier["nprobe"].index(nprobe)
    return {
        "recall": frontier["recall"][i],
        "qps": frontier["qps"][i],
        "p99_ms": frontier["p99_ms"][i],
    }


def method_row(dataset_result, method):
    result = dataset_result["methods"][method]
    out = {
        "maintenance_time_s": result["maintenance_time_s"],
        "n_reassigns": result["n_reassigns"],
        "npa_violation_rate": result["npa"]["violation_rate"],
    }
    out.update(fixed_point(result))
    return out


def lire_row(dataset_result, name):
    result = dataset_result["lire_l_sweep"][name]
    out = {
        "maintenance_time_s": result["maintenance_time_s"],
        "n_reassigns": result["n_reassigns"],
        "npa_violation_rate": result["npa"]["violation_rate"],
    }
    out.update(fixed_point(result))
    return out


def mean(values):
    return sum(values) / len(values)


def main():
    ann = load_json("claim_evidence_annbench_30k.json")
    clip = load_json("claim_evidence_l2_clip_30k.json")
    all_data = {**ann, **clip}

    maintenance_rows = {}
    speedups = []
    for dataset, result in all_data.items():
        spf = method_row(result, "SPFresh")
        nora = method_row(result, "NovaIVF")
        speedup = spf["maintenance_time_s"] / nora["maintenance_time_s"]
        speedups.append(speedup)
        maintenance_rows[dataset] = {
            "spfresh_maintenance_s": spf["maintenance_time_s"],
            "nora_maintenance_s": nora["maintenance_time_s"],
            "speedup": speedup,
            "spfresh_reassigns": spf["n_reassigns"],
            "nora_reassigns": nora["n_reassigns"],
            "spfresh_recall": spf["recall"],
            "nora_recall": nora["recall"],
        }

    embodied = {k: clip[k] for k in ["ScanNetCLIP", "Ego4D", "core50"]}
    drift_rows = {}
    for dataset, result in embodied.items():
        frozen = method_row(result, "Frozen")
        rebuild = method_row(result, "Rebuild")
        nora = method_row(result, "NovaIVF")
        drift_rows[dataset] = {
            "frozen_recall": frozen["recall"],
            "rebuild_recall": rebuild["recall"],
            "nora_recall": nora["recall"],
            "frozen_qps": frozen["qps"],
            "nora_qps": nora["qps"],
            "frozen_p99_ms": frozen["p99_ms"],
            "nora_p99_ms": nora["p99_ms"],
        }

    split_rows = {}
    for dataset, result in all_data.items():
        l0 = lire_row(result, "SPFresh-L0")
        l5 = lire_row(result, "SPFresh-L5")
        nora = method_row(result, "NovaIVF")
        split_rows[dataset] = {
            "l0_npa": l0["npa_violation_rate"],
            "l5_npa": l5["npa_violation_rate"],
            "nora_npa": nora["npa_violation_rate"],
            "l0_recall": l0["recall"],
            "l5_recall": l5["recall"],
            "nora_recall": nora["recall"],
            "l0_maintenance_s": l0["maintenance_time_s"],
            "l5_maintenance_s": l5["maintenance_time_s"],
            "nora_maintenance_s": nora["maintenance_time_s"],
            "l5_reassigns": l5["n_reassigns"],
        }

    layout_rows = {}
    for dataset, result in all_data.items():
        mixed = result["mixed_workload"]
        spf = mixed["SPFresh"]
        nora = mixed["NovaIVF"]
        layout_rows[dataset] = {
            "spfresh_delete_p99_us": spf["delete_p99_us"],
            "nora_delete_p99_us": nora["delete_p99_us"],
            "delete_p99_speedup": spf["delete_p99_us"] / nora["delete_p99_us"],
            "spfresh_query_p99_ms": spf["query_p99_ms"],
            "nora_query_p99_ms": nora["query_p99_ms"],
            "spfresh_recall": spf["recall_mean"],
            "nora_recall": nora["recall_mean"],
        }

    output = {
        "source_files": [
            "results/claim_evidence_annbench_30k.json",
            "results/claim_evidence_l2_clip_30k.json",
        ],
        "fixed_point": "nprobe=8, Recall@100 unless otherwise noted",
        "maintenance_no_reassignment": {
            "rows": maintenance_rows,
            "mean_speedup": mean(speedups),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "total_spfresh_reassigns": sum(r["spfresh_reassigns"] for r in maintenance_rows.values()),
            "total_nora_reassigns": sum(r["nora_reassigns"] for r in maintenance_rows.values()),
        },
        "embodied_drift_pressure": {
            "rows": drift_rows,
            "mean_frozen_recall": mean([r["frozen_recall"] for r in drift_rows.values()]),
            "mean_rebuild_recall": mean([r["rebuild_recall"] for r in drift_rows.values()]),
            "mean_nora_recall": mean([r["nora_recall"] for r in drift_rows.values()]),
            "mean_frozen_p99_ms": mean([r["frozen_p99_ms"] for r in drift_rows.values()]),
            "mean_nora_p99_ms": mean([r["nora_p99_ms"] for r in drift_rows.values()]),
        },
        "split_repair_tradeoff": {
            "rows": split_rows,
            "mean_l0_npa": mean([r["l0_npa"] for r in split_rows.values()]),
            "mean_l5_npa": mean([r["l5_npa"] for r in split_rows.values()]),
            "mean_nora_npa": mean([r["nora_npa"] for r in split_rows.values()]),
            "mean_l5_reassigns": mean([r["l5_reassigns"] for r in split_rows.values()]),
        },
        "layout_update_path": {
            "rows": layout_rows,
            "mean_delete_p99_speedup": mean([r["delete_p99_speedup"] for r in layout_rows.values()]),
            "min_delete_p99_speedup": min(r["delete_p99_speedup"] for r in layout_rows.values()),
            "max_delete_p99_speedup": max(r["delete_p99_speedup"] for r in layout_rows.values()),
        },
    }

    out_path = RESULTS / "claim_support_summary.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(out_path)


if __name__ == "__main__":
    main()
