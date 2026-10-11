#!/usr/bin/env python3
"""Recompute the maintenance_budget section of a claim-evidence results JSON.

Reads an existing claim-evidence JSON (default:
results/claim_evidence_sift1m.json), rebuilds ``maintenance_budget`` from the
``methods`` + ``lire_l_sweep`` sections using the shared helpers in
experiments/run_claim_evidence_annbench.py (no logic duplication), backs the
file up as ``<path>.bak`` and writes the updated JSON back in place.

Example::

    python scripts/recompute_maintenance_budget.py \
        --input results/claim_evidence_sift1m.json
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RUNNER = _ROOT / "experiments" / "run_claim_evidence_annbench.py"
DEFAULT_INPUT = _ROOT / "results" / "claim_evidence_sift1m.json"

SMOKE_BUDGETS_HOURS = [0.05, 0.15, 0.5]
FULL_BUDGETS_HOURS = [0.15, 0.3, 1.0, 3.0, 8.0]


def load_runner():
    """Import experiments/run_claim_evidence_annbench.py as a plain module."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location("run_claim_evidence_annbench", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detect_budgets_hours(ds):
    """Pick smoke/full budget tiers from the entry's frontier shape."""
    for src in ("methods", "lire_l_sweep"):
        for result in (ds.get(src) or {}).values():
            if isinstance(result, dict) and isinstance(result.get("frontier"), dict):
                if len(result["frontier"].get("nprobe", [])) == 5:  # smoke uses 5 nprobes
                    return SMOKE_BUDGETS_HOURS
                return FULL_BUDGETS_HOURS
    return FULL_BUDGETS_HOURS


def rebuild_for_entry(ds, runner):
    """Rebuild ``maintenance_budget`` for one dataset run entry (in place)."""
    if not isinstance(ds, dict):
        return False
    budget_candidates = {}
    for src in ("methods", "lire_l_sweep"):
        for name, result in (ds.get(src) or {}).items():
            if (isinstance(result, dict) and "frontier" in result
                    and "maintenance_time_s" in result):
                budget_candidates[name] = result
    ds["maintenance_budget"] = runner.run_maintenance_budget(
        budget_candidates, detect_budgets_hours(ds)
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="Path to claim-evidence results JSON (default: %(default)s)")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSON: {path}")
    with open(path) as f:
        data = json.load(f)

    runner = load_runner()
    rebuilt = 0
    for ds in data.values():
        if isinstance(ds, dict) and "seeds" in ds:
            for seed_ds in ds["seeds"].values():
                if isinstance(seed_ds, dict):
                    rebuilt += int(rebuild_for_entry(seed_ds, runner))
        else:
            rebuilt += int(rebuild_for_entry(ds, runner))

    shutil.copy2(path, str(path) + ".bak")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Rebuilt maintenance_budget for {rebuilt} dataset run(s) in {path}")
    print(f"Backup written to {path}.bak\n")
    for ds_name, ds in data.items():
        if isinstance(ds, dict) and "seeds" in ds:
            for seed_name, seed_ds in ds["seeds"].items():
                if isinstance(seed_ds, dict) and "maintenance_budget" in seed_ds:
                    print(json.dumps({f"{ds_name}/seed{seed_name}": seed_ds["maintenance_budget"]}, indent=2))
        elif isinstance(ds, dict) and "maintenance_budget" in ds:
            print(json.dumps({ds_name: ds["maintenance_budget"]}, indent=2))


if __name__ == "__main__":
    main()
