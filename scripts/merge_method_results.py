#!/usr/bin/env python3
"""Merge per-method results from single-method rebuild JSON files into the
main multi-method claim-evidence JSON files.

The rebuild files produced for the "Rebuild" baseline contain a single
dataset -> {"methods": {"Rebuild": {...}}, ...}.  This script copies the
requested method record(s) into the matching target JSON (matched by dataset
name) while leaving every other method and every top-level section (e.g.
``maintenance_budget``, ``lire_l_sweep``) untouched.

Each target file is backed up to ``<target>.bak2`` before being rewritten.

Usage:
    python scripts/merge_method_results.py \
        --source results/claim_evidence_glove_rebuild.json \
                 results/claim_evidence_audioset_rebuild.json \
        --target results/claim_evidence_glove.json \
                 results/claim_evidence_audioset.json \
        --method Rebuild
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def merge(source_path: Path, targets: dict[str, Path], method: str | None,
          dry_run: bool) -> list[str]:
    """Merge methods from one source file into its matching target."""
    source = load(source_path)
    messages: list[str] = []
    for ds_name, ds_result in source.items():
        if ds_name not in targets:
            raise SystemExit(
                f"no --target given for dataset {ds_name!r} (from {source_path})"
            )
        target_path = targets[ds_name]
        target = load(target_path)
        if ds_name not in target:
            raise SystemExit(
                f"dataset {ds_name!r} missing from target {target_path}"
            )
        src_methods = ds_result.get("methods", {})
        if method is not None:
            if method not in src_methods:
                raise SystemExit(
                    f"method {method!r} not found in {source_path} ({ds_name})"
                )
            to_merge = {method: src_methods[method]}
        else:
            to_merge = src_methods

        target_methods = target[ds_name].setdefault("methods", {})
        for name, record in to_merge.items():
            target_methods[name] = record
            messages.append(f"{ds_name}: {name}")

        if dry_run:
            continue
        backup = target_path.with_suffix(target_path.suffix + ".bak2")
        shutil.copy2(target_path, backup)
        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(target, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        messages.append(f"  backup -> {backup}")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True,
                        help="single-method rebuild JSON file(s)")
    parser.add_argument("--target", type=Path, nargs="+", required=True,
                        help="main multi-method JSON file(s) to update")
    parser.add_argument("--method", default="Rebuild",
                        help="method name to merge (default: Rebuild); "
                             "pass '' to merge every method in the source")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be merged without writing")
    args = parser.parse_args()

    # map dataset name -> target path
    targets: dict[str, Path] = {}
    for path in args.target:
        for ds_name in load(path):
            targets[ds_name] = path

    method = args.method or None
    for src in args.source:
        print(f"merging {src}:")
        for line in merge(src, targets, method, args.dry_run):
            print(f"  {line}")


if __name__ == "__main__":
    main()
