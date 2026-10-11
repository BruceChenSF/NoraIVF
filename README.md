# NoraIVF

Code for the paper **"NoraIVF: No-Reassignment Incremental IVF Index Maintenance
for Streaming Vector Search"** (under submission).

NoraIVF is a memory-native dynamic IVF index for streaming vector search. Its
split protocol, **SCDKM** (Streaming Coordinate Descent K-Means), reassigns
points only between the two clusters that a split produces. A first split pass
hands SCDKM dense postings that stay dense, so the post-split boundary repair
pass that LIRE-style maintenance requires is removed rather than deferred.

The repository also contains the baselines the paper compares against:
`Frozen`, `Rebuild`, `DeDrift`, `SPFresh` (LIRE repair), and `AdaIVF`.

## Repository layout

```
noraivf/                       Core Python package
  core.py                      IVF index implementations (Frozen, Rebuild, DeDrift,
                               SPFresh, NovaIVF, AdaIVF) + the SCDKM protocol
  datasets.py                  Dataset loading, generation, ground truth
  lean.py                      Memory-lean storage substrate: array-backed dict
                               views so the reference protocols fit at 1e9 scale
  lean_methods.py              Lean index classes + the candidate-set-width
                               (restricted / top-L / full-k) CDKM variants
  logging/                     Structured experiment logging
    logger.py                  ExperimentLogger: JSONL events, phases, checkpoints
    summary.py                 summarize_run / write_summary / format_summary
    env.py                     Environment and data-provenance capture

experiments/                   Experiment entry points
  run_claim_evidence_annbench.py         Main multi-method suite (L2 distance,
                                         full-coverage streaming protocol)
  run_experiments.py                     Earlier 30k-scale suite (cosine/IP)
  run_latency_scale.py                   Million-scale P99 stability and
                                         mixed-workload query/insert/delete latency
  run_microbenchmark_four_methods.py     Insert/delete microbenchmark
  run_p99_stability_5datasets.py         P99 query-latency stability table
  measure_scan_width.py                  SCDKM candidate-set-width microbenchmark
  run_cdkm_vs_lloyd_local.py             CDKM vs Lloyd local convergence diagnostic
  run_boundary_sampled_scdkm_diagnostic.py
  run_local_cdkm_repair_diagnostic.py
  fidelity_check.py                      Equivalence gate for the lean substrate
  run_billion_probe.py                   Maintenance probe at 1e9 scale (see note)

scripts/                       Data preparation, orchestration, plotting, QA
  prepare_annbench_datasets.py           ann-benchmarks HDF5 -> canonical npy layout
  prepare_audioset.py                    AudioSet TFRecord embeddings -> npy
  probe_audioset_tfrecord.py             AudioSet container inspection
  prepare_sift1b.py                      bigann .bin -> npy (lossless float16)
  prepare_real_vector_datasets.py        Download + encode real image datasets
  prepare_l2_clip_datasets.py            Synthetic CLIP-style datasets
  master_run.sh, master_run2.sh          Orchestration for the full-coverage suite
  run_*.sh                               Per-stage runners (nlist sweep, DeDrift,
                                         multiseed, L8 + CDF, ...)
  merge_method_results.py                Merge single-method JSONs into the main files
  recompute_maintenance_budget.py        Rebuild a results file's budget section
  check_dataset_shapes.py                Dataset shape / load-path self-check
  verify_figure_*.py                     Figure QA (sizes, gaps, subplot aspects)
  log_summary.py                         CLI summary of one logged run

  Figure generators:
    plot_claim_figures.py                Per-dataset QPS-recall frontiers, L-sweeps
    plot_rebuttal_figures.py             Composite panels (frontier_all, lire_l_sweep_1M)
    plot_robustness.py                   Cluster-count sweep figure
    plot_latency_stability.py            P99 stability vs index size
    plot_tail_latency_cdf.py             Tail-latency CDFs
    plot_split_time_breakdown.py         Split-time 100% stacked bars
    paper_figstyle.py                    Shared typography for the sigconf layout

plots/                         Earlier 30k-scale plotting scripts
tests/                         Unit tests
results/                       Experiment output; a curated subset is committed
figures/                       Generated figures (committed)
data/                          Datasets (not committed)
```

## Installation

```bash
git clone git@github.com:BruceChenSF/NoraIVF.git
cd NoraIVF

pip install -r requirements.txt

# (Optional) install the package in development mode
pip install -e .
```

## Preparing Data

Every experiment script expects datasets in `data/`, each a directory holding
`train.npy`, `queries.npy`, and `gt.npy` (unit-normalised `float32` vectors with
exact ground-truth neighbours).

### Option 0: convert local ann-benchmarks HDF5 files (recommended)

Given the standard ann-benchmarks HDF5 files, the pipeline converts them into
canonical dataset directories and records the run in the structured logger
(source fingerprints, per-file SHA-256 provenance, metrics):

```bash
python scripts/prepare_annbench_datasets.py \
    --source-dir /path/to/ann_benchmark \
    --out-root data/annbench

# subset, force re-conversion, or disable logging
python scripts/prepare_annbench_datasets.py \
    --datasets sift-128-euclidean mnist-784-euclidean \
    --overwrite --no-log
```

Output layout (loadable by `load_npy_vector_dataset_l2`):

```
data/annbench/<name>/
    train.npy        # float32 base vectors (N, d)
    queries.npy      # float32 queries (Q, d)
    gt.npy           # int32 ground truth (Q, 100)
    distances.npy    # float32 ground-truth distances (Q, 100)
    meta.json        # source fingerprint, shapes, dtype, normalization flag
```

Re-running is idempotent (unchanged datasets are skipped via source
fingerprint; `--overwrite` forces re-conversion). List converted datasets from
Python with `noraivf.list_datasets("data/annbench")`.

### Option 1: AudioSet embeddings

```bash
python scripts/prepare_audioset.py      # TFRecord embeddings -> npy
```

### Option 2: SIFT1B (for the scale probe)

```bash
# expects base.bin / query.bin / gt.bin in the bigann layout
python scripts/prepare_sift1b.py \
    --base base.bin --query query.bin --gt gt.bin --out data/annbench/sift1b
```

`train.npy` is written as **float16**. SIFT components are integers in 0..255 and
float16 represents every integer up to 2048 exactly, so the conversion is
lossless; it halves the on-disk footprint (256 GB instead of 512 GB), and disk
is the binding constraint at this scale.

### Option 3: synthetic CLIP-style datasets

```bash
python scripts/prepare_l2_clip_datasets.py \
    --datasets core50 ScanNetCLIP Ego4D \
    --n-vectors 30000 --n-queries 1000 --dim 512 \
    --out-root data
```

### Option 4: download and encode real datasets

Requires PyTorch and `torchvision`. Downloads images and encodes them with
ResNet18.

```bash
python scripts/prepare_real_vector_datasets.py \
    --datasets core50 scannet ego4d \
    --train-count 30000 --query-count 1000 \
    --out-root data
```

## Running Experiments

All scripts run from the repository root. Results go to `results/`, figures to
`figures/`.

### Main suite (L2 distance, full-coverage streams)

```bash
# quick smoke test
python experiments/run_claim_evidence_annbench.py --smoke --download

# the protocol the paper reports: build on data[:n_init], then stream n_stream
# updates, with per-dataset operating points.  See --help for the full matrix.
python experiments/run_claim_evidence_annbench.py \
    --datasets sift-128-euclidean deep-image-96-angular \
    --data-dir data/annbench \
    --n-init 500000 --n-stream 500000 --nlist 4096 --seed 42
```

Output: `results/claim_evidence_<dataset>.json`.

### Million-scale P99 stability and mixed-workload latency

```bash
python experiments/run_latency_scale.py --help
```

Output: `results/latency_scale/<dataset>.json` plus the per-operation arrays in
`<dataset>_raw.json` (these raw files are large and are not committed).

### P99 stability, microbenchmark, and SCDKM diagnostics

```bash
python experiments/run_p99_stability_5datasets.py --max-train 30000 --n-batches 150
python experiments/run_microbenchmark_four_methods.py
python experiments/measure_scan_width.py --nlist 4096 --force-splits 15

python experiments/run_cdkm_vs_lloyd_local.py \
    --datasets MNIST Fashion CoRe50 ScanNet Ego4D --max-train 50000 --local-parts 20
```

### Earlier 30k-scale suite

```bash
python experiments/run_experiments.py     # requires data/ScanNetCLIP, data/Ego4D
```

### LIRE L-sweep diagnostics

```bash
python experiments/run_claim_evidence_annbench.py \
    --only-lire-sweep --l-values 0 1 2 5 10 \
    --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
    --output results/lire_l_sweep_5datasets.json

python scripts/add_nora_to_lire_sweep.py \
    --input results/lire_l_sweep_5datasets.json \
    --output results/lire_l_sweep_5datasets_with_nora.json

python experiments/run_boundary_sampled_scdkm_diagnostic.py \
    --input results/lire_l_sweep_5datasets_with_nora.json \
    --neighbors 2 --samples 4 8 16 32

python experiments/run_local_cdkm_repair_diagnostic.py \
    --input results/lire_l_sweep_5datasets_with_nora.json --l-values 2 5
```

## Results in this repository

`results/` is ignored by default, so files written by a new run stay untracked.
The JSON and CSV that the paper's tables and figures read are committed
explicitly:

| Path | Read by |
|---|---|
| `results/claim_evidence_*.json` | Main maintenance / recall tables; split-time breakdown |
| `results/nlist_sweep_full/`, `results/nlist_sweep/` | Cluster-count sweep table and figure |
| `results/multiseed_large/`, `results/multiseed_dedrift/`, `results/multiseed_deep_fullcov/`, `results/claim_evidence_multiseed60k.json` | Multi-seed stability |
| `results/latency_scale/*.json` (aggregated) | Latency comparison panels |
| `results/p99_stability_5datasets.{json,csv}` | Tail-latency table |
| `results/microbenchmark_four_methods_30k.{json,csv}` | Insert / delete microbenchmark |
| `results/scdkm_scan_width_1m.json` | SCDKM candidate-set-width microbenchmark |
| `results/lean_fidelity.json` | Lean-substrate equivalence gate |
| `results/lire_l8/`, `results/sift1m_*.csv`, `results/claim_evidence_summary_nprobe8.csv` | L-sweep and summary panels |

Three families are deliberately **not** committed, because of their size
(about 236 MB together):

* `results/latency_scale/*_raw.json` (188 MB) — per-operation latency arrays,
  read only by `scripts/plot_tail_latency_cdf.py`. Regenerate with
  `experiments/run_latency_scale.py`.
* `results/latency_scale_dedrift/` (49 MB) — the DeDrift pass over the same
  protocol, read by `scripts/plot_latency_stability.py`. Regenerate with
  `scripts/run_dedrift_latency.sh`.
* `results/logs/`, `results/pi_logs/` — runtime logs, which are not inputs to
  anything.

Those two figures therefore need one experiment run each before they can be
reproduced; every other figure and table works from the committed JSON alone.

## Generating Plots

Figure scripts share one typography module so that the panels land consistently
in the two-column ACM sigconf layout (`\linewidth` = 3.34 in, `\textwidth` =
7.00 in):

```bash
python scripts/plot_claim_figures.py    --input 'results/claim_evidence_*.json'
python scripts/plot_rebuttal_figures.py
python scripts/plot_robustness.py
python scripts/plot_latency_stability.py
python scripts/plot_tail_latency_cdf.py
python scripts/plot_split_time_breakdown.py
```

`scripts/plot_latency_stability.py` is the authoritative writer of
`figures/latency_stability_1M.pdf`. `scripts/plot_rebuttal_figures.py` holds a
superseded four-method variant of the same figure and writes it as
`latency_stability_legacy_4method.*`, so the two cannot overwrite each other.

The older 30k-scale plots live in `plots/`:

```bash
python plots/plot_stability_summary_5datasets.py --input results/p99_stability_5datasets.json
python plots/plot_stability_traces_5datasets.py  --input results/p99_stability_5datasets.json
python plots/plot_tail_latency_cdf_5datasets.py  --input results/p99_stability_5datasets.json
python plots/plot_latency_stability_combined.py  --microbench results/microbenchmark_four_methods_30k.json
python plots/plot_lire_l_sweep_trends.py         --input results/claim_evidence_annbench.json
python plots/plot_qps_recall_with_l0.py          --input results/claim_evidence_annbench.json
```

## Running Tests

```bash
python tests/test_claim_evidence_l2.py
python tests/test_data_pipeline.py
python tests/test_logging.py
python tests/test_ada_ivf.py
```

## Logging

Every experiment run writes a self-contained, disk-persisted log under
`results/logs/<run_id>/`:

```
results/logs/<run_id>/
    meta.json            # env + config + data provenance (git commit, machine, packages)
    run.jsonl            # append-only JSONL event stream, flushed per event
    errors.jsonl         # error events with full stack traces
    events.summary.json  # written by the summary tool
```

Usage from an experiment script (the field schema is documented in
`noraivf/logging/logger.py`):

```python
from noraivf.logging import ExperimentLogger

with ExperimentLogger("exp-scale-10m", config={"nlist": 4096, "seed": 42}) as lg:
    with lg.phase("build"):
        lg.checkpoint("build", batch=1, n=10000)     # progress markers
    with lg.phase("stream"):
        lg.metric("stream", throughput=31250.0, n_splits=128)
    lg.add_data_provenance("data/sift1m/train.npy")  # sha256 fingerprint
    try:
        ...
    except Exception as exc:
        lg.error(exc, phase="stream", batch_idx=42)   # stack trace -> errors.jsonl
```

View a run:

```bash
python scripts/log_summary.py --latest
python scripts/log_summary.py --run-dir results/logs/<run_id>
python scripts/log_summary.py --run-dir results/logs/<run_id> --json
```

Every event is appended and flushed immediately (crash-safe); error events are
fsynced and mirrored to `errors.jsonl`; `meta.json` pins the exact environment
(git commit, python and package versions, machine, GPU: none) and data
fingerprints, so a run can be reproduced and a failure traced.

## A note on scale

The reference implementations in `noraivf/core.py` are Python/NumPy simulators
that materialise one Python object per vector, which costs about 1.3 KB per live
vector and puts a practical ceiling near 5e8 vectors on a 503 GB machine.
`noraivf/lean.py` replaces those containers with array-backed dict views that
present the identical interface, so the algorithms in `core.py` run unmodified
against a compact substrate; `experiments/fidelity_check.py` asserts the two
produce identical partitions. `experiments/run_billion_probe.py` uses that
substrate to measure per-split maintenance cost at 1e9 scale, and is the reason
`lean_methods.py` carries the `top-L` and `full-k` variants of the CDKM
candidate set.

## License

MIT — see `LICENSE`.
