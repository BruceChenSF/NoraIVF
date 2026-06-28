# NoraIVF

A memory-native dynamic IVF index for streaming vector search. NoraIVF introduces the **SCDKM** (Streaming Coordinate Descent K-Means) protocol, which uses coordinate-descent k-means for split operations on IVF posting lists without needing explicit boundary repair.

## Directory Structure

```
noraivf/
├── noraivf/                  # Core Python package
│   ├── __init__.py           # Package re-exports
│   ├── core.py               # IVF index implementations (Frozen, Rebuild, DeDrift, SPFresh, NovaIVF)
│   └── datasets.py           # Dataset loading, generation, and ground-truth computation
├── experiments/              # Experiment scripts
│   ├── run_experiments.py    # Main experiment suite (QPS-Recall, Stability, Microbenchmark, etc.)
│   ├── run_claim_evidence_annbench.py  # L2-distance experiments on ann-benchmarks datasets
│   ├── run_cdkm_vs_lloyd_local.py      # CDKM vs Lloyd SSE convergence diagnostic
│   ├── run_boundary_sampled_scdkm_diagnostic.py  # Boundary-sampled SCDKM variant
│   ├── run_local_cdkm_repair_diagnostic.py       # Local multi-cluster CDKM repair
│   ├── run_microbenchmark_four_methods.py        # Insert/delete microbenchmark (5 datasets)
│   └── run_p99_stability_5datasets.py            # P99 query-latency stability table
├── scripts/                  # Data preparation and utility scripts
│   ├── prepare_real_vector_datasets.py  # Download/encode real datasets from HuggingFace
│   ├── prepare_l2_clip_datasets.py      # Generate synthetic CLIP-style datasets
│   ├── add_nora_to_lire_sweep.py        # Post-hoc append Nora baselines to LIRE sweep JSON
│   └── build_claim_support_summary.py   # Build claim-evidence summary from experiment JSONs
├── plots/                    # Plotting scripts (read experiment JSONs, produce PDF/PNG)
│   ├── plot_lire_l_sweep_trends.py
│   ├── plot_qps_recall_with_l0.py
│   ├── plot_stability_summary_5datasets.py
│   ├── plot_stability_traces_5datasets.py
│   ├── plot_tail_latency_cdf_5datasets.py
│   └── plot_latency_stability_combined.py
├── tests/                    # Unit tests
│   └── test_claim_evidence_l2.py
├── results/                  # Experiment output (JSON/CSV) — created automatically
├── figures/                  # Generated figures (PDF/PNG) — created automatically
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone git@github.com:BruceChenSF/NoraIVF.git
cd NoraIVF

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install the package in development mode
pip install -e .
```

## Preparing Data

All experiment scripts expect datasets in the `data/` directory. Each dataset should be a directory containing `train.npy`, `queries.npy`, and `gt.npy` (unit-normalized float32 vectors with exact ground-truth neighbors).

### Option 1: Generate synthetic CLIP-style datasets

```bash
python scripts/prepare_l2_clip_datasets.py \
    --datasets core50 ScanNetCLIP Ego4D \
    --n-vectors 30000 --n-queries 1000 --dim 512 \
    --out-root data
```

### Option 2: Download and encode real datasets from HuggingFace

Requires PyTorch and `torchvision`. This downloads images from HuggingFace and encodes them with ResNet18.

```bash
python scripts/prepare_real_vector_datasets.py \
    --datasets core50 scannet ego4d \
    --train-count 30000 --query-count 1000 \
    --out-root data
```

### Option 3: Use ann-benchmarks datasets (MNIST, Fashion-MNIST)

Download the ann-benchmarks HDF5 files:

```bash
python experiments/run_claim_evidence_annbench.py \
    --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
    --download --data-dir data/annbench \
    --max-train 30000 --max-queries 1000
```

## Running Experiments

All scripts are run from the repository root (`noraivf/`). Output results go to `results/` and figures to `figures/`.

### Main Experiment Suite (cosine/IP distance)

Runs Experiments 3–9 on ScanNetCLIP and Ego4D datasets. Requires datasets in `data/ScanNetCLIP/` and `data/Ego4D/`.

```bash
python experiments/run_experiments.py
```

Output: `results/all_results.json`

### Claim-Evidence Experiments (L2 distance)

Full suite on ann-benchmarks and real datasets with multi-seed support:

```bash
# Quick smoke test
python experiments/run_claim_evidence_annbench.py --smoke --download

# Full run on all 5 datasets (with 3 seeds)
python experiments/run_claim_evidence_annbench.py \
    --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
    --npy-datasets core50 ScanNetCLIP Ego4D \
    --seeds 42 123 456 \
    --max-train 30000 --max-queries 1000
```

Output: `results/claim_evidence_annbench.json`

### CDKM vs Lloyd Diagnostic

Compares finite-budget CDKM and Lloyd k-means on local data partitions:

```bash
python experiments/run_cdkm_vs_lloyd_local.py \
    --datasets MNIST Fashion CoRE50 ScanNet Ego4D \
    --max-train 50000 --local-parts 20
```

Output: `results/cdkm_vs_lloyd_local_summary.csv`, `figures/cdkm_vs_lloyd_local_diagnostic.pdf`

### P99 Stability (5 datasets)

```bash
python experiments/run_p99_stability_5datasets.py \
    --max-train 30000 --n-batches 150
```

Output: `results/p99_stability_5datasets.json`

### Microbenchmark (4 methods)

```bash
python experiments/run_microbenchmark_four_methods.py
```

Output: `results/microbenchmark_four_methods_30k.json`, `figures/exp4_microbench.pdf`

### Boundary-Sampled SCDKM Diagnostic

Requires prior LIRE L-sweep results as input:

```bash
# First, generate the LIRE L-sweep baseline
python experiments/run_claim_evidence_annbench.py \
    --only-lire-sweep --l-values 0 1 2 5 10 \
    --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
    --npy-datasets core50 ScanNetCLIP Ego4D \
    --output results/lire_l_sweep_5datasets.json

# Then add Nora baselines
python scripts/add_nora_to_lire_sweep.py \
    --input results/lire_l_sweep_5datasets.json \
    --output results/lire_l_sweep_5datasets_with_nora.json

# Run the diagnostic
python experiments/run_boundary_sampled_scdkm_diagnostic.py \
    --input results/lire_l_sweep_5datasets_with_nora.json \
    --neighbors 2 --samples 4 8 16 32
```

### Local CDKM Repair Diagnostic

```bash
python experiments/run_local_cdkm_repair_diagnostic.py \
    --input results/lire_l_sweep_5datasets_with_nora.json \
    --l-values 2 5
```

## Running Tests

```bash
python tests/test_claim_evidence_l2.py
```

## Generating Plots

Each plot script reads a result JSON and produces PDF/PNG figures:

```bash
# Stability summary bar chart
python plots/plot_stability_summary_5datasets.py \
    --input results/p99_stability_5datasets.json

# Stability traces over time
python plots/plot_stability_traces_5datasets.py \
    --input results/p99_stability_5datasets.json

# Tail latency CDF
python plots/plot_tail_latency_cdf_5datasets.py \
    --input results/p99_stability_5datasets.json

# Combined latency stability
python plots/plot_latency_stability_combined.py \
    --microbench results/microbenchmark_four_methods_30k.json

# LIRE L-sweep trends
python plots/plot_lire_l_sweep_trends.py \
    --input results/claim_evidence_annbench.json

# QPS-Recall with L0
python plots/plot_qps_recall_with_l0.py \
    --input results/claim_evidence_annbench.json
```

## Key Concepts

- **SCDKM Protocol**: Uses coordinate-descent k-means (CDKM) for split operations, eliminating the NPA (Nearest Partition Assignment) repair step required by LIRE/SPFresh.
- **Posting Pointer Map (PPM)**: O(1) swap-and-pop deletion with contiguous memory layout.
- **ClusterEntry**: Maintained statistics (`n_j`, `a_j`, `s_j`) enabling per-point assignment decisions without full-data scans.

## Citation

TBD (manuscript under review).
