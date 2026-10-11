#!/bin/bash
# Multiseed stability across the full 7-dataset suite (3 seeds x SPF/Nora/Ada).
# Seeds run in parallel; datasets sequential inside each seed job.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PY=.venv/bin/python
RB="experiments/run_claim_evidence_annbench.py"
LOG="--checkpoint-every 1 --log-dir results/logs"
METHODS="SPFresh NovaIVF AdaIVF"
mkdir -p results/multiseed_large
echo "[ms-large] start $(date)"

run_seed () {
  local SEED=$1
  # SIFT-1M
  OMP_NUM_THREADS=24 $PY $RB --datasets sift-128-euclidean \
    --n-init 500000 --n-stream 500000 --seeds $SEED --methods $METHODS \
    --output results/multiseed_large/sift_seed${SEED}.json $LOG \
    > results/exp_ms_sift_${SEED}.log 2>&1
  # GIST-500k
  OMP_NUM_THREADS=24 $PY $RB --datasets gist-960-euclidean --max-train 500000 \
    --n-init 250000 --n-stream 250000 --seeds $SEED --methods $METHODS \
    --output results/multiseed_large/gist_seed${SEED}.json $LOG \
    > results/exp_ms_gist_${SEED}.log 2>&1
  # GloVe
  OMP_NUM_THREADS=24 $PY $RB --datasets glove-100-angular \
    --n-init 591757 --n-stream 591757 --seeds $SEED --methods $METHODS \
    --output results/multiseed_large/glove_seed${SEED}.json $LOG \
    > results/exp_ms_glove_${SEED}.log 2>&1
  # AudioSet
  OMP_NUM_THREADS=24 $PY $RB --datasets audioset-unbal-128 \
    --n-init 1021434 --n-stream 1020355 --seeds $SEED --methods $METHODS \
    --output results/multiseed_large/audioset_seed${SEED}.json $LOG \
    > results/exp_ms_audioset_${SEED}.log 2>&1
  # DEEP-9.9M (bounded window)
  OMP_NUM_THREADS=24 $PY $RB --datasets deep-image-96-angular \
    --n-init 4950000 --n-stream 100000 --gt-scope indexed --seeds $SEED --methods $METHODS \
    --output results/multiseed_large/deep_seed${SEED}.json $LOG \
    > results/exp_ms_deep_${SEED}.log 2>&1
  echo "[ms-large] seed ${SEED} done $(date)"
}

run_seed 42 & echo "[ms-large] seed42 pid $!"
run_seed 43 & echo "[ms-large] seed43 pid $!"
run_seed 44 & echo "[ms-large] seed44 pid $!"
wait
echo "[ms-large] ALL DONE $(date)"
