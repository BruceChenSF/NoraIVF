#!/bin/bash
# DEEP-9.9M full-coverage multiseed (3 seeds x SPF/Nora/Ada), skip LIRE sweep.
cd /home/csf/projects/NoraIVF
PY=.venv/bin/python
RB="experiments/run_claim_evidence_annbench.py"
LOG="--checkpoint-every 1 --skip-lire-sweep --log-dir results/logs"
mkdir -p results/multiseed_deep_fullcov
echo "[ms-deep] start $(date)"

run_seed () {
  local SEED=$1
  OMP_NUM_THREADS=28 $PY $RB --datasets deep-image-96-angular \
    --n-init 4995000 --n-stream 4995000 --gt-scope full --seeds $SEED \
    --methods SPFresh NovaIVF AdaIVF \
    --output results/multiseed_deep_fullcov/deep_seed${SEED}.json $LOG \
    > results/exp_ms_deep_fullcov_${SEED}.log 2>&1
  echo "[ms-deep] seed ${SEED} done $(date)"
}

run_seed 42 & echo "[ms-deep] seed42 pid $!"
run_seed 43 & echo "[ms-deep] seed43 pid $!"
run_seed 44 & echo "[ms-deep] seed44 pid $!"
wait
echo "[ms-deep] ALL DONE $(date)"
