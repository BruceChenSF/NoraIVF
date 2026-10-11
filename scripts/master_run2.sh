#!/bin/bash
# NoraIVF rebuttal experiment orchestrator v2 — full-coverage protocol + bounded deep
cd /home/csf/projects/NoraIVF
PY=.venv/bin/python
RUN="experiments/run_claim_evidence_annbench.py"
LEAN="Frozen DeDrift SPFresh NovaIVF AdaIVF"
FULL="Frozen Rebuild DeDrift SPFresh NovaIVF AdaIVF"
LOG="--checkpoint-every 1 --log-dir results/logs"

echo "[master2] start $(date)"
# 备份有界窗口的旧结果
for n in small sift_adaivf glove gist500k audioset deep; do
  [ -f results/claim_evidence_$n.json ] && cp results/claim_evidence_$n.json results/claim_evidence_${n}_bounded.json.bak
done

# ---------- Group A: 3 jobs ----------
OMP_NUM_THREADS=32 setsid nohup $PY $RUN --datasets sift-128-euclidean \
  --n-init 500000 --n-stream 500000 --methods $FULL \
  --output results/claim_evidence_sift.json $LOG \
  > results/exp_sift_full.log 2>&1 < /dev/null &
echo "[master2] A1 sift-full pid $!"

OMP_NUM_THREADS=32 setsid nohup $PY $RUN --datasets glove-100-angular \
  --n-init 591757 --n-stream 591757 --methods $LEAN \
  --output results/claim_evidence_glove.json $LOG \
  > results/exp_glove_full.log 2>&1 < /dev/null &
echo "[master2] A2 glove-full pid $!"

OMP_NUM_THREADS=32 setsid nohup $PY $RUN \
  --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
  --n-init 30093 --n-stream 29907 --methods $FULL \
  --output results/claim_evidence_small.json $LOG \
  > results/exp_small_full.log 2>&1 < /dev/null &
echo "[master2] A3 small-full pid $!"
wait

# ---------- Group B: 3 jobs ----------
OMP_NUM_THREADS=32 setsid nohup $PY $RUN --datasets gist-960-euclidean --max-train 500000 \
  --n-init 250000 --n-stream 250000 --methods $FULL \
  --output results/claim_evidence_gist500k.json $LOG \
  > results/exp_gist_full.log 2>&1 < /dev/null &
echo "[master2] B1 gist-full pid $!"

OMP_NUM_THREADS=32 setsid nohup $PY $RUN --datasets audioset-unbal-128 \
  --n-init 1021434 --n-stream 1020355 --methods $LEAN \
  --output results/claim_evidence_audioset.json $LOG \
  > results/exp_audioset_full.log 2>&1 < /dev/null &
echo "[master2] B2 audioset-full pid $!"

# Rebuild 量级探针：3 次重建直接量单次成本
OMP_NUM_THREADS=16 setsid nohup $PY $RUN \
  --datasets glove-100-angular audioset-unbal-128 \
  --n-init 300000 --n-stream 3000 --methods Rebuild \
  --output results/claim_evidence_rebuild_probe.json $LOG \
  > results/exp_rebuild_probe.log 2>&1 < /dev/null &
echo "[master2] B3 rebuild-probe pid $!"
wait

# ---------- Group C: deep (需要 --gt-scope indexed 已实现) ----------
OMP_NUM_THREADS=32 setsid nohup $PY $RUN --datasets deep-image-96-angular \
  --n-init 4950000 --n-stream 100000 --methods $LEAN --gt-scope indexed \
  --output results/claim_evidence_deep.json $LOG \
  > results/exp_deep_full.log 2>&1 < /dev/null &
echo "[master2] C1 deep-bounded pid $!"
wait
echo "[master2] ALL DONE $(date)"
