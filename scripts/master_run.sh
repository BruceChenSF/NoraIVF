#!/bin/bash
# NoraIVF rebuttal experiment orchestrator (chained batches)
# J6/J7 after small batch finishes; J3/J4 after audioset-unbal conversion finishes.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PY=.venv/bin/python
RUN="experiments/run_claim_evidence_annbench.py"
LEAN_METHODS="Frozen DeDrift SPFresh NovaIVF AdaIVF"

echo "[master] start $(date)"

# --- wait for J1 (small batch: mnist + fashion-mnist) ---
while pgrep -f "claim_evidence_small.json" > /dev/null; do sleep 60; done
echo "[master] J1 done, launching GIST batches $(date)"

# J6: GIST 500k subset, lean methods
OMP_NUM_THREADS=32 setsid nohup $PY $RUN \
  --datasets gist-960-euclidean --max-train 500000 \
  --methods $LEAN_METHODS \
  --output results/claim_evidence_gist500k.json \
  --checkpoint-every 1 --log-dir results/logs \
  > results/exp_gist.log 2>&1 < /dev/null &
echo "[master] J6 gist pid $!"

# J7: GIST Rebuild extrapolation probe (10 rebuilds at n_stream=100)
sleep 5
OMP_NUM_THREADS=16 setsid nohup $PY $RUN \
  --datasets gist-960-euclidean --max-train 500000 \
  --methods Rebuild --n-stream 100 \
  --output results/claim_evidence_gist_rebuild_est.json \
  --checkpoint-every 1 --log-dir results/logs \
  > results/exp_gist_rebuild.log 2>&1 < /dev/null &
echo "[master] J7 gist-rebuild-est pid $!"

# --- wait for unbal conversion ---
while pgrep -f "prepare_audioset_vggish.py" > /dev/null; do sleep 60; done
echo "[master] audioset unbal conversion done, launching audioset + deep batches $(date)"

# J3: audioset-unbal-128 (~2M), lean methods
OMP_NUM_THREADS=32 setsid nohup $PY $RUN \
  --datasets audioset-unbal-128 \
  --methods $LEAN_METHODS \
  --output results/claim_evidence_audioset.json \
  --checkpoint-every 1 --log-dir results/logs \
  > results/exp_audioset.log 2>&1 < /dev/null &
echo "[master] J3 audioset pid $!"

# J4: deep-image-96-angular (9.9M), lean methods
sleep 5
OMP_NUM_THREADS=32 setsid nohup $PY $RUN \
  --datasets deep-image-96-angular \
  --methods $LEAN_METHODS \
  --output results/claim_evidence_deep.json \
  --checkpoint-every 1 --log-dir results/logs \
  > results/exp_deep.log 2>&1 < /dev/null &
echo "[master] J4 deep pid $!"

echo "[master] all batches launched; master exiting $(date)"
