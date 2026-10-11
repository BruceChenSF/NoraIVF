#!/bin/bash
# run_gist_maxtrain_fix.sh
# Fix the GIST re-run: the original script omitted --max-train 500000 and so
# loaded the full 1M corpus
# (the paper uses the 500k subset, see scripts/master_run.sh:17). Affected:
#   results/multiseed_dedrift/gist-960-euclidean_seed{42,43,44}.json
#   results/nlist_sweep_full/gist-960-euclidean_nlist{256,512,2048}.json
# Note: results/latency_scale_dedrift/gist-960-euclidean_dedrift.json does NOT need fixing
#       -- the main latency run already used n_train=1000000 (full corpus), so the
#       two agree.
# Expected values for the corrected configuration, matching the paper:
# n_train=500000, n_init=n_stream=250000, coverage=1.0
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PY=.venv/bin/python
MS=experiments/run_claim_evidence_annbench.py
MOUT=results/multiseed_dedrift
NOUT=results/nlist_sweep_full
BAD=results/_bad_gist_maxtrain
mkdir -p "$BAD" results/logs
export OMP_NUM_THREADS=16
SLOTS=${SLOTS:-6}

echo "[gist-fix] start $(date '+%F %T')"

# ---- 1) multi-seed: GIST x 3 seeds (operating point nlist=1024) ----
for sd in 42 43 44; do
    f="$MOUT/gist-960-euclidean_seed${sd}.json"
    [ -f "$f" ] && mv "$f" "$BAD/" && echo "[gist-fix] backed up $(basename "$f")"
    nohup $PY $MS --datasets gist-960-euclidean --max-train 500000 \
        --n-init 250000 --n-stream 250000 --nlist 1024 \
        --gt-scope full --seeds "$sd" --methods DeDrift \
        --checkpoint-every 1 --skip-lire-sweep --log-dir results/logs \
        --output "$f" > "results/exp_gistfix_ms_seed${sd}.log" 2>&1 &
    echo "[gist-fix] launch ms seed=$sd pid=$! $(date '+%T')"
done

# ---- 2) nlist sweep: GIST at 256 / 512 / 2048 ----
for nl in 256 512 2048; do
    f="$NOUT/gist-960-euclidean_nlist${nl}.json"
    [ -f "$f" ] && mv "$f" "$BAD/" && echo "[gist-fix] backed up $(basename "$f")"
    nohup env OMP_NUM_THREADS=16 $PY $MS --datasets gist-960-euclidean --max-train 500000 \
        --n-init 250000 --n-stream 250000 --nlist "$nl" \
        --methods Frozen DeDrift SPFresh NovaIVF AdaIVF \
        --output "$f" \
        --checkpoint-every 1 --log-dir results/logs --skip-lire-sweep --gt-scope full \
        > "results/exp_gistfix_nlist${nl}.log" 2>&1 &
    echo "[gist-fix] launch nlist=$nl pid=$! $(date '+%T')"
done

wait
echo "[gist-fix] ALL DONE $(date '+%F %T')"
echo -n "multiseed gist: "; ls -1 "$MOUT"/gist-960-euclidean_seed*.json 2>/dev/null | wc -l
echo -n "nlist gist:     "; ls -1 "$NOUT"/gist-960-euclidean_nlist*.json 2>/dev/null | wc -l
