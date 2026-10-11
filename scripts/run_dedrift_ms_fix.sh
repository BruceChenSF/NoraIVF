#!/bin/bash
# run_dedrift_ms_fix.sh
# Fix the DeDrift multi-seed re-run: the original script omitted --nlist, so
# MNIST/Fashion/GIST silently used the default of 4096.
# Correct operating points: MNIST 512, Fashion 512, GIST 1024 (as in the paper).
# The wrong files are moved to _bad_nlist4096/ before re-running.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PY=.venv/bin/python
MS=experiments/run_claim_evidence_annbench.py
OUT=results/multiseed_dedrift
BAD="$OUT/_bad_nlist4096"
mkdir -p "$BAD" results/logs
export OMP_NUM_THREADS=8
SLOTS=${SLOTS:-6}

T=(
"mnist-784-euclidean|30093|29907|512"
"fashion-mnist-784-euclidean|30093|29907|512"
"gist-960-euclidean|250000|250000|1024"
)

echo "[ms-fix] start $(date '+%F %T')"
for t in "${T[@]}"; do
    IFS='|' read -r ds ni ns nl <<< "$t"
    for sd in 42 43 44; do
        f="$OUT/${ds}_seed${sd}.json"
        if [ -f "$f" ] && [ ! -f "$BAD/$(basename "$f")" ]; then
            mv "$f" "$BAD/"
            echo "[ms-fix] backed up $(basename "$f") -> _bad_nlist4096/"
        fi
        echo "[ms-fix] launch $ds seed=$sd nlist=$nl $(date '+%F %T')"
        nohup $PY $MS --datasets "$ds" --n-init "$ni" --n-stream "$ns" --nlist "$nl" \
            --gt-scope full --seeds "$sd" --methods DeDrift \
            --checkpoint-every 1 --skip-lire-sweep --log-dir results/logs \
            --output "$f" > "results/exp_msfix_${ds}_${sd}.log" 2>&1 &
        while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 20; done
    done
done
wait
echo "[ms-fix] ALL DONE $(date '+%F %T')"
ls -1 "$OUT"/*.json | wc -l
