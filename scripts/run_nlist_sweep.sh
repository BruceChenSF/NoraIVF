#!/bin/bash
# NoraIVF robustness sweep: nlist (cluster count) x full-coverage stream.
#
# Scope (nlist=4096 rows already exist in results/claim_evidence_{sift,audioset}.json
# from the main experiments -- NOT re-run):
#   sift-128-euclidean    (1M,  n_init=500000  n_stream=500000):  nlist 1024 2048 8192
#   audioset-unbal-128    (2M,  n_init=1021434 n_stream=1020355): nlist 2048 8192
# Methods per task: Frozen NovaIVF SPFresh (Frozen = free reference; the two local
# dynamic methods match the paper stability experiment's method choice).
#   Frozen/NovaIVF/SPFresh only + --skip-lire-sweep (the unconditional SPFresh-L0..10
#   re-stream would add ~3h per sift / ~7h per audioset task and blow the 10-14h budget).
#   NPA counting kept ON so rows are schema-identical to the nlist=4096 reference rows.
#
# Job control: 3 slots.  sift trio starts first; audioset tasks launch as slots free.
# The script itself is meant to be run detached:
#   setsid nohup bash scripts/run_nlist_sweep.sh > results/exp_nlist_master.log 2>&1 &
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

PY=.venv/bin/python
RUN=experiments/run_claim_evidence_annbench.py
OUT=results/nlist_sweep
mkdir -p "$OUT" results/logs
STATE="$OUT/.nlist_state.tsv"
: > "$STATE"

SLOTS=3
METHODS="Frozen NovaIVF SPFresh"
EXTRA="--checkpoint-every 1 --log-dir results/logs --skip-lire-sweep"

# tag|dataset|n_init|n_stream|nlist
TASKS=(
"sift1024|sift-128-euclidean|500000|500000|1024"
"sift2048|sift-128-euclidean|500000|500000|2048"
"sift8192|sift-128-euclidean|500000|500000|8192"
"audioset2048|audioset-unbal-128|1021434|1020355|2048"
"audioset8192|audioset-unbal-128|1021434|1020355|8192"
)

echo "[nlist-sweep] master start $(date '+%F %T')  slots=$SLOTS tasks=${#TASKS[@]}"

launch() {
    local tag="$1" ds="$2" ni="$3" ns="$4" nl="$5"
    local outjson="$OUT/${ds}_nlist${nl}.json"
    local logf="results/exp_nlist_${ds}_nlist${nl}.log"
    OMP_NUM_THREADS=24 setsid nohup "$PY" "$RUN" --datasets "$ds" \
        --n-init "$ni" --n-stream "$ns" --nlist "$nl" \
        --methods $METHODS --output "$outjson" $EXTRA \
        > "$logf" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$tag" "$ds" "$nl" "$ni" "$ns" "$(date '+%F %T')" >> "$STATE"
    echo "[$(date '+%F %T')] launched $tag dataset=$ds n_init=$ni n_stream=$ns nlist=$nl pid=$pid -> $outjson"
}

i=0
total=${#TASKS[@]}
ACTIVE=()   # basename markers of launched-but-still-running tasks
while [ "$i" -lt "$total" ] || [ "${#ACTIVE[@]}" -gt 0 ]; do
    # fill free slots
    while [ "${#ACTIVE[@]}" -lt "$SLOTS" ] && [ "$i" -lt "$total" ]; do
        IFS='|' read -r tag ds ni ns nl <<< "${TASKS[$i]}"
        launch "$tag" "$ds" "$ni" "$ns" "$nl"
        ACTIVE+=("${ds}_nlist${nl}")
        i=$((i + 1))
    done
    sleep 60
    # reap finished tasks (process gone); log completion/failure
    NEW=()
    for m in "${ACTIVE[@]}"; do
        if pgrep -f "results/nlist_sweep/${m}.json" > /dev/null 2>&1; then
            NEW+=("$m")
        else
            if [ -s "$OUT/$m.json" ]; then
                echo "[$(date '+%F %T')] finished $m OK"
            else
                echo "[$(date '+%F %T')] finished $m FAILED/EMPTY (see results/exp_nlist_*.log)"
            fi
        fi
    done
    ACTIVE=("${NEW[@]}")
done

echo "[nlist-sweep] ALL TASKS DONE $(date '+%F %T')"
echo "--- state (pid tag dataset nlist n_init n_stream launch_ts) ---"
cat "$STATE"
