#!/bin/bash
# run_dedrift_latency.sh - re-run the DeDrift latency experiment, fixing the
# singular --dataset argument used earlier
# Parameters aligned with the existing latency_scale/*_raw.json: DEEP uses a 200k window
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
PY=.venv/bin/python
LAT=experiments/run_latency_scale.py
LOUT=results/latency_scale_dedrift
mkdir -p "$LOUT" results/logs
STATE=results/dedrift_latency.state.tsv
: > "$STATE"
export OMP_NUM_THREADS=8
SLOTS=${SLOTS:-4}

LT=(
"mnist|mnist-784-euclidean|30093|29907|512"
"fashion|fashion-mnist-784-euclidean|30093|29907|512"
"sift|sift-128-euclidean|500000|500000|4096"
"gist|gist-960-euclidean|250000|250000|1024"
"glove|glove-100-angular|591757|591757|4096"
"audioset|audioset-unbal-128|1021434|1020355|4096"
"deep|deep-image-96-angular|4995000|200000|4096"
)

echo "[dedrift-lat] start $(date '+%F %T') slots=$SLOTS tasks=${#LT[@]}"
i=0; total=${#LT[@]}; PIDS=(); TAGS=()
while [ "$i" -lt "$total" ] || [ "${#PIDS[@]}" -gt 0 ]; do
    while [ "${#PIDS[@]}" -lt "$SLOTS" ] && [ "$i" -lt "$total" ]; do
        IFS='|' read -r tag ds ni ns nl <<< "${LT[$i]}"
        nohup $PY $LAT --dataset "$ds" --n-init "$ni" --n-stream "$ns" --nlist "$nl" \
            --methods DeDrift --dump-raw-latencies --seed 42 \
            --output "$LOUT/${ds}_dedrift.json" \
            >> "results/exp_dedrift_lat_${tag}.log" 2>&1 &
        pid=$!
        echo -e "$pid\t$tag\t$ds\t$nl\t$(date '+%F %T')" >> "$STATE"
        echo "[dedrift-lat] launch $tag pid=$pid nlist=$nl $(date '+%T')"
        PIDS+=("$pid"); TAGS+=("$tag")
        i=$((i + 1))
    done
    sleep 45
    NP=(); NT=()
    for k in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$k]}" 2>/dev/null; then NP+=("${PIDS[$k]}"); NT+=("${TAGS[$k]}")
        else echo "[dedrift-lat] done ${TAGS[$k]} $(date '+%T')"; fi
    done
    PIDS=("${NP[@]}"); TAGS=("${NT[@]}")
done
echo "[dedrift-lat] ALL DONE $(date '+%F %T')"
ls -1 "$LOUT"/*.json 2>/dev/null | wc -l
