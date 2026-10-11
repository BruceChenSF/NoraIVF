#!/bin/bash
# run_dedrift_extra.sh - 为 DeDrift 补齐缺失的两组实验（延迟 + 多 seed）
# 参数严格对齐既有运行：
#   延迟：experiments/run_latency_scale.py，DEEP 用 200k 测量窗口（与 deep_raw.json 一致）
#   多 seed：experiments/run_claim_evidence_annbench.py --seeds，DEEP 全覆盖
cd /home/csf/projects/NoraIVF || exit 1
PY=.venv/bin/python
LAT=experiments/run_latency_scale.py
MS=experiments/run_claim_evidence_annbench.py
LOUT=results/latency_scale_dedrift
MOUT=results/multiseed_dedrift
mkdir -p "$LOUT" "$MOUT" results/logs
STATE=results/dedrift_extra.state.tsv
: > "$STATE"

export OMP_NUM_THREADS=8
SLOTS=${SLOTS:-4}

# ---- 延迟任务: tag|dataset|n_init|n_stream|nlist ----
LT=(
"mnist|mnist-784-euclidean|30093|29907|512"
"fashion|fashion-mnist-784-euclidean|30093|29907|512"
"sift|sift-128-euclidean|500000|500000|4096"
"gist|gist-960-euclidean|250000|250000|1024"
"glove|glove-100-angular|591757|591757|4096"
"audioset|audioset-unbal-128|1021434|1020355|4096"
"deep|deep-image-96-angular|4995000|200000|4096"
)
# ---- 多 seed 任务: dataset|n_init|n_stream|seed ----
MT=(
"deep-image-96-angular|4995000|4995000|42"
"deep-image-96-angular|4995000|4995000|43"
"deep-image-96-angular|4995000|4995000|44"
"audioset-unbal-128|1021434|1020355|42"
"audioset-unbal-128|1021434|1020355|43"
"audioset-unbal-128|1021434|1020355|44"
"sift-128-euclidean|500000|500000|42"
"sift-128-euclidean|500000|500000|43"
"sift-128-euclidean|500000|500000|44"
"glove-100-angular|591757|591757|42"
"glove-100-angular|591757|591757|43"
"glove-100-angular|591757|591757|44"
"gist-960-euclidean|250000|250000|42"
"gist-960-euclidean|250000|250000|43"
"gist-960-euclidean|250000|250000|44"
"mnist-784-euclidean|30093|29907|42"
"mnist-784-euclidean|30093|29907|43"
"mnist-784-euclidean|30093|29907|44"
"fashion-mnist-784-euclidean|30093|29907|42"
"fashion-mnist-784-euclidean|30093|29907|43"
"fashion-mnist-784-euclidean|30093|29907|44"
)

echo "[dedrift-extra] start $(date '+%F %T') slots=$SLOTS"

launch_lat() {
    local tag="$1" ds="$2" ni="$3" ns="$4" nl="$5"
    nohup $PY $LAT --datasets "$ds" --n-init "$ni" --n-stream "$ns" --nlist "$nl" \
        --methods DeDrift --dump-raw-latencies --seed 42 \
        --output "$LOUT/${ds}_dedrift.json" \
        > "results/exp_dedrift_lat_${tag}.log" 2>&1 &
    echo $!
}
launch_ms() {
    local ds="$1" ni="$2" ns="$3" sd="$4"
    nohup $PY $MS --datasets "$ds" --n-init "$ni" --n-stream "$ns" --gt-scope full \
        --seeds "$sd" --methods DeDrift --checkpoint-every 1 --skip-lire-sweep \
        --log-dir results/logs --output "$MOUT/${ds}_seed${sd}.json" \
        > "results/exp_dedrift_ms_${ds}_${sd}.log" 2>&1 &
    echo $!
}

run_queue() {   # $1=kind  $2..=tasks
    local kind="$1"; shift
    local i=0; local tasks=("$@"); local total=${#tasks[@]}
    local PIDS=() TAGS=()
    while [ "$i" -lt "$total" ] || [ "${#PIDS[@]}" -gt 0 ]; do
        while [ "${#PIDS[@]}" -lt "$SLOTS" ] && [ "$i" -lt "$total" ]; do
            IFS='|' read -r a b c d e <<< "${tasks[$i]}"
            if [ "$kind" = "lat" ]; then
                pid=$(launch_lat "$a" "$b" "$c" "$d" "$e"); tag="lat_$a"
            else
                pid=$(launch_ms "$a" "$b" "$c" "$d"); tag="ms_$a_$d"
            fi
            echo -e "$pid\t$kind\t$tag\t$(date '+%F %T')" >> "$STATE"
            echo "[dedrift-extra] launch $kind $tag pid=$pid $(date '+%T')"
            PIDS+=("$pid"); TAGS+=("$tag")
            i=$((i + 1))
        done
        sleep 45
        NP=(); NT=()
        for k in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$k]}" 2>/dev/null; then NP+=("${PIDS[$k]}"); NT+=("${TAGS[$k]}")
            else echo "[dedrift-extra] done ${TAGS[$k]} $(date '+%T')"; fi
        done
        PIDS=("${NP[@]}"); TAGS=("${NT[@]}")
    done
}

echo "=== latency queue (7) ==="
run_queue lat "${LT[@]}"
echo "=== multiseed queue (21) ==="
run_queue ms "${MT[@]}"
echo "[dedrift-extra] ALL DONE $(date '+%F %T')"
ls -1 "$LOUT"/*.json 2>/dev/null | wc -l
ls -1 "$MOUT"/*.json 2>/dev/null | wc -l
