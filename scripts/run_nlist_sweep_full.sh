#!/bin/bash
# run_nlist_sweep_full.sh - 全数据集 nlist 扫描（补跑，含 AdaIVF 与 DeDrift）
#
# 设计：每个数据集在"工作点 nlist"(op) 及 op/4, op/2, op*2 各跑一次全量覆盖流；
#       op 那一列直接取主实验结果（已有，不重跑）。
#   MNIST/Fashion (op=512) : 128 / 256 / (512 from main) / 1024
#   GIST-500k     (op=1024): 256 / 512 / (1024)            / 2048
#   SIFT/GloVe/AudioSet/DEEP (op=4096): 1024 / 2048 / (4096) / 8192
# 方法：Frozen DeDrift SPFresh NovaIVF AdaIVF（Rebuild 成本过高，与主实验一致地省略）
# 目的：为论文 6.2.6 提供"全数据集覆盖的表格 + 两幅代表性面板"
cd /home/csf/projects/NoraIVF || exit 1

PY=.venv/bin/python
RUN=experiments/run_claim_evidence_annbench.py
OUT=results/nlist_sweep_full
mkdir -p "$OUT" results/logs
STATE="$OUT/.state.tsv"
: > "$STATE"

SLOTS=${SLOTS:-6}
METHODS="Frozen DeDrift SPFresh NovaIVF AdaIVF"
EXTRA="--checkpoint-every 1 --log-dir results/logs --skip-lire-sweep --gt-scope full"

# tag|dataset|n_init|n_stream|nlist   （DEEP 排最前，长任务先占槽）
TASKS=(
"deep1024|deep-image-96-angular|4995000|4995000|1024"
"deep2048|deep-image-96-angular|4995000|4995000|2048"
"deep8192|deep-image-96-angular|4995000|4995000|8192"
"audioset1024|audioset-unbal-128|1021434|1020355|1024"
"audioset2048|audioset-unbal-128|1021434|1020355|2048"
"audioset8192|audioset-unbal-128|1021434|1020355|8192"
"sift1024|sift-128-euclidean|500000|500000|1024"
"sift2048|sift-128-euclidean|500000|500000|2048"
"sift8192|sift-128-euclidean|500000|500000|8192"
"glove1024|glove-100-angular|591757|591757|1024"
"glove2048|glove-100-angular|591757|591757|2048"
"glove8192|glove-100-angular|591757|591757|8192"
"gist256|gist-960-euclidean|250000|250000|256"
"gist512|gist-960-euclidean|250000|250000|512"
"gist2048|gist-960-euclidean|250000|250000|2048"
"mnist128|mnist-784-euclidean|30093|29907|128"
"mnist256|mnist-784-euclidean|30093|29907|256"
"mnist1024|mnist-784-euclidean|30093|29907|1024"
"fashion128|fashion-mnist-784-euclidean|30093|29907|128"
"fashion256|fashion-mnist-784-euclidean|30093|29907|256"
"fashion1024|fashion-mnist-784-euclidean|30093|29907|1024"
)

echo "[nlist-full] master start $(date '+%F %T') slots=$SLOTS tasks=${#TASKS[@]}"
echo "[nlist-full] methods: $METHODS (Rebuild omitted: cost)"

launch() {
    local tag="$1" ds="$2" ni="$3" ns="$4" nl="$5"
    nohup env OMP_NUM_THREADS=16 $PY $RUN \
        --datasets "$ds" --n-init "$ni" --n-stream "$ns" --nlist "$nl" \
        --methods $METHODS \
        --output "$OUT/${ds}_nlist${nl}.json" $EXTRA \
        > "results/exp_nlistfull_${tag}.log" 2>&1 &
    echo $!
}

i=0; total=${#TASKS[@]}; ACTIVE_PIDS=(); ACTIVE_TAGS=()
while [ "$i" -lt "$total" ] || [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; do
    while [ "${#ACTIVE_PIDS[@]}" -lt "$SLOTS" ] && [ "$i" -lt "$total" ]; do
        IFS='|' read -r tag ds ni ns nl <<< "${TASKS[$i]}"
        pid=$(launch "$tag" "$ds" "$ni" "$ns" "$nl")
        echo -e "$pid\t$tag\t$ds\t$nl\t$ni\t$ns\t$(date '+%F %T')" >> "$STATE"
        echo "[nlist-full] launch $tag pid=$pid nlist=$nl $(date '+%T')"
        ACTIVE_PIDS+=("$pid"); ACTIVE_TAGS+=("$tag")
        i=$((i + 1))
    done
    sleep 60
    NEW_PIDS=(); NEW_TAGS=()
    for k in "${!ACTIVE_PIDS[@]}"; do
        p="${ACTIVE_PIDS[$k]}"; t="${ACTIVE_TAGS[$k]}"
        if kill -0 "$p" 2>/dev/null; then
            NEW_PIDS+=("$p"); NEW_TAGS+=("$t")
        elif [ -s "$OUT/${t}.json" ] || ls "$OUT"/*_nlist*.json > /dev/null 2>&1; then
            echo "[nlist-full] done $t $(date '+%T')"
        else
            echo "[nlist-full] task $t exited WITHOUT json (check results/exp_nlistfull_${t}.log)"
        fi
    done
    ACTIVE_PIDS=("${NEW_PIDS[@]}"); ACTIVE_TAGS=("${NEW_TAGS[@]}")
done

echo "[nlist-full] ALL TASKS DONE $(date '+%F %T')"
ls -1 "$OUT"/*.json 2>/dev/null | wc -l
