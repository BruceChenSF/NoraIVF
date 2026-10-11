#!/bin/bash
# run_gist_maxtrain_fix.sh
# 修正 GIST 补跑：原脚本漏了 --max-train 500000，导致加载完整 1M 语料
# （主实验用 500k 子集，见 scripts/master_run.sh:17）。受影响：
#   results/multiseed_dedrift/gist-960-euclidean_seed{42,43,44}.json
#   results/nlist_sweep_full/gist-960-euclidean_nlist{256,512,2048}.json
# 说明：results/latency_scale_dedrift/gist-960-euclidean_dedrift.json **不需要修**
#       —— 主延迟运行本身就用 n_train=1000000（全语料），两者口径一致。
# 恢复正确配置的期望值（与主实验一致）：n_train=500000, n_init=n_stream=250000, coverage=1.0
cd /home/csf/projects/NoraIVF || exit 1
PY=.venv/bin/python
MS=experiments/run_claim_evidence_annbench.py
MOUT=results/multiseed_dedrift
NOUT=results/nlist_sweep_full
BAD=results/_bad_gist_maxtrain
mkdir -p "$BAD" results/logs
export OMP_NUM_THREADS=16
SLOTS=${SLOTS:-6}

echo "[gist-fix] start $(date '+%F %T')"

# ---- 1) 多 seed：GIST × 3 seeds（工作点 nlist=1024） ----
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

# ---- 2) nlist 扫描：GIST 的 256 / 512 / 2048 三点 ----
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
