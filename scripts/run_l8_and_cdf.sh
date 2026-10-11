#!/bin/bash
# NoraIVF: L=8 sweep fill + raw-latency dumps for CDF coverage (7 datasets)
cd /home/csf/projects/NoraIVF
PY=.venv/bin/python
RB="experiments/run_claim_evidence_annbench.py"
LT="experiments/run_latency_scale.py"
LOG="--log-dir results/logs"
mkdir -p results/lire_l8 results/latency_scale
echo "[l8cdf] start $(date)"

# ---- Part 1: L=8 sweep point (SPFresh-L8 only, no main methods) ----
OMP_NUM_THREADS=12 setsid nohup $PY $RB --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
  --n-init 30093 --n-stream 29907 --only-lire-sweep --l-values 8 \
  --output results/lire_l8/small_l8.json $LOG > results/exp_l8_small.log 2>&1 < /dev/null &
echo "[l8cdf] l8-small pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $RB --datasets sift-128-euclidean \
  --n-init 500000 --n-stream 500000 --only-lire-sweep --l-values 8 \
  --output results/lire_l8/sift_l8.json $LOG > results/exp_l8_sift.log 2>&1 < /dev/null &
echo "[l8cdf] l8-sift pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $RB --datasets glove-100-angular \
  --n-init 591757 --n-stream 591757 --only-lire-sweep --l-values 8 \
  --output results/lire_l8/glove_l8.json $LOG > results/exp_l8_glove.log 2>&1 < /dev/null &
echo "[l8cdf] l8-glove pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $RB --datasets gist-960-euclidean --max-train 500000 \
  --n-init 250000 --n-stream 250000 --only-lire-sweep --l-values 8 \
  --output results/lire_l8/gist_l8.json $LOG > results/exp_l8_gist.log 2>&1 < /dev/null &
echo "[l8cdf] l8-gist pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $RB --datasets audioset-unbal-128 \
  --n-init 1021434 --n-stream 1020355 --only-lire-sweep --l-values 8 \
  --output results/lire_l8/audioset_l8.json $LOG > results/exp_l8_audioset.log 2>&1 < /dev/null &
echo "[l8cdf] l8-audioset pid $!"

# ---- Part 2: raw-latency dumps for CDF (remaining datasets) ----
OMP_NUM_THREADS=12 setsid nohup $PY $LT --dataset mnist-784-euclidean \
  --n-init 30093 --n-stream 29907 --nlist 512 --batch-size 5000 --n-batches 6 \
  --methods Frozen NovaIVF SPFresh AdaIVF --dump-raw-latencies \
  --output results/latency_scale/mnist_raw.json $LOG > results/exp_lat_mnist_raw.log 2>&1 < /dev/null &
echo "[l8cdf] lat-mnist pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $LT --dataset fashion-mnist-784-euclidean \
  --n-init 30093 --n-stream 29907 --nlist 512 --batch-size 5000 --n-batches 6 \
  --methods Frozen NovaIVF SPFresh AdaIVF --dump-raw-latencies \
  --output results/latency_scale/fashion_raw.json $LOG > results/exp_lat_fashion_raw.log 2>&1 < /dev/null &
echo "[l8cdf] lat-fashion pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $LT --dataset gist-960-euclidean \
  --n-init 250000 --n-stream 250000 --nlist 1024 --batch-size 5000 --n-batches 50 \
  --methods Frozen NovaIVF SPFresh AdaIVF --dump-raw-latencies \
  --output results/latency_scale/gist_raw.json $LOG > results/exp_lat_gist_raw.log 2>&1 < /dev/null &
echo "[l8cdf] lat-gist pid $!"

OMP_NUM_THREADS=12 setsid nohup $PY $LT --dataset glove-100-angular \
  --n-init 591757 --n-stream 591757 --nlist 4096 --batch-size 5000 --n-batches 118 \
  --methods Frozen NovaIVF SPFresh AdaIVF --dump-raw-latencies \
  --output results/latency_scale/glove_raw.json $LOG > results/exp_lat_glove_raw.log 2>&1 < /dev/null &
echo "[l8cdf] lat-glove pid $!"

wait
echo "[l8cdf] ALL DONE $(date)"
