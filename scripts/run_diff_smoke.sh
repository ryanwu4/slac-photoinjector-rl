#!/usr/bin/env bash
# Smoke-test the SHAC + BPTT training pipeline against the trained v1
# EmittanceMLP surrogate.
#
# GPU selection:
#   - Pass --gpu N (default 0) to pin to GPU N via CUDA_VISIBLE_DEVICES.
#     Inside the process, torch then sees that GPU as cuda:0.
#   - Alternatively `CUDA_VISIBLE_DEVICES=1 ./run_diff_smoke.sh` works.
#   - Use --device cpu to run on CPU (slow but verifies the pipeline).
#
# Usage:
#   ./scripts/run_diff_smoke.sh                       # GPU 0, defaults
#   ./scripts/run_diff_smoke.sh --gpu 1               # GPU 1
#   ./scripts/run_diff_smoke.sh --gpu 1 --epochs 20   # longer run
#   ./scripts/run_diff_smoke.sh --device cpu          # CPU
#   ./scripts/run_diff_smoke.sh --algo shac           # only SHAC
set -euo pipefail

GPU=1
DEVICE=""
EPOCHS=10
SEED=0
ALGO=both          # shac | bptt | both
LOGROOT=logs

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --algo)       ALGO="$2"; shift 2 ;;
        --logroot)    LOGROOT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="trained/emittance_target_hifi/checkpoints/best-epoch=191-val_loss=0.0060.ckpt"
NORM="processed/emittance_target_hifi_norm.json"
PYTHON="${PYTHON:-/home/rwu4/miniconda3/envs/slac-rl/bin/python}"

# Pin to one physical GPU. CUDA_VISIBLE_DEVICES makes torch see only that
# GPU, renumbered as cuda:0. This is the right pattern when sharing a node
# (it prevents accidental allocations on the other GPU).
if [[ -z "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    DEVICE="cuda:0"
fi

echo "=========================================="
echo " photoinjector diff-RL smoke"
echo "   device:           $DEVICE"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "   epochs:           $EPOCHS"
echo "   seed:             $SEED"
echo "   algo:             $ALGO"
echo "   surrogate ckpt:   $CKPT"
echo "=========================================="

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: surrogate ckpt missing: $CKPT" >&2; exit 1
fi
if [[ ! -f "$NORM" ]]; then
    echo "ERROR: norm json missing: $NORM" >&2; exit 1
fi

run_shac() {
    local logdir="$LOGROOT/shac_smoke_seed${SEED}"
    echo ">>> SHAC -> $logdir"
    "$PYTHON" -m photoinjector_rl.emittance_target.train_shac \
        --cfg configs/diff_rl/shac_photoinjector.yaml \
        --ckpt "$CKPT" --norm-json "$NORM" \
        --logdir "$logdir" --seed "$SEED" \
        --max-epochs "$EPOCHS" --device "$DEVICE"
}

run_bptt() {
    local logdir="$LOGROOT/bptt_smoke_seed${SEED}"
    echo ">>> BPTT -> $logdir"
    "$PYTHON" -m photoinjector_rl.emittance_target.train_bptt \
        --cfg configs/diff_rl/bptt_photoinjector.yaml \
        --ckpt "$CKPT" --norm-json "$NORM" \
        --logdir "$logdir" --seed "$SEED" \
        --max-epochs "$EPOCHS" --device "$DEVICE"
}

case "$ALGO" in
    shac) run_shac ;;
    bptt) run_bptt ;;
    both) run_shac; run_bptt ;;
    *) echo "ERROR: --algo must be one of: shac, bptt, both" >&2; exit 2 ;;
esac

echo "=========================================="
echo " Done. Log dirs under: $LOGROOT/"
echo " TensorBoard:   tensorboard --logdir $LOGROOT"
echo " Learning CSV:  $LOGROOT/*/learning_curve.csv"
echo "=========================================="
