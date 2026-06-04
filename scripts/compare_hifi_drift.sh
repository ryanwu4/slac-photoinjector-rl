#!/usr/bin/env bash
# High-fidelity PPO/SHAC/BPTT comparison WITH a nonzero distgen drift.
#
# This is the drift variant of the static "next step" comparison printed by
# regen_hifi.sh. It runs the exact same head-to-head against the hi-fi
# EmittanceMLP surrogate (trained/emittance_target_hifi), but turns on a
# per-step Gaussian random walk on the hidden 6-D distgen context to model
# cathode shot-to-shot jitter.
#
# The drift std is applied IDENTICALLY to every algo for BOTH training and the
# post-training deterministic eval (compare_diff_algos.py threads it into
# train_{shac,bptt,ppo}; SHAC/BPTT eval recovers it from each run's cfg.yaml,
# and the PPO eval env is built with the same value). Each output dir is
# self-describing: stats.json["_config"]["distgen_drift_std"] records what ran.
#
# Output goes to a drift-tagged dir so the static baseline in
# logs/compare_diff_hifi/ is left untouched. Re-running resumes: any
# (algo, seed) whose run dir already has a .done marker is skipped.
#
# ==========================================================================
#  SET THE DRIFT HERE (or override with DISTGEN_DRIFT_STD=... / --drift N).
#  Units: std of the normalized [0,1] distgen step, per env step.
#  0.0 reproduces the static baseline; 0.01-0.05 is a light-to-moderate jitter
#  (the unit tests exercise 0.05 / 0.1).
# ==========================================================================
DISTGEN_DRIFT_STD="${DISTGEN_DRIFT_STD:-0.02}"

set -euo pipefail

# --- tunables (override via env or flags) ------------------------------------
GPU="${GPU:-0}"
DEVICE="${DEVICE:-}"                 # set non-empty to skip the CUDA_VISIBLE pin
BUDGET="${BUDGET:-500000}"
SEEDS="${SEEDS:-0,1,2}"
ALGOS="${ALGOS:-ppo,shac,bptt}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-256}"
OUT_DIR="${OUT_DIR:-}"               # default derived from the drift value
SKIP_TRAIN=0

usage() {
    sed -n '2,33p' "$0"
    cat <<EOF

Usage: scripts/compare_hifi_drift.sh [flags]

  --drift N            distgen_drift_std (overrides DISTGEN_DRIFT_STD; default 0.02)
  --gpu N              pin to physical GPU N via CUDA_VISIBLE_DEVICES (default 0)
  --device DEV         explicit torch device (e.g. cuda:0, cpu); skips the GPU pin
  --budget N           env-step budget per (algo, seed) (default 500000)
  --seeds a,b,c        comma-separated seeds (default 0,1,2)
  --algos a,b          subset of ppo,shac,bptt (default all three)
  --eval-rollouts N    deterministic eval rollouts per run (default 256)
  --out-dir DIR        output dir (default logs/compare_diff_hifi_drift<DRIFT>)
  --skip-train         aggregate existing runs only (no subprocess training)
  -h | --help          this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --drift)         DISTGEN_DRIFT_STD="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --budget)        BUDGET="$2"; shift 2 ;;
        --seeds)         SEEDS="$2"; shift 2 ;;
        --algos)         ALGOS="$2"; shift 2 ;;
        --eval-rollouts) EVAL_ROLLOUTS="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --skip-train)    SKIP_TRAIN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Use the clean repo's package, not the (dead, legacy-pointing) editable
# install in slac-rl. Prepending src/ makes photoinjector_rl resolve here, and
# the compare_diff_algos training subprocesses inherit this PYTHONPATH.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/home/rwu4/miniconda3/envs/slac-rl/bin/python}"

# Default out-dir embeds the drift value so a sweep doesn't collide.
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="logs/compare_diff_hifi_drift${DISTGEN_DRIFT_STD}"
fi

# Pin one physical GPU unless an explicit --device was given.
if [[ -z "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    DEVICE="cuda:0"
fi

# --- locate surrogate artifacts ----------------------------------------------
CKPT="$(ls -1 trained/emittance_target_hifi/checkpoints/best-*.ckpt 2>/dev/null | tail -1 || true)"
NORM="processed/emittance_target_hifi_norm.json"

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[error] no hi-fi surrogate checkpoint under" \
         "trained/emittance_target_hifi/checkpoints/best-*.ckpt" >&2
    echo "        run scripts/regen_hifi.sh first." >&2
    exit 1
fi
if [[ ! -f "$NORM" ]]; then
    echo "[error] norm json missing: $NORM (run scripts/regen_hifi.sh first)" >&2
    exit 1
fi

# --- env sanity --------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "slac-rl" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}', expected 'slac-rl'." >&2
    echo "[warn] Using PYTHON=$PYTHON directly; activate slac-rl if imports fail." >&2
fi
"$PYTHON" -c "import torch, stable_baselines3, photoinjector_rl" || {
    echo "[error] required packages not importable (PYTHONPATH=$PYTHONPATH)." >&2
    exit 1
}

echo "=================================================================="
echo " hi-fi comparison  --  distgen_drift_std = ${DISTGEN_DRIFT_STD}"
echo "   device:          $DEVICE  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
echo "   algos:           $ALGOS"
echo "   seeds:           $SEEDS"
echo "   budget/run:      $BUDGET env-steps"
echo "   eval rollouts:   $EVAL_ROLLOUTS"
echo "   surrogate ckpt:  $CKPT"
echo "   out dir:         $OUT_DIR"
echo "   PYTHONPATH:      $PYTHONPATH"
echo "=================================================================="

CMD=( "$PYTHON" -m photoinjector_rl.emittance_target.compare_diff_algos
      --ckpt "$CKPT"
      --norm-json "$NORM"
      --out-dir "$OUT_DIR"
      --algos "$ALGOS"
      --seeds "$SEEDS"
      --budget "$BUDGET"
      --eval-rollouts "$EVAL_ROLLOUTS"
      --distgen-drift-std "$DISTGEN_DRIFT_STD"
      --device "$DEVICE" )
if [[ "$SKIP_TRAIN" -eq 1 ]]; then
    CMD+=( --skip-train )
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

echo
echo "=== drift comparison complete (distgen_drift_std=${DISTGEN_DRIFT_STD}) ==="
echo "  summary:   $OUT_DIR/summary.csv"
echo "  plots:     $OUT_DIR/compare.png  $OUT_DIR/rollouts_{shac,bptt}.png"
echo "  config:    $OUT_DIR/stats.json  (_config.distgen_drift_std)"
echo "  baseline:  logs/compare_diff_hifi/  (distgen_drift_std=0.0, untouched)"
