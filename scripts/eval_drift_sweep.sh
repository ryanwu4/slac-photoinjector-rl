#!/usr/bin/env bash
# Eval-only distgen-drift robustness sweep.
#
# Takes the CLEAN-trained (distgen_drift_std=0) PPO/SHAC/BPTT policies already
# sitting in runs/compare_diff_hifi/ and re-evaluates them under several jitter
# levels -- NO retraining. This is the test-time-robustness complement to
# compare_hifi_drift.sh (which trains AND evals under drift).
#
# The drift=0.0 level reproduces runs/compare_diff_hifi/summary.csv (same eval
# helpers), so it doubles as a sanity check. The source run dir is read-only;
# all outputs land in --out-dir.
#
# ==========================================================================
#  SET THE JITTER LEVELS HERE (or override with DRIFT_LEVELS=... / --levels).
#  Comma-separated normalized-[0,1] distgen step stds. Keep 0.0 as the
#  clean reference column.
# ==========================================================================
DRIFT_LEVELS="${DRIFT_LEVELS:-0.0,0.02,0.05,0.1}"

set -euo pipefail

# --- tunables (override via env or flags) ------------------------------------
GPU="${GPU:-0}"
DEVICE="${DEVICE:-}"
RUNS_ROOT="${RUNS_ROOT:-runs/compare_diff_hifi}"
OUT_DIR="${OUT_DIR:-results/eval_drift_sweep}"
ALGOS="${ALGOS:-ppo,shac,bptt}"
SEEDS="${SEEDS:-0,1,2}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-256}"
PPO_EVAL_ENV="${PPO_EVAL_ENV:-gym}"   # gym | diff (diff = same env as SHAC/BPTT)

usage() {
    sed -n '2,30p' "$0"
    cat <<EOF

Usage: scripts/eval_drift_sweep.sh [flags]

  --levels a,b,c       eval drift stds (overrides DRIFT_LEVELS; default 0.0,0.02,0.05,0.1)
  --gpu N              pin to physical GPU N (default 0)
  --device DEV         explicit torch device (cuda:0, cpu); skips the GPU pin
  --runs-root DIR      trained-policy root (default runs/compare_diff_hifi)
  --out-dir DIR        output dir (default results/eval_drift_sweep)
  --algos a,b          subset of ppo,shac,bptt (default all three)
  --seeds a,b,c        comma-separated seeds (default 0,1,2)
  --eval-rollouts N    rollouts per (algo,seed,drift) (default 256)
  -h | --help          this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --levels)        DRIFT_LEVELS="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --runs-root)     RUNS_ROOT="$2"; shift 2 ;;
        --out-dir)       OUT_DIR="$2"; shift 2 ;;
        --algos)         ALGOS="$2"; shift 2 ;;
        --seeds)         SEEDS="$2"; shift 2 ;;
        --eval-rollouts) EVAL_ROLLOUTS="$2"; shift 2 ;;
        --ppo-eval-env)  PPO_EVAL_ENV="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Use the clean repo's package, not the dead legacy editable install.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/home/rwu4/miniconda3/envs/slac-rl/bin/python}"

if [[ -z "$DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    DEVICE="cuda:0"
fi

CKPT="$(ls -1 models/emittance_target_hifi/checkpoints/best-*.ckpt 2>/dev/null | tail -1 || true)"
NORM="data/processed/emittance_target_hifi_norm.json"
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
    echo "[error] no hi-fi surrogate checkpoint under models/emittance_target_hifi/checkpoints/" >&2
    exit 1
fi
if [[ ! -d "$RUNS_ROOT" ]]; then
    echo "[error] runs root not found: $RUNS_ROOT (run the baseline comparison first)" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "slac-rl" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}', expected 'slac-rl'." >&2
fi
"$PYTHON" -c "import torch, stable_baselines3, photoinjector_rl" || {
    echo "[error] required packages not importable (PYTHONPATH=$PYTHONPATH)." >&2
    exit 1
}

echo "=================================================================="
echo " eval-only drift sweep  (clean-trained policies, jitter at eval)"
echo "   drift levels:    $DRIFT_LEVELS"
echo "   device:          $DEVICE  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})"
echo "   policies from:   $RUNS_ROOT  (read-only)"
echo "   algos / seeds:   $ALGOS  /  $SEEDS"
echo "   eval rollouts:   $EVAL_ROLLOUTS"
echo "   surrogate ckpt:  $CKPT"
echo "   out dir:         $OUT_DIR"
echo "=================================================================="

"$PYTHON" -m photoinjector_rl.surrogates.mlp.eval_drift_sweep \
    --ckpt "$CKPT" --norm-json "$NORM" \
    --runs-root "$RUNS_ROOT" --out-dir "$OUT_DIR" \
    --algos "$ALGOS" --seeds "$SEEDS" \
    --drift-levels "$DRIFT_LEVELS" \
    --eval-rollouts "$EVAL_ROLLOUTS" \
    --ppo-eval-env "$PPO_EVAL_ENV" \
    --device "$DEVICE"

echo
echo "=== eval drift sweep complete ==="
echo "  summary:  $OUT_DIR/summary.csv"
echo "  plot:     $OUT_DIR/drift_sweep.png"
echo "  stats:    $OUT_DIR/stats.json"
echo "  source:   $RUNS_ROOT/  (untouched)"
