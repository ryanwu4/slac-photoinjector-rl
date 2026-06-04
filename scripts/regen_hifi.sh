#!/usr/bin/env bash
# Regenerate the high-fidelity (20k particles, 32^3 mesh) dataset and retrain
# the EmittanceMLP surrogate end-to-end.
#
# Stages:
#   1. LHS sweep         -> archives/train_hifi/*.h5         (resumable)
#   2. Preprocess        -> processed/emittance_target_hifi.h5 + _norm.json
#   3. Surrogate train   -> trained/emittance_target_hifi/checkpoints/best-*.ckpt
#
# Step 4 (BPTT/SHAC/PPO comparison) is intentionally NOT in this script -- run
# compare_diff_algos.py manually after eyeballing the val_pred_vs_true plot.
#
# Each stage is idempotent: re-running the script after a kill or reboot
# resumes the sweep (via xopt dump_train_hifi.yaml) and skips already-finished
# downstream stages unless the matching --force-* flag is passed.

set -euo pipefail
cd "$(dirname "$0")/.."

SWEEP_CONFIG="configs/sweep/lhs_train_hifi.yaml"
ARCHIVE_DIR="archives/train_hifi"
PROCESSED_H5="processed/emittance_target_hifi.h5"
PROCESSED_NORM="processed/emittance_target_hifi_norm.json"
TRAIN_OUT="trained/emittance_target_hifi"
EXPECTED_ARCHIVES=10000

SKIP_SWEEP=0
FORCE_SWEEP=0
FORCE_PREPROCESS=0
FORCE_TRAIN=0
ASSUME_YES=0

usage() {
    cat <<EOF
Usage: scripts/regen_hifi.sh [flags]

  --skip-sweep         skip the LHS sweep (use existing archives/train_hifi/*.h5)
  --force-sweep        re-run sweep even if archive count >= ${EXPECTED_ARCHIVES}
  --force-preprocess   re-run preprocess even if ${PROCESSED_H5} exists
  --force-train        re-run training even if ${TRAIN_OUT}/final_metrics.json exists
  --yes                skip the disk/wall-clock confirmation prompt
  --mpi-ranks N        passed through as MPI_RANKS to run_sweep_local.sh (default 49)
  -h | --help          this message

Environment:
  MPI_RANKS            same as --mpi-ranks; --mpi-ranks wins if both set
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-sweep)        SKIP_SWEEP=1 ;;
        --force-sweep)       FORCE_SWEEP=1 ;;
        --force-preprocess)  FORCE_PREPROCESS=1 ;;
        --force-train)       FORCE_TRAIN=1 ;;
        --yes)               ASSUME_YES=1 ;;
        --mpi-ranks)         MPI_RANKS="$2"; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

export MPI_RANKS="${MPI_RANKS:-49}"

# --- 1. env sanity check ------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "slac-rl" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}', expected 'slac-rl'." >&2
    echo "[warn] Activate with: conda activate slac-rl" >&2
fi
python -c "import xopt, distgen, impact, lightning, h5py, beamphysics" || {
    echo "[error] required Python packages not importable in current env." >&2
    exit 1
}

mkdir -p "$ARCHIVE_DIR" processed "$TRAIN_OUT" workdir

# --- 2. user confirmation -----------------------------------------------------
if [[ "$ASSUME_YES" -ne 1 && "$SKIP_SWEEP" -ne 1 ]]; then
    cat <<EOF

  About to launch a 10,000-sample Impact-T sweep at 20k particles / 32^3 mesh.
  Expected footprint:  ~60-80 GB on disk in ${ARCHIVE_DIR}/
  Expected wall clock: multi-day on local 49-rank MPI (calibrate with
                       configs/sweep/lhs_smoke_hifi.yaml first).
  Sweep is resumable via dump_train_hifi.yaml.

  Press Enter to continue, or Ctrl-C to abort.
EOF
    read -r _
fi

# --- 3. sweep -----------------------------------------------------------------
n_archives() {
    # find is used (rather than ls *.h5) so an empty dir doesn't trip
    # `set -euo pipefail`. -maxdepth 1 mirrors a non-recursive ls.
    find "${ARCHIVE_DIR}" -maxdepth 1 -name '*.h5' -printf '.' 2>/dev/null | wc -c
}

# Background progress watcher. Polls the archive directory every POLL_SECS and
# prints "[sweep-progress]" lines with current count, rate, and ETA. Started
# just before the xopt sweep launches and stopped via trap afterwards.
POLL_SECS="${SWEEP_POLL_SECS:-30}"

start_progress_watcher() {
    local start_count="$1"
    local target="$2"
    local start_ts; start_ts=$(date +%s)
    (
        while true; do
            sleep "$POLL_SECS"
            local now cur done_count elapsed rate remaining eta_str
            now=$(date +%s)
            cur=$(n_archives)
            done_count=$(( cur - start_count ))
            elapsed=$(( now - start_ts ))
            if (( done_count > 0 && elapsed > 0 )); then
                rate=$(awk -v d="$done_count" -v e="$elapsed" 'BEGIN{printf "%.2f", d*60.0/e}')
                remaining=$(( target - cur ))
                if (( remaining > 0 )); then
                    eta_sec=$(awk -v r="$remaining" -v d="$done_count" -v e="$elapsed" 'BEGIN{printf "%d", r*e/d}')
                    eta_str=$(printf "%dh%02dm" $(( eta_sec/3600 )) $(( (eta_sec%3600)/60 )))
                else
                    eta_str="0h00m"
                fi
            else
                rate="--"
                eta_str="--"
            fi
            printf "[sweep-progress] %s  %d/%d (+%d this run, %s/min, ETA %s)\n" \
                "$(date '+%H:%M:%S')" "$cur" "$target" "$done_count" "$rate" "$eta_str"
        done
    ) &
    PROGRESS_PID=$!
}

stop_progress_watcher() {
    if [[ -n "${PROGRESS_PID:-}" ]] && kill -0 "$PROGRESS_PID" 2>/dev/null; then
        kill "$PROGRESS_PID" 2>/dev/null || true
        wait "$PROGRESS_PID" 2>/dev/null || true
    fi
    PROGRESS_PID=""
}

CURRENT=$(n_archives)
if [[ "$SKIP_SWEEP" -eq 1 ]]; then
    echo "[sweep] --skip-sweep set; current archive count: ${CURRENT}"
elif [[ "$FORCE_SWEEP" -ne 1 && "$CURRENT" -ge "$EXPECTED_ARCHIVES" ]]; then
    echo "[sweep] archive count ${CURRENT} >= ${EXPECTED_ARCHIVES}; skipping (--force-sweep to override)"
else
    echo "[sweep] launching: MPI_RANKS=${MPI_RANKS} scripts/run_sweep_local.sh ${SWEEP_CONFIG}"
    echo "[sweep] current archive count before sweep: ${CURRENT}"
    echo "[sweep] progress polled every ${POLL_SECS}s (override with SWEEP_POLL_SECS)"
    trap 'stop_progress_watcher' EXIT INT TERM
    start_progress_watcher "$CURRENT" "$EXPECTED_ARCHIVES"
    scripts/run_sweep_local.sh "${SWEEP_CONFIG}"
    stop_progress_watcher
    trap - EXIT INT TERM
    AFTER=$(n_archives)
    echo "[sweep] archive count after sweep: ${AFTER}"
fi

# --- 4. preprocess ------------------------------------------------------------
if [[ "$FORCE_PREPROCESS" -ne 1 && -f "$PROCESSED_H5" ]]; then
    echo "[preprocess] ${PROCESSED_H5} exists; skipping (--force-preprocess to override)"
else
    echo "[preprocess] building ${PROCESSED_H5} from ${ARCHIVE_DIR}/*.h5"
    python -m photoinjector_rl.emittance_target.preprocess \
        --archives "${ARCHIVE_DIR}/*.h5" \
        --out "$PROCESSED_H5"
fi

if [[ ! -f "$PROCESSED_NORM" ]]; then
    echo "[error] preprocess did not produce ${PROCESSED_NORM}" >&2
    exit 1
fi

# --- 5. surrogate train -------------------------------------------------------
FINAL_METRICS="${TRAIN_OUT}/final_metrics.json"
if [[ "$FORCE_TRAIN" -ne 1 && -f "$FINAL_METRICS" ]]; then
    echo "[train] ${FINAL_METRICS} exists; skipping (--force-train to override)"
else
    echo "[train] training EmittanceMLP -> ${TRAIN_OUT}"
    # --devices 1 forces single-GPU/CPU. Lightning's "auto" can pick multi-GPU
    # DDP on this box, which deadlocks a small MLP run on the log_dir broadcast
    # (NCCL watchdog kills it after ~30 min). The MLP is tiny -- single device
    # is faster anyway.
    python -m photoinjector_rl.emittance_target.train \
        --processed "$PROCESSED_H5" \
        --out-dir "$TRAIN_OUT" \
        --devices 1
fi

# --- 6. summary ---------------------------------------------------------------
BEST_CKPT=$(ls -1 "${TRAIN_OUT}/checkpoints/"best-*.ckpt 2>/dev/null | tail -1 || true)

echo
echo "=== high-fi pipeline complete ==="
echo "  archives:       ${ARCHIVE_DIR}/  ($(n_archives) files)"
echo "  processed:      ${PROCESSED_H5}"
echo "  norm json:      ${PROCESSED_NORM}"
echo "  train out:      ${TRAIN_OUT}/"
echo "  best ckpt:      ${BEST_CKPT:-<not found>}"
if [[ -f "$FINAL_METRICS" ]]; then
    python - <<PY
import json
with open("${FINAL_METRICS}") as f:
    m = json.load(f)
for k in ("val_loss", "val_r2", "val_mape_pct", "val_mae_phys_m2", "best_val_loss"):
    if k in m:
        print(f"  {k:<20s} {m[k]}")
PY
fi

cat <<EOF

Next step (manual): run the BPTT/SHAC/PPO comparison against this surrogate.

  python -m photoinjector_rl.emittance_target.compare_diff_algos \
      --ckpt "${BEST_CKPT}" \
      --norm-json ${PROCESSED_NORM} \
      --out-dir logs/compare_diff_hifi \
      --algos ppo,shac,bptt \
      --seeds 0,1,2 \
      --budget 500000 \
      --device cuda:0

EOF

