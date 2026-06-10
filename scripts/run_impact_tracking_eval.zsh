#!/usr/bin/env zsh
# ---------------------------------------------------------------------------
# Evaluate the trained moving-target (aspect,tilt) controllers on REAL Impact-T.
#
# Runs every (model x held-out trajectory) combination as a CLOSED-LOOP rollout:
# at each of the 64 steps the policy reacts to the previous Impact-T bunch, so a
# single rollout is ~65 Impact-T jobs run back-to-back (inherently sequential).
# The combinations are independent, so they fan out in parallel.
#
#   models       : shac (runs/move_shac/seed0), bptt (runs/move_bptt/seed0),
#                  ppo (runs/move_ppo)            -> --<algo> <run_dir>
#   trajectories : staircase, tilt_rotation, aspect_ramp   (held-out schedules)
#   fidelity     : lofi (2k particles, 8^3 mesh, ~9 s/run -> ~10 min/rollout)
#                  hifi (20k particles, 32^3 mesh, ~35 s/run -> ~40 min/rollout)
#
# "both" runs lofi first, then hifi (the lofi flow surrogate is matched to lofi
# Impact-T, so lofi isolates the surrogate->real-sim transfer; hifi is the
# higher-fidelity "real" number).
#
# Outputs (uniquely named per algo_traj_fidelity, no collisions):
#   figures/impact_tracking_<algo>_<traj>_<fid>.{png,json}
#   runs/impact_eval/<algo>_<traj>_<fid>/rollout.npz   (+ live rollout.partial.npz)
#   runs/impact_eval/impact_eval_<algo>_<traj>_<fid>.log           (per-run console)
#
# Usage:
#   ./scripts/run_impact_tracking_eval.zsh                 # lofi then hifi (9+9 parallel)
#   ./scripts/run_impact_tracking_eval.zsh --fidelity lofi # lofi only
#   ./scripts/run_impact_tracking_eval.zsh --fidelity hifi --parity
#   ./scripts/run_impact_tracking_eval.zsh --dry-run       # print commands, run nothing
#
# --parity  : add a flow-vs-Impact-T shape-parity pre-check to ONE job per phase
#             (writes figures/impact_parity_<fid>.png). Worth it for hifi.
# ---------------------------------------------------------------------------
set -o pipefail

# --- config -----------------------------------------------------------------
REPO=${0:A:h:h}
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
CKPT="models/flow_surrogate/checkpoints/best-epoch=493-val_loss=-0.9555.ckpt"
NORM=data/processed/flow_surrogate_norm.json

export IMPACTT_BIN=/home/rwu4/miniconda3/envs/slac-rl/bin/ImpactTexe
export OMP_NUM_THREADS=1        # one thread per process -> no CPU oversubscription
export PYTHONPATH=$REPO/src

# model label -> policy run dir (the CLI flag is --<label>)
typeset -A MODELS=(
  shac runs/move_shac/seed0
  bptt runs/move_bptt/seed0
  ppo  runs/move_ppo
)
TRAJS=(staircase tilt_rotation aspect_ramp)

# --- args -------------------------------------------------------------------
FIDELITY=both        # lofi | hifi | both
EPISODE=64
PARITY=0
DRYRUN=0
while (( $# )); do
  case $1 in
    --fidelity)       FIDELITY=$2; shift 2 ;;
    --episode-length) EPISODE=$2;  shift 2 ;;
    --parity)         PARITY=1;    shift   ;;
    --dry-run)        DRYRUN=1;    shift   ;;
    -h|--help)        sed -n '2,40p' $0;   exit 0 ;;
    *) print -u2 "unknown arg: $1"; exit 1 ;;
  esac
done

cd $REPO || { print -u2 "cannot cd $REPO"; exit 1; }
mkdir -p runs figures runs/impact_eval

# --- one fidelity phase: fan out all models x trajectories ------------------
run_phase () {
  local fid=$1
  local -a pids labels
  local first=$PARITY            # give the parity pre-check to the first job only
  print "=== $fid phase: launching ${#MODELS} models x ${#TRAJS} trajectories ==="
  for algo dir in ${(kv)MODELS}; do
    for traj in $TRAJS; do
      local log="runs/impact_eval/impact_eval_${algo}_${traj}_${fid}.log"
      local -a cmd=(
        $PY -m photoinjector_rl.surrogates.flow.impact_eval_tracking
        --flow-ckpt "$CKPT" --norm-json "$NORM"
        --$algo "$dir" --which-traj "$traj" --fidelity "$fid"
        --episode-length $EPISODE
      )
      (( first )) && { cmd+=(--parity-check); first=0 }
      if (( DRYRUN )); then
        print "  [dry-run] ${cmd[*]}  > $log 2>&1 &"
      else
        "${cmd[@]}" > "$log" 2>&1 &
        pids+=($!); labels+=("${algo}_${traj}_${fid}")
        print "  launched ${algo}/${traj} (pid $!) -> $log"
      fi
    done
  done
  (( DRYRUN )) && return 0

  local i fail=0
  for i in {1..$#pids}; do
    if wait $pids[$i]; then
      print "  [ok]   $labels[$i]"
    else
      print "  [FAIL] $labels[$i]  (tail runs/impact_eval/impact_eval_$labels[$i].log)"
      (( fail++ ))
    fi
  done
  print "=== $fid phase done: $(( ${#pids} - fail ))/${#pids} ok, $fail failed ==="
}

# --- run --------------------------------------------------------------------
case $FIDELITY in
  lofi) time run_phase lofi ;;
  hifi) time run_phase hifi ;;
  both) time run_phase lofi; print; time run_phase hifi ;;
  *) print -u2 "--fidelity must be lofi|hifi|both"; exit 1 ;;
esac
print "ALL DONE ($FIDELITY)"
