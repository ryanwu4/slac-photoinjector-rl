#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/sweep/lhs_smoke_hifi.yaml}"

# 1 mpi4py.futures master + 48 workers (matches max_workers: 48 in the sweep YAMLs).
# Override with MPI_RANKS env var if running on a smaller box.
MPI_RANKS="${MPI_RANKS:-49}"

case "$CONFIG" in
    *smoke_hifi*) SUBDIR=smoke_hifi ;;
    *train_hifi*) SUBDIR=train_hifi ;;
    *val_hifi*)   SUBDIR=val_hifi ;;
    *smoke*)      SUBDIR=smoke ;;
    *train*)      SUBDIR=train ;;
    *val*)        SUBDIR=val ;;
    *)            SUBDIR=other ;;
esac

mkdir -p workdir "data/archives/${SUBDIR}"

# --oversubscribe lets the master share a core with a worker; harmless because
# the master is lightweight. Default (49 ranks on 64 physical cores) is well
# under capacity, but keep the flag in case MPI_RANKS is bumped past 64.
mpirun --oversubscribe -n "$MPI_RANKS" python -m mpi4py.futures -m xopt.mpi.run "$CONFIG"
