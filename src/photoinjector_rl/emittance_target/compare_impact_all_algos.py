"""
Orchestrate a head-to-head Impact-T evaluation of trained PPO, SHAC, and
BPTT policies (no fine-tuning).

Workflow:
  1. Read `<compare-dir>/summary.csv` (produced by compare_diff_algos).
  2. For each algo, pick the surrogate-training seed with the lowest
     `terminal_emit_median_m2` — i.e., the seed that was best on the
     surrogate. Tie-break by seed index.
  3. Resolve the policy file path for that (algo, seed):
        ppo  → <run_dir>/ppo_final.zip  (fallback: <run_dir>/eval/best_model.zip)
        shac → <run_dir>/final_policy.pt (fallback: <run_dir>/best_policy.pt)
        bptt → same as shac
  4. Hand `--policy LABEL=PATH` triples to `compare_n_impact.main()` so the
     existing N-way Impact-T comparator does the work.

Usage:
    python -m photoinjector_rl.emittance_target.compare_impact_all_algos \\
        --compare-dir logs/compare_diff \\
        --impact-config configs/impact/ImpactT_config.yaml \\
        --distgen-input configs/impact/distgen_template.yaml \\
        --norm-json processed/emittance_target_norm.json \\
        --out logs/compare_diff_impact/all_algos \\
        --n-samples 20 --n-workers 8 --max-steps 64
"""
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import sysconfig
from pathlib import Path

from . import compare_n_impact


def _ensure_impact_binary() -> None:
    """If `IMPACTT_BIN` is unset and `ImpactTexe` isn't on PATH, try to find
    it in the active Python's conda environment and export it.

    Running `<env>/bin/python -m foo` does NOT automatically put `<env>/bin`
    on PATH the way `conda activate` would. Without this fix, Impact-T fails
    with `ValueError: Could not find executable: exename=ImpactTexe,
    envname=IMPACTT_BIN`.
    """
    if os.environ.get("IMPACTT_BIN"):
        return
    if shutil.which("ImpactTexe"):
        return
    bin_dir = Path(sysconfig.get_path("scripts"))
    candidate = bin_dir / "ImpactTexe"
    if candidate.exists() and os.access(candidate, os.X_OK):
        os.environ["IMPACTT_BIN"] = str(candidate)
        print(f"[compare_impact_all_algos] auto-set IMPACTT_BIN={candidate}")


def _pick_best_seeds(summary_csv: Path) -> dict[str, dict]:
    """algo -> dict(seed=int, run_dir=str, median=float) for the lowest
    `terminal_emit_median_m2` row of that algo. Tie-break by seed index.
    """
    rows_by_algo: dict[str, list[dict]] = {}
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            rows_by_algo.setdefault(row["algo"], []).append(row)
    best: dict[str, dict] = {}
    for algo, rows in rows_by_algo.items():
        winner = min(
            rows,
            key=lambda r: (float(r["terminal_emit_median_m2"]),
                           int(r["seed"])),
        )
        best[algo] = {
            "seed": int(winner["seed"]),
            "run_dir": Path(winner["run_dir"]),
            "median": float(winner["terminal_emit_median_m2"]),
        }
    return best


def _policy_path_for(algo: str, run_dir: Path) -> Path:
    if algo == "ppo":
        candidates = [run_dir / "ppo_final.zip",
                      run_dir / "eval" / "best_model.zip"]
    elif algo in {"shac", "bptt"}:
        candidates = [run_dir / "final_policy.pt",
                      run_dir / "best_policy.pt"]
    else:
        raise ValueError(f"unknown algo {algo!r}")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no policy file under {run_dir} for algo={algo}; "
        f"tried {[str(c) for c in candidates]}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare-dir", required=True, type=Path,
                   help="directory containing summary.csv from compare_diff_algos")
    p.add_argument("--impact-config", required=True)
    p.add_argument("--distgen-input", required=True)
    p.add_argument("--norm-json", required=True)
    p.add_argument("--out", required=True,
                   help="output stem; compare_n_impact writes <out>.png + .npz")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--n-workers", type=int, default=0,
                   help="0 = auto (min(cpu_count, total_tasks))")
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--algos", default="ppo,shac,bptt",
                   help="comma-separated subset of {ppo,shac,bptt}")
    p.add_argument("--constants-yaml", default=None,
                   help="sweep YAML (e.g. configs/sweep/lhs_train_hifi.yaml) "
                        "whose vocs.constants block pins Impact-T fidelity. "
                        "Forwarded to compare_n_impact.")
    args = p.parse_args()
    _ensure_impact_binary()

    summary_csv = args.compare_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"missing {summary_csv}; run compare_diff_algos first")
    best = _pick_best_seeds(summary_csv)

    wanted = [a.strip() for a in args.algos.split(",") if a.strip()]
    missing = [a for a in wanted if a not in best]
    if missing:
        raise SystemExit(f"summary.csv has no rows for algos: {missing}")

    policy_args: list[str] = []
    print(f"[compare_impact_all_algos] best surrogate-seed per algo "
          f"(from {summary_csv}):")
    for algo in wanted:
        entry = best[algo]
        path = _policy_path_for(algo, entry["run_dir"])
        print(f"  {algo:>4s}  seed={entry['seed']:>2d}  "
              f"surrogate-median={entry['median']:.3e} m²  "
              f"path={path}")
        policy_args.extend(["--policy", f"{algo}={path}"])

    # Build argv for compare_n_impact.main() — splice our --policy list and
    # forward the pass-through args.
    forwarded = [
        "--impact-config", args.impact_config,
        "--distgen-input", args.distgen_input,
        "--norm-json", args.norm_json,
        "--out", args.out,
        "--n-samples", str(args.n_samples),
        "--n-workers", str(args.n_workers),
        "--max-steps", str(args.max_steps),
        "--seed", str(args.seed),
    ]
    if args.constants_yaml:
        forwarded += ["--constants-yaml", args.constants_yaml]
    sys.argv = [sys.argv[0]] + policy_args + forwarded
    print(f"[compare_impact_all_algos] handing off to compare_n_impact "
          f"with argv: {sys.argv[1:]}")
    compare_n_impact.main()


if __name__ == "__main__":
    main()
