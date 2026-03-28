"""Compare outputs from two EigenBench runs (e.g. pairwise vs pointwise).

Usage:
    python scripts/compare_runs.py <run_dir_a> <run_dir_b> [--dim 2]

Example:
    python scripts/compare_runs.py \
        runs/pointwise_experiment/pointwise_experiment_pairwise \
        runs/pointwise_experiment/pointwise_experiment_pointwise \
        --dim 2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import stats


def load_eigentrust_scores(path: str) -> np.ndarray:
    """Load EigenTrust scores from eigentrust.txt."""
    with open(path) as f:
        lines = f.readlines()
    # First line is header "EigenTrust scores:", second line is the array
    scores_str = lines[1].strip().strip("[]")
    return np.fromstring(scores_str, sep=",")


def load_log(path: str) -> dict[str, float]:
    """Load training log from log_train.txt."""
    data: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            try:
                data[key.strip()] = float(value.strip())
            except ValueError:
                pass
    return data


def eigentrust_to_elo(trust_scores: np.ndarray, num_models: int) -> np.ndarray:
    """Convert EigenTrust scores to Elo scale (same formula as pipeline)."""
    trust_safe = np.clip(trust_scores, 1e-12, None)
    return 1500.0 + 400.0 * np.log10(num_models * trust_safe)


def main():
    parser = argparse.ArgumentParser(description="Compare two EigenBench runs.")
    parser.add_argument("run_a", help="Path to first run output directory")
    parser.add_argument("run_b", help="Path to second run output directory")
    parser.add_argument("--dim", type=int, default=2, help="BTD dimension to compare (default: 2)")
    args = parser.parse_args()

    dim_dir = f"btd_d{args.dim}"
    dir_a = os.path.join(args.run_a, dim_dir)
    dir_b = os.path.join(args.run_b, dim_dir)

    for d, label in [(dir_a, "Run A"), (dir_b, "Run B")]:
        if not os.path.isdir(d):
            print(f"Error: {label} directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Load data
    trust_a = load_eigentrust_scores(os.path.join(dir_a, "eigentrust.txt"))
    trust_b = load_eigentrust_scores(os.path.join(dir_b, "eigentrust.txt"))
    log_a = load_log(os.path.join(dir_a, "log_train.txt"))
    log_b = load_log(os.path.join(dir_b, "log_train.txt"))

    num_models = len(trust_a)
    if len(trust_b) != num_models:
        print(f"Error: model count mismatch ({len(trust_a)} vs {len(trust_b)})", file=sys.stderr)
        sys.exit(1)

    elo_a = eigentrust_to_elo(trust_a, num_models)
    elo_b = eigentrust_to_elo(trust_b, num_models)

    rank_a = np.argsort(-elo_a)  # Descending
    rank_b = np.argsort(-elo_b)

    # Compute metrics
    kendall_tau, kendall_p = stats.kendalltau(elo_a, elo_b)
    spearman_rho, spearman_p = stats.spearmanr(elo_a, elo_b)
    top1_match = rank_a[0] == rank_b[0]

    test_loss_a = log_a.get("test_loss", float("nan"))
    test_loss_b = log_b.get("test_loss", float("nan"))
    loss_ratio = test_loss_b / test_loss_a if test_loss_a > 0 else float("nan")

    # Print results
    print("=" * 60)
    print("EigenBench Run Comparison")
    print("=" * 60)
    print(f"Run A: {args.run_a}")
    print(f"Run B: {args.run_b}")
    print(f"Dimension: {args.dim}")
    print(f"Models: {num_models}")
    print()

    print("--- EigenTrust Scores ---")
    print(f"  Run A: {trust_a}")
    print(f"  Run B: {trust_b}")
    print()

    print("--- Elo Scores ---")
    print(f"  Run A: {np.round(elo_a, 1)}")
    print(f"  Run B: {np.round(elo_b, 1)}")
    print()

    print("--- Rankings (model indices, best first) ---")
    print(f"  Run A: {list(rank_a)}")
    print(f"  Run B: {list(rank_b)}")
    print()

    print("--- Pre-Registered Metrics ---")
    print()

    # Kendall's tau
    tau_pass = kendall_tau >= 0.80
    print(f"  Kendall's tau:    {kendall_tau:.4f}  (p={kendall_p:.4f})")
    print(f"    Threshold:      >= 0.80")
    print(f"    Result:         {'PASS' if tau_pass else 'FAIL'}")
    print()

    # Top-1 agreement
    print(f"  Top-1 agreement:  Run A={rank_a[0]}, Run B={rank_b[0]}")
    print(f"    Result:         {'PASS' if top1_match else 'FAIL'}")
    print()

    # Spearman rho
    rho_pass = spearman_rho >= 0.80
    print(f"  Spearman rho:     {spearman_rho:.4f}  (p={spearman_p:.4f})")
    print(f"    Threshold:      >= 0.80")
    print(f"    Result:         {'PASS' if rho_pass else 'FAIL'}")
    print()

    # Test loss ratio
    loss_pass = loss_ratio <= 2.0
    print(f"  Test loss A:      {test_loss_a:.6f}")
    print(f"  Test loss B:      {test_loss_b:.6f}")
    print(f"  Loss ratio (B/A): {loss_ratio:.4f}")
    print(f"    Threshold:      <= 2.0")
    print(f"    Result:         {'PASS' if loss_pass else 'FAIL'}")
    print()

    # Overall verdict
    all_pass = tau_pass and top1_match and rho_pass and loss_pass
    print("=" * 60)
    if all_pass:
        print("VERDICT: ALL METRICS PASS")
        print("Pointwise is a viable low-cost alternative for ranking.")
    elif top1_match and not (tau_pass and rho_pass):
        print("VERDICT: PARTIAL PASS (coarse ordering preserved)")
        print("Top model agrees but fine-grained rankings diverge.")
        print("Useful for screening, not final evaluation.")
    elif not top1_match:
        print("VERDICT: FAIL (top-1 disagrees)")
        print("Pointwise is not a reliable substitute in this setting.")
    else:
        print("VERDICT: PARTIAL FAIL")
        print("Some metrics pass, some fail. Review details above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
