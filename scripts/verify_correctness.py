#!/usr/bin/env python3
"""Full verification: DP vs brute-force across multiple (m, K) configurations.

For each (m, K, T) config, runs Surrogate TI-EW and Column-Ind Chain EW
round-by-round under the main-experiment adversary (Bernoulli labels with
linear isotonic mean), compares against brute-force enumeration of
generalised experts, and records the maximum absolute discrepancy.

Saves results to results/correctness.npz; the table in Appendix A of the
paper is read off from this file.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.algorithms import (  # noqa: E402
    ColumnIndEW,
    SurrogateTIEW,
    brute_force_column_ew,
    brute_force_surrogate_ew,
)


CONFIGS = [
    # (m, K, T)
    (3, 2, 200),
    (3, 3, 200),
    (3, 4, 200),
    (4, 2, 200),
    (4, 3, 200),
    (5, 2, 200),
    (5, 3, 200),
    (6, 2, 200),
]


def f_star(i, j, m):
    """Linear isotonic mean used as adversary."""
    return 0.25 + (i + j) / (4.0 * (m - 1)) if m > 1 else 0.5


def run_dp(algo_cls, m, K, queries, labels):
    """Run a DP-based algorithm round-by-round; return predictions array."""
    algo = algo_cls(m, K)
    T = len(queries)
    preds = np.zeros(T)
    for t in range(T):
        i, j = int(queries[t, 0]), int(queries[t, 1])
        preds[t] = algo.predict(i, j)
        algo.update(i, j, float(labels[t]))
    return preds


def verify(m, K, T, seed):
    rng = np.random.default_rng(seed)
    queries = rng.integers(0, m, size=(T, 2))
    means = np.array([f_star(int(i), int(j), m) for i, j in queries])
    labels = (rng.uniform(0, 1, size=T) < means).astype(np.float64)

    preds_surr_dp = run_dp(SurrogateTIEW, m, K, queries, labels)
    preds_surr_bf = brute_force_surrogate_ew(m, K, queries, labels)

    preds_col_dp = run_dp(ColumnIndEW, m, K, queries, labels)
    preds_col_bf = brute_force_column_ew(m, K, queries, labels)

    return {
        "surr_max_disc": float(np.max(np.abs(preds_surr_dp - preds_surr_bf))),
        "col_max_disc":  float(np.max(np.abs(preds_col_dp  - preds_col_bf))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=1e-9)
    parser.add_argument("--output", type=str,
                        default="results/correctness.npz")
    parser.add_argument("--max-m", type=int, default=None,
                        help="Only run configs with m <= this value (filter)")
    parser.add_argument("--max-K", type=int, default=None,
                        help="Only run configs with K <= this value (filter)")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [(m, K, T) for (m, K, T) in CONFIGS
               if (args.max_m is None or m <= args.max_m)
               and (args.max_K is None or K <= args.max_K)]

    print(f"Running {len(configs)} verification configs (seed = {args.seed})...",
          flush=True)
    print()

    results = []
    total_start = time.time()
    for (m, K, T) in configs:
        print(f"  (m, K, T) = ({m}, {K}, {T}): running ...", flush=True)
        start = time.time()
        res = verify(m, K, T, args.seed)
        elapsed = time.time() - start
        res.update({"m": m, "K": K, "T": T, "seed": args.seed, "elapsed": elapsed})
        res["pass_surr"] = res["surr_max_disc"] < args.threshold
        res["pass_col"] = res["col_max_disc"] < args.threshold
        results.append(res)
        print(f"    surr: {res['surr_max_disc']:.3e} "
              f"{'PASS' if res['pass_surr'] else 'FAIL'};  "
              f"col: {res['col_max_disc']:.3e} "
              f"{'PASS' if res['pass_col'] else 'FAIL'};  "
              f"time: {elapsed:.1f}s",
              flush=True)
    total_elapsed = time.time() - total_start

    np.savez(
        output_path,
        configs=np.array([(r["m"], r["K"], r["T"]) for r in results]),
        surr_max_disc=np.array([r["surr_max_disc"] for r in results]),
        col_max_disc=np.array([r["col_max_disc"] for r in results]),
        elapsed=np.array([r["elapsed"] for r in results]),
        seed=np.array([args.seed]),
    )
    print(flush=True)
    print(f"Saved to {output_path}")

    print()
    print(f"{'m':>3} {'K':>3} {'T':>4} {'surr_max':>12} {'col_max':>12} {'time':>9}")
    print("-" * 50)
    for r in results:
        print(f"{r['m']:>3} {r['K']:>3} {r['T']:>4} "
              f"{r['surr_max_disc']:>12.3e} {r['col_max_disc']:>12.3e} "
              f"{r['elapsed']:>8.1f}s")
    print(f"Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
