#!/usr/bin/env python3
"""Panel (a) experiment: Surrogate TI-EW and Column-Ind Chain EW across T.

(The horizon-free wrapper has its own trajectory experiment; see
scripts/run_panel_c.py.)

Usage:
    python run_panel_a.py --m 8 --seeds 3 --workers 4 --n-t 6 --c-max 5
    python run_panel_a.py --m 20 --seeds 100 --workers 220 --c-max 12
"""

# CRITICAL: oversubscription guards MUST come before numpy import
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Make src/ importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.theory import psi_surr, psi_col
from src.algorithms import SurrogateTIEW, ColumnIndEW


def f_star(i, j, m):
    if m <= 1:
        return 0.5
    return 0.25 + (i + j) / (4.0 * (m - 1))


def run_single(task):
    """Single job: (algo_name, m, T, K, seed) -> (algo_name, T, seed, regret).

    Must be at module top-level for ProcessPoolExecutor pickling.
    """
    algo_name, m, T, K, seed = task

    if algo_name == "surr":
        algo = SurrogateTIEW(m, K)
    elif algo_name == "col":
        algo = ColumnIndEW(m, K)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    rng = np.random.RandomState(seed)
    f_grid = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            f_grid[i, j] = f_star(i, j, m)

    loss_algo = 0.0
    loss_best = 0.0
    for t in range(T):
        i0 = rng.randint(m)
        j0 = rng.randint(m)
        p = f_grid[i0, j0]
        y = float(rng.rand() < p)
        yhat = algo.predict(i0, j0)
        loss_algo += (yhat - y) ** 2
        loss_best += (p - y) ** 2
        algo.update(i0, j0, y)

    return (algo_name, T, seed, loss_algo - loss_best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True,
                        help="Grid size m")
    parser.add_argument("--seeds", type=int, default=100,
                        help="Number of seeds")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--t-min", type=int, default=100,
                        help="Minimum T value")
    parser.add_argument("--c-max", type=float, default=12.0,
                        help="Maximum c = T/m^4")
    parser.add_argument("--n-t", type=int, default=15,
                        help="Number of T values (geomspace)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path")
    args = parser.parse_args()

    m = args.m
    t_max = int(args.c_max * m ** 4)
    T_values = np.unique(np.geomspace(args.t_min, t_max, args.n_t).astype(int))
    T_values = T_values[T_values >= 1]

    if args.output is None:
        args.output = f"panel_a_m{m}.npz"

    print(f"=== Panel (a) experiment ===", flush=True)
    print(f"  m = {m}", flush=True)
    print(f"  T range: [{T_values[0]}, {T_values[-1]}] ({len(T_values)} values)",
          flush=True)
    print(f"  c range: [{T_values[0]/m**4:.4f}, {T_values[-1]/m**4:.2f}]",
          flush=True)
    print(f"  Seeds: {args.seeds}", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    print(f"  Output: {args.output}", flush=True)
    print(flush=True)

    # Build task list
    tasks = []
    K_info = {}
    for T in T_values:
        T = int(T)
        _, K_surr = psi_surr(m, T)
        _, K_col = psi_col(m, T)
        K_surr = max(K_surr, 1)
        K_col = max(K_col, 1)
        K_info[T] = (K_surr, K_col)
        for seed in range(args.seeds):
            tasks.append(("surr", m, T, K_surr, seed))
            tasks.append(("col", m, T, K_col, seed))

    print(f"Total tasks: {len(tasks)}", flush=True)
    print(f"Per-T optimal K: "
          f"{ {int(T): K_info[int(T)] for T in T_values[:5]} } ...", flush=True)
    print(flush=True)

    # Parallel execution
    t_start = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single, t) for t in tasks]
        n_done = 0
        total = len(tasks)
        report_every = max(1, total // 20)
        for future in as_completed(futures):
            algo_name, T, seed, regret = future.result()
            key = (algo_name, T)
            results.setdefault(key, []).append((seed, regret))
            n_done += 1
            if n_done % report_every == 0 or n_done == total:
                elapsed = time.time() - t_start
                eta = elapsed * (total - n_done) / max(n_done, 1)
                print(f"  Progress: {n_done}/{total} "
                      f"({100*n_done/total:.1f}%), "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
                      flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.2f} min)",
          flush=True)

    # Organise by seed
    save_dict = {}
    save_dict["T_values"] = np.array(T_values, dtype=int)
    save_dict["m"] = np.array([m])
    save_dict["n_seeds"] = np.array([args.seeds])
    save_dict["K_surr"] = np.array([K_info[int(T)][0] for T in T_values])
    save_dict["K_col"] = np.array([K_info[int(T)][1] for T in T_values])

    for (algo_name, T), seed_regrets in results.items():
        seed_regrets.sort(key=lambda x: x[0])
        regrets_arr = np.array([r for _, r in seed_regrets], dtype=np.float64)
        save_dict[f"{algo_name}_{T}"] = regrets_arr

    np.savez(args.output, **save_dict)
    print(f"Saved to {args.output}", flush=True)

    # Summary
    print("\n=== Summary ===", flush=True)
    print(f"{'T':>10} {'c':>8} | {'K_s':>4} {'K_c':>4} | "
          f"{'surr':>10} {'col':>10}", flush=True)
    print("-" * 70, flush=True)
    for T in T_values:
        T = int(T)
        c = T / m ** 4
        line = f"  {T:>8d} {c:>8.4f} | "
        K_s, K_c = K_info[T]
        line += f"{K_s:>4d} {K_c:>4d} | "
        for algo in ["surr", "col"]:
            data = save_dict.get(f"{algo}_{T}", np.array([]))
            if len(data) > 0:
                line += f"{np.mean(data):>10.2f} "
            else:
                line += f"{'---':>10} "
        print(line, flush=True)


if __name__ == "__main__":
    main()
