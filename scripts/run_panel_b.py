#!/usr/bin/env python3
"""Generate data for Figure 3 Panel (b): regret ratio across m.

For each m in {8, 12, 18}, for each c in geomspace(0.03, 10, 15), for each algo
in {surr, col}, run N seeds in parallel.

Saves per-m checkpoint to results/panel_b_m{m}.npz after each m completes,
so interruption is recoverable.

Usage:
    python run_panel_b.py                    # 100 seeds per config (default)
    python run_panel_b.py --seeds 50         # custom seed count
    python run_panel_b.py --resume           # skip m's with existing checkpoint
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.theory import psi_surr, psi_col
from src.algorithms import SurrogateTIEW, ColumnIndEW


M_VALUES_DEFAULT = [8, 12, 18]
C_RANGE_DEFAULT = (0.03, 10.0)
N_C_DEFAULT = 15


def make_adversary(m):
    f = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            f[i, j] = 0.25 + (i + j) / (4.0 * (m - 1))
    return f


def run_single(algo_name, m, T, K, seed):
    rng = np.random.RandomState(seed)
    f_star = make_adversary(m)
    if algo_name == "surr":
        algo = SurrogateTIEW(m, K)
    else:
        algo = ColumnIndEW(m, K)
    regret = 0.0
    for t in range(T):
        i0 = rng.randint(m)
        j0 = rng.randint(m)
        p = f_star[i0, j0]
        y = float(rng.rand() < p)
        yhat = algo.predict(i0, j0)
        algo.update(i0, j0, y)
        regret += (yhat - y) ** 2 - (p - y) ** 2
    return regret


def get_T_values(m, c_values):
    raw = (np.asarray(c_values) * m ** 4).astype(int)
    raw = np.maximum(raw, 1)
    return sorted(set(int(t) for t in raw))


def run_m(m, c_values, n_seeds, n_jobs, ckpt_path):
    """Run all configs for a single m, return dict keyed by (algo, T)."""
    T_list = get_T_values(m, c_values)
    results = {}

    print(f"\n=== m={m}  ({len(T_list)} T values × 2 algos × {n_seeds} seeds) ===",
          flush=True)
    t_m_start = time.time()

    for T in T_list:
        c = T / m ** 4
        for algo_name, psi_func in [("surr", psi_surr), ("col", psi_col)]:
            _, K_opt = psi_func(m, T)
            K_opt = max(K_opt, 1)

            print(f"  m={m:>2d} {algo_name} T={T:>8d} c={c:>7.3f} K={K_opt:>3d} ...",
                  end="", flush=True)
            t0 = time.time()

            regrets = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_single)(algo_name, m, T, K_opt, seed)
                for seed in range(n_seeds)
            )
            regrets = np.array(regrets, dtype=np.float64)
            elapsed = time.time() - t0

            results[(algo_name, T)] = regrets
            mean = regrets.mean()
            sem = regrets.std() / np.sqrt(n_seeds)
            print(f"  R={mean:>8.2f}±{sem:>5.2f}  R/m²={mean/m**2:>6.3f}"
                  f"  ({elapsed:>6.1f}s)", flush=True)

    # Save per-m checkpoint
    save_dict = {}
    save_dict["n_seeds"] = np.array([n_seeds])
    save_dict["T_values"] = np.array(T_list)
    save_dict["K_surr"] = np.array([max(psi_surr(m, T)[1], 1) for T in T_list])
    save_dict["K_col"] = np.array([max(psi_col(m, T)[1], 1) for T in T_list])
    for (algo, T), regrets in results.items():
        save_dict[f"{algo}_{T}"] = regrets
    np.savez(ckpt_path, **save_dict)

    total_m = time.time() - t_m_start
    print(f"  [m={m} checkpoint saved: {ckpt_path}  total={total_m:.1f}s]",
          flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100,
                        help="Number of seeds per (m, algo, T) config")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel workers (-1 = all cores)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip m values that already have a checkpoint")
    parser.add_argument("--m-list", type=str, default=None,
                        help="Comma-separated m values "
                        f"(default: {M_VALUES_DEFAULT})")
    parser.add_argument("--c-min", type=float, default=C_RANGE_DEFAULT[0],
                        help=f"Minimum c = T/m^4 (default {C_RANGE_DEFAULT[0]})")
    parser.add_argument("--c-max", type=float, default=C_RANGE_DEFAULT[1],
                        help=f"Maximum c = T/m^4 (default {C_RANGE_DEFAULT[1]})")
    parser.add_argument("--n-c", type=int, default=N_C_DEFAULT,
                        help=f"Number of c values (default {N_C_DEFAULT})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for panel_b_m{m}.npz checkpoints "
                        "and the consolidated JSON (default: results/)")
    args = parser.parse_args()

    if args.m_list is None:
        m_values = list(M_VALUES_DEFAULT)
    else:
        m_values = [int(s) for s in args.m_list.split(",") if s.strip()]
    c_values = np.geomspace(args.c_min, args.c_max, args.n_c)

    # Oversubscription guards (required before joblib runs)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    n_jobs = args.jobs if args.jobs > 0 else os.cpu_count()
    if args.output_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"
    else:
        results_dir = Path(args.output_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60, flush=True)
    print(f"Panel (b) data generation", flush=True)
    print(f"=" * 60, flush=True)
    print(f"m_values: {m_values}", flush=True)
    print(f"c range: [{c_values[0]:.4f}, {c_values[-1]:.2f}], "
          f"{len(c_values)} pts", flush=True)
    print(f"Seeds: {args.seeds},  n_jobs: {n_jobs}", flush=True)
    print(f"Executor: joblib LokyBackend (process-based)", flush=True)
    print(f"Resume mode: {args.resume}", flush=True)
    print(flush=True)

    t_start = time.time()
    all_results = {}

    for m in m_values:
        ckpt_path = results_dir / f"panel_b_m{m}.npz"
        if args.resume and ckpt_path.exists():
            print(f"=== m={m}: SKIP (checkpoint exists at {ckpt_path}) ===",
                  flush=True)
            loaded = np.load(ckpt_path)
            for key in loaded.files:
                if key.startswith("surr_") or key.startswith("col_"):
                    algo, T = key.split("_")
                    all_results[(m, algo, int(T))] = loaded[key]
            continue

        m_results = run_m(m, c_values, args.seeds, n_jobs, ckpt_path)
        for (algo, T), regrets in m_results.items():
            all_results[(m, algo, T)] = regrets

    wall = time.time() - t_start

    # Save consolidated JSON (means + stderrs for easy plotting)
    summary = {
        "m_values": m_values,
        "c_values": c_values.tolist(),
        "n_seeds": args.seeds,
        "n_jobs": n_jobs,
        "wall_time_seconds": wall,
        "data": {},
    }
    for m in m_values:
        summary["data"][str(m)] = {}
        for algo in ["surr", "col"]:
            T_list = get_T_values(m, c_values)
            algo_data = {
                "T": [],
                "c": [],
                "mean": [],
                "stderr": [],
                "normalized_mean": [],  # R/m²
                "normalized_stderr": [],
                "K": [],
            }
            for T in T_list:
                regrets = all_results.get((m, algo, T))
                if regrets is None:
                    continue
                _, K_opt = (psi_surr if algo == "surr" else psi_col)(m, T)
                K_opt = max(K_opt, 1)
                mean = float(regrets.mean())
                stderr = float(regrets.std() / np.sqrt(len(regrets)))
                norm = m ** 2
                algo_data["T"].append(int(T))
                algo_data["c"].append(float(T / m ** 4))
                algo_data["mean"].append(mean)
                algo_data["stderr"].append(stderr)
                algo_data["normalized_mean"].append(mean / norm)
                algo_data["normalized_stderr"].append(stderr / norm)
                algo_data["K"].append(int(K_opt))
            summary["data"][str(m)][algo] = algo_data

    out_path = results_dir / "panel_b_full.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Full run complete", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Wall time: {wall:.1f}s = {wall/60:.1f} min = {wall/3600:.2f} h",
          flush=True)
    print(f"Summary JSON: {out_path}", flush=True)
    print(f"Per-m checkpoints: {results_dir}/panel_b_m*.npz", flush=True)


if __name__ == "__main__":
    main()
