#!/usr/bin/env python3
"""Panel (c) experiment: full per-round trajectory of the horizon-free wrapper.

Records the cumulative pseudo-regret R_t (vs. the generating f*) of
AnytimeCombined at every round t = 1..T_max, averaged over seeds, so that the
rate-doubling restarts are visible and one of them can be zoomed in an inset.

Same protocol as run_panel_a.py: f*(i,j) = 0.25 + (i+j)/[4(m-1)], i.i.d.
uniform queries, Bernoulli labels, RandomState(seed) -- with a given seed the
stream is the same infinite stream whose prefixes run_panel_a.py consumes.

Saved npz:
    mean, sem          float64[T_max]   across-seed mean / SEM of R_t
    final_regrets      float64[seeds]   per-seed R_{T_max} (cross-check)
    epoch_starts       int64[E]         first round t of each epoch (1-based)
    epoch_horizons     int64[E]         scheduled horizon H_j of each epoch
    epoch_engines      str[E]           'surr' / 'col' / 'triv'
    epoch_K            int64[E]         discretization level used in the epoch
    m, T_max, n_seeds  scalars

Usage:
    python3 scripts/run_panel_c.py --m 12 --seeds 2 --t-max 20000   # smoke
    python3 scripts/run_panel_c.py --m 12 --seeds 100 --workers 8
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.theory import compute_epoch_schedule, psi_surr, psi_col
from src.algorithms import AnytimeCombined


def f_star(i, j, m):
    if m <= 1:
        return 0.5
    return 0.25 + (i + j) / (4.0 * (m - 1))


def run_single(task):
    """(m, T_max, seed) -> (seed, float32 trajectory of cumulative regret)."""
    m, T_max, seed = task
    algo = AnytimeCombined(m, T_max)

    rng = np.random.RandomState(seed)
    f_grid = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            f_grid[i, j] = f_star(i, j, m)

    traj = np.empty(T_max, dtype=np.float64)
    loss_algo = 0.0
    loss_best = 0.0
    for t in range(T_max):
        i0 = rng.randint(m)
        j0 = rng.randint(m)
        p = f_grid[i0, j0]
        y = float(rng.rand() < p)
        yhat = algo.predict(i0, j0)
        loss_algo += (yhat - y) ** 2
        loss_best += (p - y) ** 2
        algo.update(i0, j0, y)
        traj[t] = loss_algo - loss_best

    return (seed, traj)


def epoch_metadata(m, T_max):
    """Epoch starts (1-based), lengths, engine kind, and K, mirroring
    AnytimeCombined._start_new_epoch (paper rule: col iff Psi_col <= Psi_surr)."""
    schedule = compute_epoch_schedule(m, T_max)
    starts, engines, Ks = [], [], []
    t = 1
    for L in schedule:
        starts.append(t)
        ps, Ks_s = psi_surr(m, L, cap=False)
        pc, Kc = psi_col(m, L, cap=False)
        if pc <= ps:
            engines.append("col"); Ks.append(max(int(Kc), 1))
        else:
            engines.append("surr"); Ks.append(max(int(Ks_s), 1))
        t += L
    return (np.array(starts, dtype=np.int64),
            np.array(schedule, dtype=np.int64),
            np.array(engines),
            np.array(Ks, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--t-max", type=int, default=None,
                        help="default: 10 * m^4")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    m = args.m
    T_max = args.t_max or 10 * m ** 4
    root = Path(__file__).resolve().parent.parent
    out = args.output or str(root / "results" / f"panel_c_m{m}.npz")

    starts, horizons, engines, Ks = epoch_metadata(m, T_max)
    print(f"=== Panel (c) trajectory ===  m={m} T_max={T_max} "
          f"seeds={args.seeds} workers={args.workers}", flush=True)
    print("  epoch starts:  ", starts.tolist(), flush=True)
    print("  epoch horizons:", horizons.tolist(), flush=True)
    print("  epoch engines: ", engines.tolist(), flush=True)
    print("  epoch K:       ", Ks.tolist(), flush=True)

    tasks = [(m, T_max, s) for s in range(args.seeds)]
    t0 = time.time()
    acc = np.zeros(T_max)
    acc2 = np.zeros(T_max)
    finals = np.zeros(args.seeds)
    n_done = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, args.seeds)) as ex:
        futs = [ex.submit(run_single, t) for t in tasks]
        for f in as_completed(futs):
            seed, traj = f.result()
            acc += traj
            acc2 += traj * traj
            finals[seed] = traj[-1]
            n_done += 1
            el = time.time() - t0
            print(f"  {n_done}/{args.seeds} done ({el:.0f}s, "
                  f"ETA {el * (args.seeds - n_done) / n_done:.0f}s)", flush=True)

    n = args.seeds
    mean = acc / n
    var = np.maximum(acc2 / n - mean ** 2, 0.0) * n / max(n - 1, 1)
    sem = np.sqrt(var / n)

    np.savez_compressed(
        out,
        mean=mean, sem=sem, final_regrets=finals,
        epoch_starts=starts, epoch_horizons=horizons,
        epoch_engines=engines, epoch_K=Ks,
        m=np.array([m]), T_max=np.array([T_max]),
        n_seeds=np.array([n]),
    )
    print(f"\nWall {time.time() - t0:.0f}s   saved -> {out}", flush=True)
    print(f"final regret: {mean[-1]:.1f} +- {sem[-1]:.1f}", flush=True)
    for s, e in zip(starts.tolist(), engines.tolist()):
        if 1 < s <= T_max:
            idx = s - 1  # R_t just before the restart round
            print(f"  restart at t={s} ({e}): R={mean[idx - 1]:.1f}", flush=True)


if __name__ == "__main__":
    main()
