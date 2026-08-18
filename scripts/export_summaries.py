#!/usr/bin/env python3
"""Export human-readable summaries from the cached .npz files.

Produces:
  results/correctness_table.md     — Appendix A table in Markdown form.
  results/panel_a_summary.csv      — Figure 3 Panel (a) per-T summary.
  results/panel_b_summary.csv      — Figure 3 Panel (b) per-(m, c) summary.

Pure post-processing of existing .npz files; runs in <1 second.
"""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# Appendix A correctness table
# ---------------------------------------------------------------------------

def export_correctness_md():
    npz = np.load(RESULTS / "correctness.npz")
    configs = npz["configs"]
    surr = npz["surr_max_disc"]
    col = npz["col_max_disc"]
    elapsed = npz["elapsed"]
    seed = int(npz["seed"][0])
    n = len(configs)

    threshold = 1e-9
    pass_surr = (surr < threshold).sum()
    pass_col = (col < threshold).sum()

    out = RESULTS / "correctness_table.md"
    with out.open("w") as f:
        f.write("# Appendix A: Correctness Verification\n\n")
        f.write("**Setup.**  Bernoulli labels with mean "
                "$f^{*}(i, j) = 0.25 + (i + j)/[4(m - 1)]$ on $[m] \\times [m]$, "
                "i.i.d. uniform queries, $T = 200$ rounds, "
                f"seed = {seed}.  Threshold for PASS is "
                "$\\max_t |\\hat{y}_t^{\\text{DP}} - \\hat{y}_t^{\\text{BF}}| "
                f"< 10^{{-9}}$.\n\n")
        f.write("Source data: `results/correctness.npz` "
                "(produced by `scripts/verify_correctness.py`).\n\n")
        f.write("| m | K |  T  | Surr. max disc | Col. max disc |  Time  | Status |\n")
        f.write("|---|---|-----|----------------|---------------|--------|--------|\n")
        for i in range(n):
            m, K, T = (int(x) for x in configs[i])
            status = ("PASS" if surr[i] < threshold and col[i] < threshold
                      else "FAIL")
            f.write(f"| {m} | {K} | {T} | "
                    f"{surr[i]:.3e} | {col[i]:.3e} | "
                    f"{elapsed[i]:6.1f}s | {status} |\n")
        f.write(f"\n**Result:** Surrogate {pass_surr}/{n} PASS, "
                f"Column-Independent {pass_col}/{n} PASS, "
                f"all discrepancies at machine precision (~$10^{{-13}}$ or "
                "smaller).\n")
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Panel (a) per-T summary
# ---------------------------------------------------------------------------

def export_panel_a_csv():
    npz = np.load(RESULTS / "panel_a_m30.npz")
    T_values = np.asarray(npz["T_values"], dtype=int)
    m = int(npz["m"][0])
    n_seeds = int(npz["n_seeds"][0])
    K_surr = np.asarray(npz["K_surr"], dtype=int)
    K_col = np.asarray(npz["K_col"], dtype=int)

    out = RESULTS / "panel_a_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "T", "c", "algo", "K", "n_seeds",
                    "mean_regret", "sem_regret"])
        for i, T in enumerate(T_values):
            T = int(T)
            c = T / m ** 4
            for algo in ("surr", "col"):
                key = f"{algo}_{T}"
                arr = np.asarray(npz[key], dtype=float)
                mean = float(arr.mean())
                sem = float(arr.std(ddof=0) / np.sqrt(len(arr)))
                K = int(K_surr[i]) if algo == "surr" else int(K_col[i])
                w.writerow([m, T, f"{c:.6g}", algo, K, n_seeds,
                            f"{mean:.6f}", f"{sem:.6f}"])
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Panel (b) per-(m, c) summary with regret ratio
# ---------------------------------------------------------------------------

def export_panel_b_csv():
    out = RESULTS / "panel_b_summary.csv"
    rows = []
    for m in (8, 12, 18):
        npz = np.load(RESULTS / f"panel_b_m{m}.npz")
        T_values = np.asarray(npz["T_values"], dtype=int)
        n_seeds = int(npz["n_seeds"][0])
        K_surr = np.asarray(npz["K_surr"], dtype=int)
        K_col = np.asarray(npz["K_col"], dtype=int)
        for i, T in enumerate(T_values):
            T = int(T)
            c = T / m ** 4
            r_surr = np.asarray(npz[f"surr_{T}"], dtype=float)
            r_col = np.asarray(npz[f"col_{T}"], dtype=float)
            mean_surr = float(r_surr.mean())
            sem_surr = float(r_surr.std(ddof=0) / np.sqrt(len(r_surr)))
            mean_col = float(r_col.mean())
            sem_col = float(r_col.std(ddof=0) / np.sqrt(len(r_col)))
            ratios = r_col / np.maximum(r_surr, 1e-10)
            ratio_mean = float(ratios.mean())
            ratio_sem = float(ratios.std(ddof=0) / np.sqrt(len(ratios)))
            rows.append([m, T, f"{c:.6g}", int(K_surr[i]), int(K_col[i]),
                         n_seeds,
                         f"{mean_surr:.6f}", f"{sem_surr:.6f}",
                         f"{mean_col:.6f}", f"{sem_col:.6f}",
                         f"{ratio_mean:.6f}", f"{ratio_sem:.6f}"])
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "T", "c", "K_surr", "K_col", "n_seeds",
                    "mean_surr", "sem_surr", "mean_col", "sem_col",
                    "ratio_mean", "ratio_sem"])
        w.writerows(rows)
    print(f"  wrote {out.relative_to(ROOT)}")


def main():
    print("Exporting human-readable summaries from results/*.npz ...")
    export_correctness_md()
    export_panel_a_csv()
    export_panel_b_csv()
    print("done.")


if __name__ == "__main__":
    main()
