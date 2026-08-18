#!/usr/bin/env python3
"""Figure 3 of the paper (Appendix A): three panels.

Panel (a): m=30 regret curves of the two engines + T^{1/3} and log T references.
Panel (b): regret ratio R_col/R_surr vs c = T/m^4, m in {8, 12, 18}.
Panel (c): full trajectory of the horizon-free wrapper at m=12 with the
rate-doubling restarts marked and one restart zoomed in an inset; fresh
known-horizon runs of the better engine overlaid as crosses.

Data: results/panel_a_m30.npz, results/panel_b_m{8,12,18}.npz,
      results/panel_c_m12.npz (produced by scripts/run_panel_c.py).
Output: figures/figure1.pdf (the file included by the paper source).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


plt.style.use(["science", "ieee"])
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 7,
    "axes.labelsize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.6,
    "lines.markersize": 2.5,
})

ALGO_STYLE_A = {
    "surr":    {"color": "#0072B2", "marker": "o", "ls": "-",
                "label": r"Surrogate TI-EW"},
    "col":     {"color": "#D55E00", "marker": "s", "ls": "-",
                "label": r"Column-Ind.\ Chain EW"},
    "anytime": {"color": "#009E73", "marker": "D", "ls": "-",
                "label": r"Horizon-Free"},
}

M_STYLE_B = {
    8:  {"color": "#1f77b4", "marker": "o"},
    12: {"color": "#ff7f0e", "marker": "s"},
    18: {"color": "#2ca02c", "marker": "^"},
}


# ===================================================================
# Panel (a): m=30
# ===================================================================

def plot_panel_a(ax, npz_path, m=30, xlim=(1e3, 5e6)):
    d = np.load(npz_path)
    T_values = np.array(sorted(set(
        int(k.split("_")[1]) for k in d.files
        if "_" in k and k.split("_")[1].isdigit()
    )))

    for algo_name in ["surr", "col"]:
        means = []
        sems = []
        for T in T_values:
            r = d[f"{algo_name}_{int(T)}"]
            means.append(r.mean())
            sems.append(r.std(ddof=1) / np.sqrt(len(r)))
        means = np.array(means)
        sems = np.array(sems)
        st = ALGO_STYLE_A[algo_name]
        ax.errorbar(T_values, means, yerr=sems,
                    color=st["color"], marker=st["marker"],
                    linestyle=st["ls"],
                    markersize=2.5, linewidth=0.6, capsize=0,
                    label=st["label"], zorder=3)

    # ∝ T^{1/3} reference, anchored at (T=10199, R=191) on Surrogate
    T_anchor1 = 10199
    R_anchor1 = 191.0
    T_ref1 = np.geomspace(xlim[0], xlim[1], 300)
    ax.plot(T_ref1, R_anchor1 * (T_ref1 / T_anchor1) ** (1 / 3),
            color="gray", linewidth=0.8, linestyle=":", alpha=0.95,
            zorder=5, label=r"$\propto T^{1/3}$")

    # ∝ log T reference: two-point anchored on Column-Ind Phase 2
    T_ref2 = np.geomspace(3e5, 5e6, 300)
    T1, R1 = 1.04e6, 944.0
    T2, R2 = 4.86e6, 1305.0
    R_ref2 = R1 + (R2 - R1) * (np.log(T_ref2) - np.log(T1)) / (np.log(T2) - np.log(T1))
    ax.plot(T_ref2, R_ref2,
            color="gray", linewidth=0.8, linestyle="-.", alpha=0.95,
            zorder=5, label=r"$\propto \log T$")

    ax.axvline(m ** 4, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(m ** 4 * 1.08, 0.5, r"$T\!=\!m^4$",
            transform=ax.get_xaxis_transform(),
            fontsize=6, color="gray", ha="left", va="center", alpha=0.8)

    ax.set_xlim(xlim)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"Regret against $f^{*}$")
    ax.set_title(r"(a) $m = 30$", fontsize=8)
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.92, fontsize=5.5,
              handlelength=1.6, labelspacing=0.25)


# ===================================================================
# Panel (b): Regret ratio
# ===================================================================

def plot_panel_b(ax, npz_paths):
    for m, npz_path in npz_paths.items():
        d = np.load(npz_path)
        T_vals = np.array(d["T_values"])
        c_vals = T_vals / m ** 4

        cs, means, sems = [], [], []
        for T, c in zip(T_vals, c_vals):
            if c < 0.5:
                continue
            r_surr = d[f"surr_{int(T)}"]
            r_col = d[f"col_{int(T)}"]
            ratios = r_col / np.maximum(r_surr, 1e-10)
            cs.append(c)
            means.append(ratios.mean())
            sems.append(ratios.std(ddof=1) / np.sqrt(len(ratios)))

        st = M_STYLE_B[m]
        ax.errorbar(cs, means, yerr=sems,
                    color=st["color"], marker=st["marker"], linestyle="-",
                    markersize=2.5, linewidth=0.6, capsize=0,
                    label=rf"$m = {m}$", zorder=3)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               zorder=1)
    ax.axvline(1.0, color="lightgray", linestyle="--", linewidth=0.6, zorder=1)
    ax.text(1.08, 0.5, r"$T\!=\!m^4$",
            transform=ax.get_xaxis_transform(),
            fontsize=6, color="gray", ha="left", va="center", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$c = T / m^4$")
    ax.set_ylabel(r"$R_{\mathrm{col}} \,/\, R_{\mathrm{surr}}$")
    ax.set_title(r"(b) Regret ratio", fontsize=8)
    ax.legend(loc="lower left", fontsize=6, frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.85,
              handlelength=1.6, labelspacing=0.25)


# ===================================================================
# Panel (c): horizon-free trajectory with restarts and a zoomed inset
# ===================================================================

def plot_panel_c(ax, traj_path, collapse_path, m=12, xlim=(1e2, 2.2e5)):
    d = np.load(traj_path)
    mean = d["mean"]
    T_max = int(d["T_max"][0])
    t = np.arange(1, T_max + 1)
    starts = d["epoch_starts"]

    # Known-horizon combined strategy: better engine at its tuned K, per T
    dc = np.load(collapse_path)
    T_ref = np.array(dc["T_values"], dtype=int)
    ref = np.array([min(dc[f"surr_{T}"].mean(), dc[f"col_{T}"].mean())
                    for T in T_ref])

    # Subsample the trajectory for a light PDF (log-dense, keep restarts exact);
    # start at the left edge so the autoscale fits the visible window
    idx = np.unique(np.concatenate([
        np.geomspace(0.9 * xlim[0], T_max, 4000).astype(int) - 1,
        (starts[starts > 1] - 2), (starts[starts > 1] - 1),
    ]))
    idx = idx[(idx >= int(0.9 * xlim[0]) - 1) & (idx < T_max)]

    restarts = starts[starts > 1]
    visible = restarts[restarts >= xlim[0]]
    for s in visible:
        ax.axvline(s, color="gray", linewidth=0.5, linestyle="--", alpha=0.55,
                   zorder=1)
    ax.text(visible[0] * 1.2, 0.05, "restarts",
            transform=ax.get_xaxis_transform(),
            fontsize=5.5, color="gray", ha="left", va="bottom", alpha=0.9)

    st = ALGO_STYLE_A["anytime"]
    ax.plot(t[idx], mean[idx], color=st["color"], linewidth=0.8,
            label=r"Horizon-Free", zorder=3)
    ax.plot(T_ref, ref, linestyle="none", marker="x", markersize=3.0,
            markeredgewidth=0.7, color="0.25",
            label=r"known-horizon runs", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    # headroom so the inset sits clear of the trajectory
    ax.set_ylim(top=3.2 * mean[-1])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Regret against $f^{*}$")
    ax.set_title(rf"(c) Horizon-free, $m = {m}$", fontsize=8)
    ax.legend(loc="lower right", fontsize=5.5, frameon=True, fancybox=False,
              edgecolor="0.7", framealpha=0.92,
              handlelength=1.6, labelspacing=0.25)

    # Inset: zoom on the restart nearest T = m^4 (linear scale)
    s0 = restarts[np.argmin(np.abs(restarts - m ** 4))]
    lo, hi = int(0.70 * s0), int(1.45 * s0)
    ymid = (mean[lo - 1] + mean[hi - 1]) / 2
    yspan = (mean[hi - 1] - mean[lo - 1]) * 0.75
    y0, y1 = ymid - yspan, ymid + yspan
    axins = ax.inset_axes([0.055, 0.60, 0.40, 0.35])
    win = (t >= lo) & (t <= hi)
    axins.plot(t[win], mean[win], color=st["color"], linewidth=0.8)
    axins.axvline(s0, color="gray", linewidth=0.5, linestyle="--", alpha=0.7)
    axins.set_xlim(lo, hi)
    axins.set_ylim(y0, y1)
    axins.set_xticks([s0])
    axins.set_xticklabels([rf"$t\!=\!{s0:,}$".replace(",", "{,}")], fontsize=5)
    axins.set_yticks([])
    axins.tick_params(length=1.5, pad=1)
    for sp in axins.spines.values():
        sp.set_linewidth(0.5)
    # mark the zoomed region on the main axes
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((lo, y0), hi - lo, y1 - y0, fill=False,
                           edgecolor="0.35", linewidth=0.5, zorder=5))


# ===================================================================
# Main
# ===================================================================

def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)

    pa_path = results_dir / "panel_a_m30.npz"
    pb_paths = {m: results_dir / f"panel_b_m{m}.npz" for m in [8, 12, 18]}
    pc_path = results_dir / "panel_c_m12.npz"
    for p in [pa_path, *pb_paths.values(), pc_path]:
        assert p.exists(), f"Missing: {p}"

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3,
        figsize=(6.5, 2.05),
        gridspec_kw={"width_ratios": [1.2, 0.92, 1.0]},
    )
    plot_panel_a(ax_a, pa_path, m=30)
    plot_panel_b(ax_b, pb_paths)
    plot_panel_c(ax_c, pc_path, pb_paths[12], m=12)

    fig.tight_layout(pad=0.4, w_pad=0.7)
    out_pdf = figures_dir / "figure1.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_pdf}", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
