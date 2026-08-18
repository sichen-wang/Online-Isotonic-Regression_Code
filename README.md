# Supplementary Material

Reproduction code for the experimental section of the paper *"The Minimax Rate of Online Isotonic Regression on Product Orders"*.

This package reproduces **Figure 3** in Appendix A (the three-panel synthetic-stream figure), the **calibration experiment** of Appendix A (UCI Adult; every number quoted in the calibration paragraphs), and **Table 1** in Appendix A (numerical equivalence between the DP implementations and direct brute-force enumeration of the underlying expert classes). The pre-computed `.npz` / `.json` files are bundled under `results/`, so the figure, the table, and all quoted numbers can be inspected without rerunning any experiment. The UCI Adult dataset itself is not redistributed; see *Calibration experiment* below for the one-line download.


## Quick start

To inspect the bundled results without running any code, open these files directly:

* `figures/figure1.pdf` — Figure 3 of the paper (Appendix A).
* `results/correctness_table.md` — the Appendix A table in human-readable Markdown.
* `results/panel_a_summary.csv`, `results/panel_b_summary.csv` — per-$T$ and per-$(m, c)$ aggregated regret (mean ± SEM).
* `results/calibration_*.json` — the eight calibration configurations: per-seed cumulative losses of every learner and oracle, plus aggregated summaries (see *Mapping to paper*).

To re-execute the pipeline:

```bash
pip install -r requirements.txt
bash reproduce.sh
```

This regenerates `figures/figure1.pdf` from the cached `.npz` files, runs the cheap subset of the correctness verification, and refreshes the Markdown / CSV summaries. It finishes in about three minutes on a single core.


## Repository layout

```
supplementary/
├── README.md
├── LICENSE                       # MIT License
├── requirements.txt
├── reproduce.sh                  # end-to-end driver (6 modes)
├── src/
│   ├── __init__.py
│   ├── algorithms.py             # Surrogate TI-EW, Column-Ind. Chain EW,
│   │                             # FTL, Anytime Combined, brute-force references
│   └── theory.py                 # MacMahon formula, Psi/Phi bounds, epoch schedule
├── scripts/
│   ├── run_panel_a.py            # data for Figure 3, Panel (a)
│   ├── run_panel_b.py            # data for Figure 3, Panel (b)
│   ├── run_panel_c.py            # data for Figure 3, Panel (c)
│   ├── run_calibration.py        # the Appendix A calibration experiment
│   ├── plot_figure1.py           # produces figures/figure1.pdf
│   ├── verify_correctness.py     # data for Appendix A, Table 1
│   └── export_summaries.py       # derives Markdown / CSV summaries
├── results/                      # pre-computed inputs and outputs
│   ├── panel_a_m30.npz   # Panel (a): m = 30, 100 seeds
│   ├── panel_b_m8.npz        # Panel (b): m = 8,  100 seeds
│   ├── panel_b_m12.npz       # Panel (b): m = 12, 100 seeds
│   ├── panel_b_m18.npz       # Panel (b): m = 18, 100 seeds
│   ├── panel_c_m12.npz           # Panel (c): m = 12, 100 seeds
│   ├── calibration_m16.json      # calibration: 16 x 16 headline grid
│   ├── calibration_m12.json      # calibration: grid sweep, m = 12
│   ├── calibration_m24.json      # calibration: grid sweep, m = 24
│   ├── calibration_m16_t0.1.json # calibration: horizon sweep, first 10%
│   ├── calibration_m16_t0.25.json#   ... first 25%
│   ├── calibration_m16_t0.5.json #   ... first 50%
│   ├── calibration_m16_placebo.json  # control: shuffled covariate
│   ├── calibration_m16_gbm.json  # control: gradient-boosting base model
│   ├── correctness.npz           # Appendix A discrepancies (8 configs)
│   ├── correctness_table.md      # Appendix A table in Markdown
│   ├── panel_a_summary.csv       # Panel (a) aggregated regret summary
│   └── panel_b_summary.csv       # Panel (b) aggregated regret summary
├── figures/
│   └── figure1.pdf               # Figure 3, as included by the paper source
└── data/                         # created by you: data/adult.zip (see below)
```


## Reproduction modes

The driver script `reproduce.sh` exposes six modes. Pick the one that matches your time budget:

| Mode | Command | Wall time | What it does |
| --- | --- | --- | --- |
| Default | `bash reproduce.sh` | ~3 min single-core | Regenerates `figure1.pdf` from the cached `.npz` files, runs the cheap $K = 2$ subset of the correctness verification, and refreshes the Markdown / CSV summaries. |
| Plot only | `bash reproduce.sh --plot-only` | < 1 min | Redraws `figure1.pdf` from the cached `.npz` files and exits. |
| Smoke test | `bash reproduce.sh --smoke` | seconds with `JOBS=8` | End-to-end pipeline check at reduced $(m, T, \text{seeds})$, including the Panel (c) trajectory. Writes everything under `results/smoke/`; does not redraw the figure and never overwrites cached reference data. (The calibration experiment is not exercised: it needs the external dataset.) |
| Verify-full | `bash reproduce.sh --verify-full` | ~1 h single-core | Same as Default, but runs all eight correctness configurations, including the expensive $(m, K) = (5, 3)$ case (~50 min on its own). |
| Calibration | `bash reproduce.sh --calibration` | ~2 h single-core | Re-runs all eight configurations of the Appendix A calibration experiment (20 seeds each). Requires `data/adult.zip`; see *Calibration experiment* below. |
| Full | `bash reproduce.sh --full` | several hours | Regenerates every synthetic `.npz` from scratch — Panels (a), (b), and (c) with 100 seeds, plus the full 8-config correctness verification — then plots and exports summaries. |

All modes set `OMP_NUM_THREADS=1` (and the analogous MKL / OpenBLAS / Accelerate guards) before launching any worker process; data generation runs under process-based parallelism (`joblib` with the `loky` backend, or `concurrent.futures.ProcessPoolExecutor`). The worker count defaults to `nproc` and may be overridden with the `JOBS` environment variable:

```bash
JOBS=32 bash reproduce.sh --full
```

**Overwrite semantics.** The cached reference files are rewritten only by the modes designed to regenerate them: *Verify-full* overwrites `results/correctness.npz`, *Full* overwrites the panel `.npz` files, and *Calibration* overwrites the `calibration_*.json`. *Default* and *Plot only* redraw `figures/figure1.pdf` from the cached inputs and write the cheap correctness subset to a separate `results/correctness_K2.npz`; *Smoke test* writes only under `results/smoke/`.


## Mapping to paper

| Paper artefact | Script | Input data | Output |
| --- | --- | --- | --- |
| Figure 3, Panel (a) | `scripts/plot_figure1.py` | `results/panel_a_m30.npz` | `figures/figure1.pdf` |
| Figure 3, Panel (b) | `scripts/plot_figure1.py` | `results/panel_b_m{8,12,18}.npz` | `figures/figure1.pdf` |
| Figure 3, Panel (c) | `scripts/plot_figure1.py` | `results/panel_c_m12.npz` | `figures/figure1.pdf` |
| Panel (c) trajectory data | `scripts/run_panel_c.py` | (none; synthetic) | `results/panel_c_m12.npz` |
| Calibration, 16 × 16 headline | `scripts/run_calibration.py --m 16` | `data/adult.zip` | `results/calibration_m16.json` |
| Calibration, grid sweep | `... --m 12` / `--m 24` | `data/adult.zip` | `results/calibration_m{12,24}.json` |
| Calibration, horizon sweep | `... --t-frac 0.1/0.25/0.5` | `data/adult.zip` | `results/calibration_m16_t*.json` |
| Calibration, shuffled covariate | `... --placebo` | `data/adult.zip` | `results/calibration_m16_placebo.json` |
| Calibration, GBM base model | `... --clf gbm` | `data/adult.zip` | `results/calibration_m16_gbm.json` |
| Appendix A, Table 1 | `scripts/verify_correctness.py` | (none; regenerates internally) | `results/correctness.npz` |
| Appendix A table (Markdown) | `scripts/export_summaries.py` | `results/correctness.npz` | `results/correctness_table.md` |
| Panel (a) summary CSV | `scripts/export_summaries.py` | `results/panel_a_m30.npz` | `results/panel_a_summary.csv` |
| Panel (b) summary CSV | `scripts/export_summaries.py` | `results/panel_b_m{8,12,18}.npz` | `results/panel_b_summary.csv` |
| Surrogate TI-EW | `src/algorithms.py::SurrogateTIEW` | — | — |
| Column-Independent Chain EW | `src/algorithms.py::ColumnIndEW` | — | — |
| Horizon-Free Combined | `src/algorithms.py::AnytimeCombined` | — | — |
| Optimal $K$, $\Psi$, $\Phi$, epoch schedule | `src/theory.py` | — | — |

In each `calibration_*.json`, `rows` holds the per-seed cumulative squared losses of every learner and oracle, and `summary` the across-seed mean ± SEM; the margins quoted in the paper are paired differences of the per-seed rows.


## Adversary (synthetic streams)

All synthetic experiments use Bernoulli labels with the linear isotonic mean

$$
f^{*}(i, j) = 0.25 + \frac{i + j}{4 (m - 1)} \qquad \text{on } [m] \times [m],
$$

in the 0-indexed coordinates of the code (equivalently $0.25 + (i + j - 2) / [4 (m - 1)]$ in the 1-indexed coordinates of the paper), and i.i.d. uniform queries. The definition appears in `scripts/run_panel_a.py::f_star`, `scripts/run_panel_b.py::make_adversary`, `scripts/run_panel_c.py::f_star`, and `scripts/verify_correctness.py::f_star`.


## Calibration experiment

The Appendix A calibration experiment streams the UCI Adult dataset (Becker and Kohavi, 1996; DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20); licensed CC BY 4.0). The dataset is not redistributed with this package; download it once:

```bash
mkdir -p data
curl -L -o data/adult.zip https://archive.ics.uci.edu/static/public/2/adult.zip
```

Then `bash reproduce.sh --calibration` re-runs all eight configurations, or run a single one, e.g.

```bash
python scripts/run_calibration.py --m 16 --seeds 20
```

Each seed draws its own train/stream split (40% / 60%), trains the base classifier without the education feature, bins the stream onto the score-quantile × education grid, and processes it online under squared loss; `--diagnose` prints the split, cell-occupancy, and oracle statistics without running any learner.

A reproducibility note: the base logistic model is fit with scikit-learn, and its fitted scores are sensitive at rank-tie level to the underlying library builds, so individual quoted digits can move by about one posted standard error across environments (the gradient-boosting control reproduced exactly across the two environments we tested); all qualitative comparisons and win counts are unaffected.


## Environment

The package was tested with Python 3.10–3.13 and the package versions listed in `requirements.txt` (`numpy`, `scipy`, `matplotlib`, `scienceplots`, `joblib`, `pandas`, `scikit-learn`). A working LaTeX installation is required for the SciencePlots IEEE style used in `plot_figure1.py` (the script sets `text.usetex = True`). On Debian / Ubuntu, the following packages are sufficient:

```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended cm-super dvipng
```

A typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Computing resources

All experiments were run on single multi-core CPU machines (no GPU, no distributed cluster). Total compute is approximately **2400 CPU-hours**, distributed as follows.

| Experiment | Configurations | Approx. CPU-hours |
| --- | --- | --- |
| Figure 3, Panel (a) — `panel_a_m30.npz` | $m = 30$; 15 values of $T$ up to $5 \times 10^{6}$; 100 seeds; 2 algorithms | ~2240 |
| Figure 3, Panel (b) — `panel_b_m{8,12,18}.npz` | $m \in \{8, 12, 18\}$; 15 values of $c$ up to $10$ per $m$; 100 seeds; 2 algorithms | ~190 |
| Figure 3, Panel (c) — `panel_c_m12.npz` | $m = 12$; per-round trajectory to $T \approx 2.1 \times 10^{5}$; 100 seeds | ~2 |
| Calibration — `calibration_*.json` | 8 configurations; 20 seeds each | ~2 |
| Appendix A correctness — `correctness.npz` | 8 configurations of $(m, K)$ with $T = 200$, seed 42 | ~1 |

Panel (a) is dominated by the largest $T$ at the largest discretization ($K \approx 30$–$40$). Panel (b) is dominated by $m = 18$ at $c = 10$. The correctness verification is single-threaded and dominated by the brute-force baseline at $(m, K) = (5, 3)$.

The data-generation steps are embarrassingly parallel across seeds and configurations, so wall time on a reviewer's machine scales roughly inversely with the available worker count (controlled by the `JOBS` environment variable; see *Reproduction modes* above).


## License

The code in this package is released under the MIT License; see the `LICENSE` file for the full text. The UCI Adult dataset is credited above and carries its own license (CC BY 4.0).
