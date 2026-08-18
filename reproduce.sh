#!/usr/bin/env bash
# End-to-end reproduction script for the experiments in the paper.
#
# Default mode (no flag): regenerate the three-panel figure
#   (figures/figure1.pdf) from the pre-computed .npz files in results/
#   and run the cheap subset of the Appendix A correctness verification
#   (configs with K = 2; matches the first four rows of the Appendix A
#   table).  Wall time ~3 minutes single-core.
#
# --verify-full:  regenerate the figure from cached .npz and additionally
#   run all 8 correctness configs, including the (m, K) = (5, 3) and
#   (6, 2) cases that dominate the wall time.  Wall time: about one hour
#   single-core; (5, 3) alone takes roughly 50 minutes.
#
# --calibration:  re-run all eight configurations of the UCI Adult
#   calibration experiment (Appendix A), overwriting the cached
#   results/calibration_*.json.  Requires the dataset at data/adult.zip
#   (see README, "Calibration experiment").  Wall time: about two hours
#   single-core, scaling down with $JOBS (the twenty seeds of each
#   configuration are independent).
#
# --full:  also re-run the data generation for all three panels of the
#   figure from scratch (100 seeds), then run --verify-full and plot.
#   Requires a multi-core machine; wall time scales with cores and is
#   in the order of several hours.
#
# --plot-only:  only redraw figures/figure1.pdf from existing .npz
#   files.  Wall time: <1 minute.  Useful as a smoke test that the
#   plotting pipeline is wired up correctly.
#
# --smoke: light-weight end-to-end check.  Exercises the data-generation
#   and correctness pipelines at small (m, T, seeds), writing everything
#   under results/smoke/.  Does not redraw the figure and does not
#   overwrite any cached file.  Wall time: a few seconds with $JOBS=8.
#   (The calibration experiment is not exercised: it needs the external
#   dataset.)

set -euo pipefail

cd "$(dirname "$0")"

MODE="default"
SEEDS_FULL=100
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)         MODE="full" ;;
    --verify-full)  MODE="verify-full" ;;
    --calibration)  MODE="calibration" ;;
    --plot-only)    MODE="plot" ;;
    --smoke)        MODE="smoke" ;;
    -h|--help)
      sed -n '2,36p' "$0"
      exit 0 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

# Oversubscription guards for any numpy / joblib worker processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Suppress .pyc / __pycache__ generation so the package directory stays clean.
export PYTHONDONTWRITEBYTECODE=1

echo "[reproduce] mode=$MODE  jobs=$JOBS"

case "$MODE" in
  plot)
    python scripts/plot_figure1.py
    ;;

  default)
    # Plot the figure from the cached .npz files, then run the cheap
    # subset (K = 2) of the Appendix A correctness verification, then
    # refresh the human-readable summary tables (Markdown + CSV).
    # The reference results/correctness.npz (full 8 configs) is left
    # untouched; the K=2 subset is written to results/correctness_K2.npz.
    python scripts/plot_figure1.py
    python scripts/verify_correctness.py --max-K 2 \
        --output results/correctness_K2.npz
    python scripts/export_summaries.py
    ;;

  verify-full)
    # Plot the figure from the cached .npz files, then run the full
    # 8-config correctness verification used in Appendix A, and
    # refresh the human-readable summaries.
    python scripts/plot_figure1.py
    python scripts/verify_correctness.py
    python scripts/export_summaries.py
    ;;

  calibration)
    # All eight configurations of the Appendix A calibration experiment.
    # Intentionally overwrites the cached results/calibration_*.json.
    if [[ ! -f data/adult.zip ]]; then
      echo "[reproduce] data/adult.zip not found." >&2
      echo "  Download UCI Adult (CC BY 4.0, DOI 10.24432/C5XW20) first:" >&2
      echo "    mkdir -p data" >&2
      echo "    curl -L -o data/adult.zip https://archive.ics.uci.edu/static/public/2/adult.zip" >&2
      exit 3
    fi
    # Headline grid and the grid-resolution sweep
    python scripts/run_calibration.py --m 16 --seeds 20 --workers "$JOBS"
    python scripts/run_calibration.py --m 12 --seeds 20 --workers "$JOBS"
    python scripts/run_calibration.py --m 24 --seeds 20 --workers "$JOBS"
    # Horizon sweep (prefixes of the stream)
    python scripts/run_calibration.py --m 16 --seeds 20 --workers "$JOBS" --t-frac 0.1
    python scripts/run_calibration.py --m 16 --seeds 20 --workers "$JOBS" --t-frac 0.25
    python scripts/run_calibration.py --m 16 --seeds 20 --workers "$JOBS" --t-frac 0.5
    # Controls: shuffled covariate; gradient-boosting base model
    python scripts/run_calibration.py --m 16 --seeds 20 --workers "$JOBS" --placebo
    python scripts/run_calibration.py --m 16 --seeds 20 --workers "$JOBS" --clf gbm
    ;;

  full)
    # Re-generate every synthetic-input artefact, then plot, then refresh
    # human-readable summaries.  (The calibration experiment has its own
    # mode, --calibration, because it needs the external dataset.)
    python scripts/run_panel_a.py --m 30 --seeds "$SEEDS_FULL" \
        --workers "$JOBS" --c-max 6 \
        --output results/panel_a_m30.npz
    python scripts/run_panel_b.py --seeds "$SEEDS_FULL" --jobs "$JOBS"
    python scripts/run_panel_c.py --m 12 --seeds "$SEEDS_FULL" \
        --workers "$JOBS"
    python scripts/verify_correctness.py
    python scripts/plot_figure1.py
    python scripts/export_summaries.py
    ;;

  smoke)
    mkdir -p results/smoke
    # A short end-to-end pass on small (m, T, seeds) to exercise the
    # data-generation and correctness pipelines.  Designed to finish in
    # well under five minutes on a small workstation; not intended to
    # reproduce the paper numbers, and does not redraw the figure.
    python scripts/run_panel_a.py --m 8 --seeds 3 --workers "$JOBS" \
        --n-t 3 --c-max 1 \
        --output results/smoke/panel_a_m8.npz
    python scripts/run_panel_b.py --seeds 3 --jobs "$JOBS" \
        --m-list 8 --c-min 0.1 --c-max 1 --n-c 3 \
        --output-dir results/smoke
    python scripts/run_panel_c.py --m 8 --seeds 2 --t-max 2000 \
        --workers "$JOBS" --output results/smoke/panel_c_m8.npz
    python scripts/verify_correctness.py \
        --max-m 4 --max-K 2 \
        --output results/smoke/correctness.npz
    echo "[reproduce] smoke run completed.  Inspect results/smoke/ for output."
    ;;
esac

echo "[reproduce] done."
