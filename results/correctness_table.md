# Appendix A: Correctness Verification

**Setup.**  Bernoulli labels with mean $f^{*}(i, j) = 0.25 + (i + j)/[4(m - 1)]$ on $[m] \times [m]$, i.i.d. uniform queries, $T = 200$ rounds, seed = 42.  Threshold for PASS is $\max_t |\hat{y}_t^{\text{DP}} - \hat{y}_t^{\text{BF}}| < 10^{-9}$.

Source data: `results/correctness.npz` (produced by `scripts/verify_correctness.py`).

| m | K |  T  | Surr. max disc | Col. max disc |  Time  | Status |
|---|---|-----|----------------|---------------|--------|--------|
| 3 | 2 | 200 | 1.332e-15 | 9.992e-16 |    0.1s | PASS |
| 3 | 3 | 200 | 1.721e-15 | 1.221e-15 |    1.7s | PASS |
| 3 | 4 | 200 | 1.332e-15 | 1.221e-15 |   51.3s | PASS |
| 4 | 2 | 200 | 2.276e-15 | 6.661e-16 |    0.8s | PASS |
| 4 | 3 | 200 | 1.887e-15 | 9.992e-16 |   80.2s | PASS |
| 5 | 2 | 200 | 1.998e-15 | 6.661e-16 |   21.9s | PASS |
| 5 | 3 | 200 | 5.351e-14 | 7.772e-16 | 3245.1s | PASS |
| 6 | 2 | 200 | 2.665e-15 | 8.882e-16 |  138.0s | PASS |

**Result:** Surrogate 8/8 PASS, Column-Independent 8/8 PASS, all discrepancies at machine precision (~$10^{-13}$ or smaller).
