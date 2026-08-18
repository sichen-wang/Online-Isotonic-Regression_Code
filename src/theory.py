"""Theoretical quantities for online isotonic regression on [m]x[m].

Provides MacMahon formula, upper bound functions Psi, phi/psi scaling functions,
and optimal discretization level K computations.
"""

import numpy as np
from scipy.special import gammaln, logsumexp


# ---------------------------------------------------------------------------
# Core combinatorial quantity: order polynomial
# ---------------------------------------------------------------------------

def log_omega(m: int, K: int) -> float:
    r"""Compute log Omega([m]^2, K+1) via MacMahon product formula.

    .. math::
        \log\Omega = \sum_{i=1}^m \sum_{j=1}^m \log\!\bigl(1 + K/(i+j-1)\bigr)
    """
    total = 0.0
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            total += np.log1p(K / (i + j - 1))
    return total


def log_omega_vec(m: int, K_arr: np.ndarray) -> np.ndarray:
    """Vectorised log_omega for an array of K values."""
    ij = np.zeros(m * m)
    idx = 0
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            ij[idx] = i + j - 1
            idx += 1
    # shape (len(K_arr), m*m)
    return np.sum(np.log1p(K_arr[:, None] / ij[None, :]), axis=1)


# ---------------------------------------------------------------------------
# Upper-bound functions  Psi
# ---------------------------------------------------------------------------

def _log_binom_2m_m(m: int) -> float:
    """log C(2m, m)."""
    return gammaln(2 * m + 1) - 2 * gammaln(m + 1)


def _log_binom_mpK_K(m: int, K: int) -> float:
    """log C(m+K, K)."""
    return gammaln(m + K + 1) - gammaln(K + 1) - gammaln(m + 1)


def psi_info(m: int, T: float, K_max: int | None = None):
    """Information-theoretic upper bound Psi_{[m]^2}(T).

    Returns (value, optimal_K).
    """
    if K_max is None:
        K_max = max(int(np.sqrt(T)) + 2, 2 * m)
    best, best_K = float(T), 0
    for K in range(1, K_max + 1):
        val = 2.0 * log_omega(m, K) + T / (4.0 * K * K)
        if val < best:
            best, best_K = val, K
    return best, best_K


def psi_surr(m: int, T: float, K_max: int | None = None, cap: bool = True):
    """Surrogate TI-EW bound: inf_K [2K log C(2m,m) + T/(4K^2)].

    Returns (value, optimal_K).  With cap=True (default) the value is
    additionally capped at the trivial bound T (K=0 signals the cap bound);
    cap=False returns the pure inf-of-affine value, as in the paper's
    epoch-selection rule (Appendix F.3).
    """
    lb = _log_binom_2m_m(m)
    if K_max is None:
        K_max = max(int((T / (4.0 * lb)) ** (1 / 3) * 2) + 2, 10)
    best, best_K = (float(T), 0) if cap else (float("inf"), 0)
    for K in range(1, K_max + 1):
        val = 2.0 * K * lb + T / (4.0 * K * K)
        if val < best:
            best, best_K = val, K
    return best, best_K


def psi_col(m: int, T: float, K_max: int | None = None, cap: bool = True):
    """Column-independent chain EW bound: inf_K [2m log C(m+K,K) + T/(4K^2)].

    Returns (value, optimal_K).  cap semantics as in psi_surr.
    """
    if K_max is None:
        K_max = max(int(np.sqrt(T)) + 2, 2 * m)
    best, best_K = (float(T), 0) if cap else (float("inf"), 0)
    for K in range(1, K_max + 1):
        val = 2.0 * m * _log_binom_mpK_K(m, K) + T / (4.0 * K * K)
        if val < best:
            best, best_K = val, K
    return best, best_K


def phi_combined(m: int, T: float):
    """Combined strategy bound Phi(T) = min{T, Psi_surr, Psi_col}.

    Returns (value, chosen_algorithm, optimal_K).
    chosen_algorithm: 'trivial', 'surr', or 'col'.
    """
    ps, Ks = psi_surr(m, T)
    pc, Kc = psi_col(m, T)
    if T <= ps and T <= pc:
        return float(T), "trivial", 0
    if ps <= pc:
        return ps, "surr", Ks
    return pc, "col", Kc


# ---------------------------------------------------------------------------
# Continuous scaling functions phi, psi
# ---------------------------------------------------------------------------

def varphi(alpha: float) -> float:
    r"""Continuous scaling function phi(alpha).

    .. math::
        \varphi(\alpha) = \tfrac12 \alpha^2\ln\alpha
            + \tfrac12(\alpha+2)^2\ln(\alpha+2)
            - (\alpha+1)^2\ln(\alpha+1) - 2\ln 2
    """
    if alpha <= 0:
        return 0.0
    a = alpha
    return (0.5 * a * a * np.log(a)
            + 0.5 * (a + 2) ** 2 * np.log(a + 2)
            - (a + 1) ** 2 * np.log(a + 1)
            - 2.0 * np.log(2))


def psi_continuous(c: float, n_grid: int = 2000) -> float:
    r"""Continuous crossover function psi(c) = inf_{alpha>0} [2 phi(alpha) + c/(4 alpha^2)]."""
    if c <= 0:
        return 0.0
    alphas = np.geomspace(1e-4, max(c, 10.0), n_grid)
    vals = np.array([2.0 * varphi(a) + c / (4.0 * a * a) for a in alphas])
    return float(np.min(vals))


# ---------------------------------------------------------------------------
# Epoch schedule for anytime wrapper
# ---------------------------------------------------------------------------

def compute_epoch_schedule(m: int, T_max: int) -> list[int]:
    """Rate-doubling epoch lengths of the paper's Algorithm 3 (eq. l-k-def).

    L_k = max{L in N+ : Phi(L) <= 2^{k-1}}, k = 1, 2, ...  (last-below rule).
    Returns [L_1, L_2, ...] until the cumulative length reaches T_max.

    [2026-08-14] Corrected: an earlier version used the relative rule
    H_{j+1} = min{h > H_j : Phi(h) >= 2 Phi(H_j)}, which drifts from the
    paper's schedule (e.g. m=12: epoch 10 started at t=12,964 instead of
    the paper's 12,880).
    """
    schedule = []
    cumulative = 0
    k = 1
    while cumulative < T_max:
        budget = 2.0 ** (k - 1)
        # find max{L : Phi(L) <= budget}; Phi(1) = 1 <= budget for all k >= 1
        lo, hi = 1, 2
        while phi_combined(m, hi)[0] <= budget:
            hi *= 2
        # invariant: Phi(lo) <= budget < Phi(hi); binary search the last-below point
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if phi_combined(m, mid)[0] <= budget:
                lo = mid
            else:
                hi = mid
        schedule.append(lo)
        cumulative += lo
        k += 1
    return schedule
