"""Algorithm implementations for online isotonic regression on [m]x[m].

All algorithms use 0-indexed cells (i, j) with i, j in {0, ..., m-1}.
The product order: (i,j) <= (i',j') iff i<=i' and j<=j'.

Algorithms:
    SurrogateTIEW   -- Phase 1 rate-optimal (Model R)
    ColumnIndEW     -- Phase 2 rate-optimal
    FTL             -- Follow-the-Leader baseline
    AnytimeCombined -- Rate-doubling anytime wrapper
"""

import numpy as np
from scipy.special import logsumexp

from .theory import psi_surr, psi_col, phi_combined, compute_epoch_schedule


# ===================================================================
# Surrogate TI-EW  (Section 5.2)
# ===================================================================

class SurrogateTIEW:
    """Surrogate Threshold-Index Exponential Weights.

    Each of K levels maintains an independent upset distribution on [m]x[m].
    Per-round prediction cost: O(K m^2).
    """

    def __init__(self, m: int, K: int):
        self.m = m
        self.K = K
        # Per-cell statistics (shared across levels)
        self.S = np.zeros((m, m), dtype=np.float64)   # sum of labels
        self.N = np.zeros((m, m), dtype=np.float64)   # query count

    def reset(self):
        self.S[:] = 0.0
        self.N[:] = 0.0

    def predict(self, i0: int, j0: int) -> float:
        """Return prediction at 0-indexed cell (i0, j0)."""
        probs = self._marginals_all_levels(i0, j0)  # shape (K,)
        return float(np.sum(probs)) / self.K

    def update(self, i0: int, j0: int, y: float):
        """Update after observing label y at (i0, j0)."""
        self.S[i0, j0] += y
        self.N[i0, j0] += 1.0

    # ------------------------------------------------------------------
    # Upset marginal DP  (Lemma DP, vectorised over K levels)
    # ------------------------------------------------------------------

    def _marginals_all_levels(self, i0: int, j0: int) -> np.ndarray:
        """Compute Pr[(i0,j0) in U_r] for all levels r simultaneously.

        Upset parameterisation (1-indexed in paper):
            c_1 >= c_2 >= ... >= c_m,  c_i in {0,...,m}.
            Cell (i,j) in U  iff  j > c_i   (1-indexed).
        In 0-indexed code: cell (i,j) in U  iff  j+1 > c[i]  iff  c[i] <= j.

        Weight of cell (i,j) at level r (0-indexed r, paper r+1):
            log w[r,i,j] = S[i,j]/K - (2r+1) N[i,j] / (2K^2)
        """
        m, K = self.m, self.K

        # Precompute suffix sums of S and N along each column
        # cumS[i, c] = sum_{j=c}^{m-1} S[i,j],  c in {0,...,m}
        cumS = np.zeros((m, m + 1))
        cumN = np.zeros((m, m + 1))
        for i in range(m):
            for c in range(m - 1, -1, -1):
                cumS[i, c] = cumS[i, c + 1] + self.S[i, c]
                cumN[i, c] = cumN[i, c + 1] + self.N[i, c]

        # Level coefficients: (2r+1) for r = 0, ..., K-1
        r_coeff = (2.0 * np.arange(K) + 1.0) / (2.0 * K * K)  # shape (K,)

        # log a_r[i, c] = cumS[i,c]/K - r_coeff[r] * cumN[i,c]
        # Broadcast: (K,1,1) * (1, m, m+1) -> (K, m, m+1)
        # We compute per-column during the DP to save memory.

        # Forward DP --------------------------------------------------
        # log_F[r, c] = log F_{column i}(cutoff c), shape (K, m+1)
        # Column 0
        log_a0 = (cumS[0, :][np.newaxis, :] / K
                  - r_coeff[:, np.newaxis] * cumN[0, :][np.newaxis, :])
        log_F = log_a0.copy()  # (K, m+1)

        for i in range(1, m):
            log_ai = (cumS[i, :][np.newaxis, :] / K
                      - r_coeff[:, np.newaxis] * cumN[i, :][np.newaxis, :])
            # Suffix logsumexp of log_F along cutoff axis (axis=1)
            log_suffix = np.logaddexp.accumulate(log_F[:, ::-1], axis=1)[:, ::-1]
            log_F = log_ai + log_suffix

        # Partition function
        log_Z = logsumexp(log_F, axis=1)  # (K,)

        # Backward DP -------------------------------------------------
        # log_B[r, c], shape (K, m+1)
        log_B = np.zeros((K, m + 1))  # B[m-1][c] = 1 => log = 0

        for i in range(m - 2, -1, -1):
            log_ai1 = (cumS[i + 1, :][np.newaxis, :] / K
                       - r_coeff[:, np.newaxis] * cumN[i + 1, :][np.newaxis, :])
            log_ab = log_ai1 + log_B  # (K, m+1)
            # Prefix logsumexp along cutoff axis
            log_B = np.logaddexp.accumulate(log_ab, axis=1)

        # Marginal: Pr[c_{i0} <= j0] = sum_{c=0}^{j0} F[i0,c]*B[i0,c] / Z
        # We need to recompute F and B at column i0.
        # Rather than storing full arrays, recompute with two passes.
        # -- Actually we only stored the *final* column's F.
        # Let me redo with stored per-column values for i0.

        # Efficient approach: single forward pass storing F at i0,
        # single backward pass storing B at i0.
        return self._marginals_two_pass(i0, j0, cumS, cumN, r_coeff)

    def _marginals_two_pass(self, i0, j0, cumS, cumN, r_coeff):
        """Two-pass DP computing marginal at column i0."""
        m, K = self.m, self.K

        def log_a(i):
            """log a_r[i, c] for all r and c, shape (K, m+1)."""
            return (cumS[i, :][np.newaxis, :] / K
                    - r_coeff[:, np.newaxis] * cumN[i, :][np.newaxis, :])

        # Forward pass: compute log_F at column i0
        log_F = log_a(0)  # (K, m+1)
        for i in range(1, i0 + 1):
            la = log_a(i)
            log_suffix = np.logaddexp.accumulate(log_F[:, ::-1], axis=1)[:, ::-1]
            log_F = la + log_suffix
        log_F_i0 = log_F  # (K, m+1)

        # Backward pass: compute log_B at column i0
        log_B = np.zeros((K, m + 1))  # column m-1: B = 1
        for i in range(m - 2, i0 - 1, -1):
            log_ab = log_a(i + 1) + log_B
            log_B = np.logaddexp.accumulate(log_ab, axis=1)
        log_B_i0 = log_B  # (K, m+1)

        # Partition function (from any column, use i0)
        log_FB_all = log_F_i0 + log_B_i0  # (K, m+1)
        log_Z = logsumexp(log_FB_all, axis=1)  # (K,)

        # Marginal: sum_{c=0}^{j0} F*B / Z
        # c <= j0 means cutoff values 0, 1, ..., j0
        log_FB_sel = log_F_i0[:, :j0 + 1] + log_B_i0[:, :j0 + 1]
        if log_FB_sel.shape[1] == 0:
            return np.zeros(K)
        log_probs = logsumexp(log_FB_sel, axis=1) - log_Z  # (K,)
        return np.exp(np.clip(log_probs, -500, 0))


# ===================================================================
# Column-Independent Chain EW  (Section 5.1)
# ===================================================================

class ColumnIndEW:
    """Column-independent Exponential Weights.

    Each column runs independent EW over monotone functions [m] -> {0,...,K}.
    Per-round prediction cost: O(m K)  (process one column via transfer-matrix DP).
    """

    def __init__(self, m: int, K: int):
        self.m = m
        self.K = K
        self.S = np.zeros((m, m), dtype=np.float64)
        self.N = np.zeros((m, m), dtype=np.float64)

    def reset(self):
        self.S[:] = 0.0
        self.N[:] = 0.0

    def predict(self, i0: int, j0: int) -> float:
        """Predict at (i0, j0) using column i0's chain EW."""
        return self._column_marginal(i0, j0)

    def update(self, i0: int, j0: int, y: float):
        self.S[i0, j0] += y
        self.N[i0, j0] += 1.0

    def _column_marginal(self, i0: int, j0: int) -> float:
        """Compute E[g(j0)/K] under column-i0 EW distribution.

        States k = 0, ..., K.  Monotone: k_{j} <= k_{j+1}.
        Cell weight: log w(i0, j, k) = k S[i0,j]/K - k^2 N[i0,j]/(2K^2).
        """
        m, K = self.m, self.K
        Kp1 = K + 1

        # Per-cell log-weights for column i0, all rows
        # log_w[j, k] for j=0..m-1, k=0..K
        k_vals = np.arange(Kp1, dtype=np.float64)  # 0, 1, ..., K
        S_col = self.S[i0, :]  # (m,)
        N_col = self.N[i0, :]  # (m,)
        log_w = (k_vals[np.newaxis, :] * S_col[:, np.newaxis] / K
                 - k_vals[np.newaxis, :] ** 2 * N_col[:, np.newaxis] / (2.0 * K * K))
        # shape (m, K+1)

        # Forward DP: log_F[j, k]
        # F[0, k] = w(0, k)
        log_F = log_w[0, :].copy()  # (K+1,)

        for j in range(1, m):
            # Prefix logsumexp: log_prefix[k] = logsumexp(log_F[0..k])
            log_prefix = np.logaddexp.accumulate(log_F)  # (K+1,)
            log_F = log_w[j, :] + log_prefix

        # Backward DP: log_B[j, k]
        log_B = np.zeros(Kp1)  # B[m-1, k] = 1

        for j in range(m - 2, -1, -1):
            # Suffix logsumexp of (log_w[j+1, :] + log_B)
            log_wB = log_w[j + 1, :] + log_B  # (K+1,)
            log_B = np.logaddexp.accumulate(log_wB[::-1])[::-1]

        # Marginal at row j0
        log_FB = log_F_at_j0 = None
        # Recompute F at j0 and B at j0
        log_F2 = log_w[0, :].copy()
        for j in range(1, j0 + 1):
            log_prefix = np.logaddexp.accumulate(log_F2)
            log_F2 = log_w[j, :] + log_prefix

        log_B2 = np.zeros(Kp1)
        for j in range(m - 2, j0 - 1, -1):
            log_wB = log_w[j + 1, :] + log_B2
            log_B2 = np.logaddexp.accumulate(log_wB[::-1])[::-1]

        log_FB = log_F2 + log_B2  # (K+1,)
        log_Z = logsumexp(log_FB)

        # E[k/K] = sum_k (k/K) * exp(log_FB[k] - log_Z)
        log_probs = log_FB - log_Z
        probs = np.exp(np.clip(log_probs, -500, 0))
        return float(np.dot(k_vals / K, probs))


# ===================================================================
# FTL  (Appendix H)
# ===================================================================

class FTL:
    """Follow-the-Leader on [m]x[m].

    Each round: solve weighted 2D isotonic regression on accumulated data.
    Prediction at (i0,j0) = isotonic regression value.
    """

    def __init__(self, m: int):
        self.m = m
        self.S = np.zeros((m, m), dtype=np.float64)
        self.N = np.zeros((m, m), dtype=np.float64)
        self._solution = np.full((m, m), 0.5)

    def reset(self):
        self.S[:] = 0.0
        self.N[:] = 0.0
        self._solution[:] = 0.5

    def predict(self, i0: int, j0: int) -> float:
        return float(self._solution[i0, j0])

    def update(self, i0: int, j0: int, y: float):
        self.S[i0, j0] += y
        self.N[i0, j0] += 1.0
        self._refit()

    def _refit(self):
        """Solve weighted 2D isotonic regression via alternating PAVA."""
        m = self.m
        w = self.N.copy()
        z = np.where(w > 0, self.S / np.maximum(w, 1e-30), 0.5)

        f = self._solution.copy()
        # Warm-start from previous solution; only a few iterations needed.
        for _ in range(6):
            f_old = f.copy()
            # Row-wise PAVA: non-decreasing in i for each j
            for j in range(m):
                f[:, j] = _pava_nondecreasing(f[:, j], z[:, j], w[:, j])
            # Column-wise PAVA: non-decreasing in j for each i
            for i in range(m):
                f[i, :] = _pava_nondecreasing(f[i, :], z[i, :], w[i, :])
            if np.max(np.abs(f - f_old)) < 1e-12:
                break
        self._solution = f


def _pava_nondecreasing(f_init, z, w):
    """Weighted PAVA projecting onto non-decreasing sequences.

    Minimises sum_i w[i]*(f[i]-z[i])^2  s.t. f non-decreasing,
    using f_init as warm-start hint (ignored; exact PAVA is used).

    Zero-weight elements are interpolated to maintain monotonicity.
    """
    n = len(z)
    # Blocks: (weighted_sum, total_weight, start_idx)
    ws = w * z
    blocks_ws = []
    blocks_w = []
    blocks_end = []

    for i in range(n):
        blocks_ws.append(ws[i])
        blocks_w.append(w[i])
        blocks_end.append(i)
        # Merge while violation exists
        while len(blocks_ws) >= 2:
            s1, w1 = blocks_ws[-2], blocks_w[-2]
            s2, w2 = blocks_ws[-1], blocks_w[-1]
            tw = w1 + w2
            if tw < 1e-30:
                # Both zero-weight: merge
                blocks_ws.pop()
                blocks_w.pop()
                end = blocks_end.pop()
                blocks_end[-1] = end
            elif w1 < 1e-30:
                # Left block zero-weight: absorb into right
                blocks_ws.pop()
                blocks_w.pop()
                end = blocks_end.pop()
                blocks_ws[-1] = s2
                blocks_w[-1] = w2
                blocks_end[-1] = end
                break
            elif w2 < 1e-30:
                # Right block zero-weight: absorb into left
                blocks_ws.pop()
                blocks_w.pop()
                end = blocks_end.pop()
                blocks_end[-1] = end
            else:
                mean1 = s1 / w1
                mean2 = s2 / w2
                if mean1 > mean2 + 1e-15:
                    # Merge
                    blocks_ws.pop()
                    blocks_w.pop()
                    end = blocks_end.pop()
                    blocks_ws[-1] = s1 + s2
                    blocks_w[-1] = w1 + w2
                    blocks_end[-1] = end
                else:
                    break

    # Reconstruct
    result = np.empty(n)
    start = 0
    for idx in range(len(blocks_ws)):
        end = blocks_end[idx] + 1
        bw = blocks_w[idx]
        if bw > 1e-30:
            result[start:end] = blocks_ws[idx] / bw
        else:
            result[start:end] = np.nan  # mark for interpolation
        start = end

    # Interpolate NaN blocks to maintain monotonicity
    _interpolate_nans(result)
    return np.clip(result, 0.0, 1.0)


def _interpolate_nans(f):
    """Fill NaN entries to maintain monotonicity in-place."""
    n = len(f)
    # Forward fill: NaN gets value of last non-NaN to the left (or 0)
    last_val = 0.0
    for i in range(n):
        if np.isnan(f[i]):
            f[i] = last_val
        else:
            last_val = f[i]
    # Backward pass: ensure non-decreasing
    for i in range(n - 2, -1, -1):
        if f[i] > f[i + 1]:
            f[i] = f[i + 1]


# ===================================================================
# Anytime Combined Strategy  (Section 5.3)
# ===================================================================

class AnytimeCombined:
    """Rate-doubling anytime wrapper over Surrogate TI-EW and Column-Ind EW."""

    def __init__(self, m: int, T_max: int):
        self.m = m
        self.T_max = T_max
        self.schedule = compute_epoch_schedule(m, T_max)
        self._current_epoch = 0
        self._rounds_in_epoch = 0
        self._algo = None
        self._start_new_epoch()

    def reset(self):
        self._current_epoch = 0
        self._rounds_in_epoch = 0
        self._start_new_epoch()

    def predict(self, i0: int, j0: int) -> float:
        return self._algo.predict(i0, j0)

    def update(self, i0: int, j0: int, y: float):
        self._algo.update(i0, j0, y)
        self._rounds_in_epoch += 1
        if self._rounds_in_epoch >= self.schedule[self._current_epoch]:
            self._current_epoch += 1
            if self._current_epoch < len(self.schedule):
                self._rounds_in_epoch = 0
                self._start_new_epoch()

    def _start_new_epoch(self):
        # Paper's rule (Appendix F.3): within epoch k of length L_k, run
        # Column-Ind EW at K*_col(L_k) if Psi_col(L_k) <= Psi_surr(L_k),
        # and Surrogate TI-EW at K*_surr(L_k) otherwise.
        L = self.schedule[self._current_epoch]
        ps, Ks = psi_surr(self.m, L, cap=False)
        pc, Kc = psi_col(self.m, L, cap=False)
        if pc <= ps:
            self._algo = ColumnIndEW(self.m, max(Kc, 1))
        else:
            self._algo = SurrogateTIEW(self.m, max(Ks, 1))


# ===================================================================
# Brute-force EW (for correctness verification, small m and K only)
# ===================================================================

def brute_force_surrogate_ew(m, K, queries, labels):
    """Brute-force surrogate TI-EW by enumerating all C(2m,m)^K generalized experts.

    Returns array of predictions (length T).
    Only feasible for very small m (m<=3) and K (K<=3).
    """
    from itertools import product as iproduct

    # Enumerate all upsets of [m]x[m]
    # Upset = non-increasing cutoff sequence c_0 >= c_1 >= ... >= c_{m-1}, c_i in {0,...,m}
    upsets = []
    _enum_cutoffs(m, m, 0, [], upsets)

    n_upsets = len(upsets)
    # Generalised experts: K-tuples of upsets
    experts = list(iproduct(range(n_upsets), repeat=K))
    n_experts = len(experts)

    T = len(queries)
    predictions = np.zeros(T)
    log_weights = np.zeros(n_experts)  # uniform prior => log(1) = 0

    for t in range(T):
        i0, j0 = queries[t]
        y = labels[t]

        # Compute prediction for each expert
        expert_preds = np.zeros(n_experts)
        for e_idx, expert in enumerate(experts):
            val = 0.0
            for r in range(K):
                u_idx = expert[r]
                cutoffs = upsets[u_idx]
                # (i0, j0) in upset iff j0 >= cutoffs[i0]  (0-indexed j, cutoff c)
                # Paper: (i+1, j+1) in U iff j+1 > c_{i+1} iff c <= j0
                if cutoffs[i0] <= j0:
                    val += 1.0
            expert_preds[e_idx] = val / K

        # EW prediction (exp-concave with eta=1/2)
        log_probs = log_weights - logsumexp(log_weights)
        probs = np.exp(log_probs)
        predictions[t] = np.dot(probs, expert_preds)

        # Surrogate loss for each expert
        for e_idx, expert in enumerate(experts):
            surr_loss = y * y
            for r in range(K):
                r_paper = r + 1
                u_idx = expert[r]
                cutoffs = upsets[u_idx]
                z = 1.0 if cutoffs[i0] <= j0 else 0.0
                surr_loss += ((2 * r_paper - 1) / K**2 - 2 * y / K) * z
            log_weights[e_idx] -= 0.5 * surr_loss

    return predictions


def brute_force_column_ew(m, K, queries, labels):
    """Brute-force column-independent EW by enumerating all C(m+K,K)^m experts.

    Returns array of predictions (length T).
    """
    from itertools import product as iproduct

    # Enumerate monotone functions on chain [m] -> {0,...,K}
    mono_funcs = []
    _enum_mono(m, K, 0, 0, [], mono_funcs)

    n_mono = len(mono_funcs)
    # Column-independent expert: one mono func per column
    # For brute force, we run each column independently.

    T = len(queries)
    predictions = np.zeros(T)

    # Per-column log-weights
    col_log_weights = [np.zeros(n_mono) for _ in range(m)]

    for t in range(T):
        i0, j0 = queries[t]
        y = labels[t]

        # Prediction from column i0
        lw = col_log_weights[i0]
        log_probs = lw - logsumexp(lw)
        probs = np.exp(log_probs)

        # Each mono func's prediction at row j0
        mono_preds = np.array([f[j0] / K for f in mono_funcs])
        predictions[t] = np.dot(probs, mono_preds)

        # Update column i0: standard EW with square loss
        for f_idx, f in enumerate(mono_funcs):
            pred = f[j0] / K
            loss = (pred - y) ** 2
            col_log_weights[i0][f_idx] -= 0.5 * loss

    return predictions


def _enum_cutoffs(m, max_c, col, current, result):
    """Enumerate non-increasing cutoff sequences of length m, values in {0,...,max_c_init}."""
    if col == m:
        result.append(tuple(current))
        return
    upper = max_c if col == 0 else current[-1]
    for c in range(upper, -1, -1):
        current.append(c)
        _enum_cutoffs(m, max_c, col + 1, current, result)
        current.pop()


def _enum_mono(m, K, row, min_val, current, result):
    """Enumerate non-decreasing sequences of length m, values in {0,...,K}."""
    if row == m:
        result.append(tuple(current))
        return
    for v in range(min_val, K + 1):
        current.append(v)
        _enum_mono(m, K, row + 1, v, current, result)
        current.pop()
