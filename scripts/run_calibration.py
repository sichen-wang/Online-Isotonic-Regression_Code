#!/usr/bin/env python3
"""E2: online two-feature probability calibration on real data (UCI Adult).

Scenario (i) of the paper's introduction: a deployed classifier's calibration
map is jointly nondecreasing in the model score and in an auxiliary covariate
the model does not use. The classifier is trained without education, and
education-num (16 ordinal levels) is the auxiliary covariate.

Protocol: train the classifier on a held-out split; on the remaining stream,
bin the score into m quantiles and the covariate into its natural levels;
process the stream online under squared loss.

Learners
  surr-2d    Surrogate TI-EW on [m]^2, fixed horizon      (both features, full 2D monotone structure)
  col-2d     Column-Indep Chain EW on [m]^2, fixed horizon(both features, monotone within score bins)
  anytime-2d the paper's horizon-free wrapper             (same, without knowing T)
  chain-1d   chain EW on the score bins alone             (the predecessor's setting)
  cell-1d    per-score-bin smoothed running mean          (score only, no structure)
  cell-2d    per-cell smoothed running mean on [m]^2      (both features, no structure)
  const      global running mean                          (neither)

Oracle rows bound what is achievable: the in-sample Bayes loss using both
features and using the score alone.

Supports: two-dimensional monotone structure has incremental value on a real
calibration task. Does NOT support: anything about worst-case optimality.

Usage:
    python run_calibration.py --m 16 --seeds 20 --diagnose
    python run_calibration.py --m 16 --seeds 20 --workers 16
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import lgamma
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.algorithms import SurrogateTIEW, ColumnIndEW, AnytimeCombined
from src.theory import psi_surr, psi_col

COLS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'income']
EXCLUDED = ['education', 'education-num', 'income', 'y', 'fnlwgt']
LEARNERS = ['surr-2d', 'col-2d', 'anytime-2d', 'chain-1d', 'cell-1d',
            'cell-2d', 'const']


def load_adult(path):
    z = zipfile.ZipFile(path)
    tr = pd.read_csv(z.open('adult.data'), header=None, names=COLS,
                     skipinitialspace=True, na_values='?')
    te = pd.read_csv(z.open('adult.test'), header=None, names=COLS,
                     skipinitialspace=True, na_values='?', skiprows=1)
    df = pd.concat([tr, te], ignore_index=True).dropna().reset_index(drop=True)
    df['y'] = (df['income'].str.replace('.', '', regex=False).str.strip()
               == '>50K').astype(int)
    return df


def fit_scores(df, seed, train_frac=0.4, kind='lr'):
    """Train the base classifier without the covariate.

    kind='gbm' is the robustness check against the objection that a weak base
    model leaves artificially much signal in the covariate.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n_tr = int(train_frac * len(df))
    tr, st = df.iloc[idx[:n_tr]], df.iloc[idx[n_tr:]].reset_index(drop=True)
    feats = [c for c in df.columns if c not in EXCLUDED]
    cat = [c for c in feats if df[c].dtype == object]
    num = [c for c in feats if c not in cat]

    if kind == 'gbm':
        from sklearn.ensemble import HistGradientBoostingClassifier
        pre = ColumnTransformer(
            [('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
             ('n', 'passthrough', num)])
        clf = make_pipeline(pre, HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed))
    else:
        from sklearn.linear_model import LogisticRegression
        pre = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore'), cat),
                                 ('n', StandardScaler(), num)])
        clf = make_pipeline(pre, LogisticRegression(max_iter=2000))
    clf.fit(tr[feats], tr['y'])
    return st, clf.predict_proba(st[feats])[:, 1]


def score_bins(values, m):
    return np.asarray(pd.qcut(pd.Series(values).rank(method='first'), m,
                              labels=False, duplicates='drop'), dtype=int)


def cov_bins(values, m):
    """Natural ordinal levels of education-num, collapsed to m groups if needed."""
    v = np.asarray(values, dtype=int)
    levels = np.sort(np.unique(v))
    if len(levels) <= m:
        lut = {lv: k for k, lv in enumerate(levels)}
        return np.array([lut[x] for x in v], dtype=int)
    edges = np.linspace(0, len(levels), m + 1).astype(int)
    lut = {}
    for k in range(m):
        for lv in levels[edges[k]:edges[k + 1]]:
            lut[lv] = k
    return np.array([lut[x] for x in v], dtype=int)


def chain_K(m, T):
    best, bk = None, 1
    for K in range(1, int(4 * T ** 0.5) + 2):
        lc = lgamma(m + K + 1) - lgamma(K + 1) - lgamma(m + 1)
        v = 2 * lc + T / (4.0 * K * K)
        if best is None or v < best:
            best, bk = v, K
    return bk


def oracle_losses(i, j, y, m):
    """In-sample Bayes squared loss using both features and the score alone."""
    n2 = np.zeros((m, m)); s2 = np.zeros((m, m))
    n1 = np.zeros(m); s1 = np.zeros(m)
    for a, b, lab in zip(i, j, y):
        n2[a, b] += 1; s2[a, b] += lab
        n1[a] += 1; s1[a] += lab
    p2 = np.where(n2 > 0, s2 / np.maximum(n2, 1), 0.0)
    p1 = np.where(n1 > 0, s1 / np.maximum(n1, 1), 0.0)
    return (float(np.sum(n2 * p2 * (1 - p2))), float(np.sum(n1 * p1 * (1 - p1))))


def run_seed(task):
    m, seed, data_path = task[:3]
    placebo = task[3] if len(task) > 3 else False
    t_frac = task[4] if len(task) > 4 else 1.0
    clf_kind = task[5] if len(task) > 5 else 'lr'
    df = load_adult(data_path)
    st, sc = fit_scores(df, seed=seed, kind=clf_kind)
    i_all = score_bins(sc, m)
    j_all = cov_bins(st['education-num'].values, m)
    if placebo:
        # Destroy the covariate signal, keep its marginal, the grid, and the
        # algorithms: isolates how much of the 2D gain is the covariate.
        j_all = np.random.RandomState(7 * seed + 3).permutation(j_all)
    y_all = st['y'].values.astype(float)

    rng = np.random.RandomState(1000 + seed)
    order = rng.permutation(len(y_all))
    if t_frac < 1.0:
        order = order[:max(1, int(t_frac * len(order)))]
    i, j, y = i_all[order], j_all[order], y_all[order]
    T = len(y)

    res = {'seed': seed, 'T': T}
    o2, o1 = oracle_losses(i, j, y, m)
    res['oracle-2d'], res['oracle-1d'] = o2, o1

    _, Ks = psi_surr(m, T); Ks = max(int(Ks), 1)
    _, Kc = psi_col(m, T); Kc = max(int(Kc), 1)
    K1 = chain_K(m, T)
    res['_K'] = dict(surr=Ks, col=Kc, chain=K1)

    def stream(pred_update):
        loss = 0.0
        for a, b, lab in zip(i, j, y):
            p = pred_update(a, b, None)
            loss += (p - lab) ** 2
            pred_update(a, b, lab)
        return loss

    algo = SurrogateTIEW(m, Ks)
    res['surr-2d'] = stream(lambda a, b, lab: algo.predict(a, b) if lab is None
                            else algo.update(a, b, lab))
    algo = ColumnIndEW(m, Kc)
    res['col-2d'] = stream(lambda a, b, lab: algo.predict(a, b) if lab is None
                           else algo.update(a, b, lab))
    algo = AnytimeCombined(m, T)
    res['anytime-2d'] = stream(lambda a, b, lab: algo.predict(a, b) if lab is None
                               else algo.update(a, b, lab))
    algo = ColumnIndEW(m, K1)
    res['chain-1d'] = stream(lambda a, b, lab: algo.predict(0, a) if lab is None
                             else algo.update(0, a, lab))

    s = np.zeros((m, m)); n = np.zeros((m, m)); loss = 0.0
    for a, b, lab in zip(i, j, y):
        p = (s[a, b] + 0.5) / (n[a, b] + 1.0)
        loss += (p - lab) ** 2
        s[a, b] += lab; n[a, b] += 1
    res['cell-2d'] = loss

    s1 = np.zeros(m); n1 = np.zeros(m); loss = 0.0
    for a, lab in zip(i, y):
        p = (s1[a] + 0.5) / (n1[a] + 1.0)
        loss += (p - lab) ** 2
        s1[a] += lab; n1[a] += 1
    res['cell-1d'] = loss

    ts, tn, loss = 0.0, 0.0, 0.0
    for lab in y:
        p = (ts + 0.5) / (tn + 1.0)
        loss += (p - lab) ** 2
        ts += lab; tn += 1
    res['const'] = loss
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--placebo", action="store_true",
                    help="shuffle the covariate to destroy its signal")
    ap.add_argument("--t-frac", type=float, default=1.0,
                    help="use this fraction of the stream (moves along the phase axis)")
    ap.add_argument("--clf", choices=["lr", "gbm"], default="lr")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data = args.data or str(root / "data" / "adult.zip")
    tag = ("_placebo" if args.placebo else "") + \
          (f"_t{args.t_frac:g}" if args.t_frac < 1.0 else "") + \
          ("_gbm" if args.clf == "gbm" else "")
    out = args.output or str(root / "results" / f"calibration_m{args.m}{tag}.json")
    m = args.m

    if args.diagnose:
        df = load_adult(data)
        st, sc = fit_scores(df, seed=0)
        i = score_bins(sc, m); j = cov_bins(st['education-num'].values, m)
        y = st['y'].values.astype(float)
        T = len(y)
        o2, o1 = oracle_losses(i, j, y, m)
        cnt = np.zeros((m, m))
        for a, b in zip(i, j):
            cnt[a, b] += 1
        print(f"n={len(df)}, stream T={T}, m={m}, cells={m*m}, "
              f"T/cells={T/m**2:.1f}, T vs m^4={T/m**4:.4f}")
        print(f"cells with 0 obs: {int((cnt==0).sum())}, "
              f"min {int(cnt.min())}, median {int(np.median(cnt))}, max {int(cnt.max())}")
        print(f"oracle 2-feature loss {o2:.1f} ({o2/T:.5f}/round)")
        print(f"oracle score-only loss {o1:.1f} ({o1/T:.5f}/round)")
        print(f"maximum achievable gain from the covariate: {o1-o2:.1f} "
              f"({100*(o1-o2)/o1:.2f}% of the score-only oracle)")
        _, Ks = psi_surr(m, T); _, Kc = psi_col(m, T)
        print(f"grid levels: surr K={Ks}, col K={Kc}, chain K={chain_K(m,T)}")
        return

    tasks = [(m, s, data, args.placebo, args.t_frac, args.clf)
             for s in range(args.seeds)]
    print(f"=== E2 calibration ===  m={m} seeds={args.seeds} "
          f"workers={args.workers} placebo={args.placebo} t_frac={args.t_frac}",
          flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, args.seeds)) as ex:
        futs = [ex.submit(run_seed, t) for t in tasks]
        for f in as_completed(futs):
            rows.append(f.result())
            print(f"  {len(rows)}/{args.seeds} done ({time.time()-t0:.0f}s)", flush=True)

    T = rows[0]['T']
    print(f"\n=== m={m}, T={T}, {args.seeds} seeds: cumulative squared loss ===")
    print(f"{'learner':12s} {'total':>16s} {'per round':>11s} {'vs chain-1d':>12s}")
    base = float(np.mean([r['chain-1d'] for r in rows]))
    summary = {}
    for k in LEARNERS + ['oracle-2d', 'oracle-1d']:
        v = np.array([r[k] for r in rows])
        sem = v.std(ddof=1) / np.sqrt(len(v))
        summary[k] = dict(mean=float(v.mean()), sem=float(sem),
                          per_round=float(v.mean() / T))
        print(f"{k:12s} {v.mean():11.2f}+-{sem:5.2f} {v.mean()/T:11.5f} "
              f"{100*(v.mean()-base)/base:+11.2f}%")
    print()
    for k in ('surr-2d', 'col-2d', 'anytime-2d'):
        for ref in ('chain-1d', 'cell-2d'):
            d = np.array([r[ref] - r[k] for r in rows])
            print(f"{k:11s} beats {ref:9s} by {d.mean():8.2f} +- "
                  f"{d.std(ddof=1)/np.sqrt(len(d)):5.2f}   "
                  f"(wins {int((d>0).sum())}/{len(d)})")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(m=m, T=T, seeds=args.seeds, summary=summary, rows=rows),
              open(out, 'w'), indent=1)
    print(f"\nWall {time.time()-t0:.0f}s   saved -> {out}")


if __name__ == "__main__":
    main()
