"""Build Submission.ipynb following the exact Sample Assignment template.

Best per-stock configuration we verified across grid search on s1..s30:
  - 18 engineered features (returns, MA ratios, vol, RSI, MACD, OBV, vol-z, interactions)
  - walk-forward expanding ridge with periodic refit (every 5 days, warmup 60)
  - LAM=5.0, train-only z-scoring (no leakage), no penalty on intercept
  - SHRINK=0.10 (verified optimal on grid 0.00..1.00 — see improve_grid.py)
  - CLIP=0.03 on output
  - ytr clipped to [-0.05, 0.05] before fit (defensive against outlier days)
  - uses ADJUSTED close (col 7), not raw close (col 2)

Result on 30 sample stocks:
  median abs = 0.0177    median rel = +0.0067    70% of stocks have rel > 0
"""
import json
from pathlib import Path

cells = []

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

# 1. authors
cells.append(code("""# 1. authors
# Nguyen Van Hoang Khang
"""))

# 2. warning
cells.append(code("""# 2. warning:
# only customize the 3-4-5 blocks
"""))

# 3. library
cells.append(code("""# 3. library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
"""))

# 4. input data
cells.append(code("""# 4. input your data here
COL_pv = ['date', 'opn', 'cls', 'low', 'high', 'nsh', 'vol', 'adj']

base_dir = Path.cwd()
home = str(base_dir)

stk = 's1'  # change to s2..s30 to test other stocks

# Try the assignment's standard layout first; fall back to sample_data/
legacy_path  = base_dir / 'data' / 'pv' / f"{stk}.npy"
default_path = base_dir / 'sample_data' / f"{stk}.npy"
A = np.load(legacy_path if legacy_path.exists() else default_path, allow_pickle=True)

# Use ADJUSTED close (col 7) instead of raw close (col 2):
# raw close has artificial jumps from splits/dividends that contaminate returns.
P = A[:, 7].astype(float)
V = A[:, 6].astype(float)
"""))

# 5. customize your prediction (THE ACTUAL SUBMISSION)
cells.append(code("""# 5. customize your prediction
def prediction(P, V, h=20):
    # Walk-forward ridge predictor with engineered features.
    # For each day i, train on (X[k], r[k+1]) for k < i and predict the
    # next-day return r[i+1] from features X[i] computed using only data
    # up to day i. Strictly causal — no look-ahead.

    EPS    = 1e-12
    CLIP   = 0.03
    SHRINK = 0.10
    LAM    = 5.0
    REFIT  = 5
    WARM   = max(60, 3 * h)

    P = np.asarray(P, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(P)
    if n == 0:
        return []

    # ----- 1. simple returns -----
    r = np.zeros(n)
    r[1:] = P[1:] / np.maximum(P[:-1], EPS) - 1.0

    # ----- 2. causal rolling helpers -----
    def rmean(a, w):
        out = np.zeros_like(a, dtype=float)
        c = np.cumsum(np.insert(a, 0, 0.0))
        for i in range(len(a)):
            j = max(0, i - w + 1)
            out[i] = (c[i + 1] - c[j]) / (i - j + 1)
        return out

    def rstd(a, w):
        out = np.zeros_like(a, dtype=float)
        for i in range(len(a)):
            j = max(0, i - w + 1)
            out[i] = np.std(a[j:i + 1]) if i > j else 0.0
        return out

    def ema(a, span):
        al = 2.0 / (span + 1.0)
        out = np.zeros_like(a, dtype=float); out[0] = a[0]
        for i in range(1, len(a)):
            out[i] = al * a[i] + (1 - al) * out[i - 1]
        return out

    # ----- 3. feature engineering -----
    ma5  = rmean(P, 5);   ma10 = rmean(P, 10)
    ma20 = rmean(P, 20);  ma50 = rmean(P, 50)
    vol5  = rstd(r, 5)  + EPS
    vol20 = rstd(r, 20) + EPS

    logV   = np.log(np.maximum(V, 1.0))
    vma20  = rmean(logV, 20)
    vstd20 = rstd(logV, 20) + EPS

    # RSI(14)
    d  = np.diff(P, prepend=P[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    rsi14 = 100.0 - 100.0 / (1.0 + rmean(up, 14) / (rmean(dn, 14) + EPS))

    # MACD(12, 26) histogram, normalized by price
    macd      = ema(P, 12) - ema(P, 26)
    macd_sig  = ema(macd, 9)
    macd_hist = (macd - macd_sig) / np.maximum(P, EPS)

    # OBV z-scored
    obv = np.zeros(n)
    for i in range(1, n):
        sgn = 1.0 if P[i] > P[i - 1] else (-1.0 if P[i] < P[i - 1] else 0.0)
        obv[i] = obv[i - 1] + sgn * V[i]
    obv_z = (obv - rmean(obv, 20)) / (rstd(obv, 20) + EPS)

    feats = np.column_stack([
        r,                                       # 0  lag-1 return
        np.r_[0.0, r[:-1]],                      # 1  lag-2 return
        np.r_[0.0, 0.0, r[:-2]],                 # 2  lag-3 return
        r / vol20,                               # 3  vol-normalized return
        P / np.maximum(ma5,  EPS) - 1.0,         # 4  short mean-reversion
        P / np.maximum(ma10, EPS) - 1.0,         # 5
        P / np.maximum(ma20, EPS) - 1.0,         # 6  medium mean-reversion
        P / np.maximum(ma50, EPS) - 1.0,         # 7  long trend
        ma5  / np.maximum(ma20, EPS) - 1.0,      # 8  short/medium momentum
        ma10 / np.maximum(ma50, EPS) - 1.0,      # 9  medium/long momentum
        vol5 / vol20 - 1.0,                      # 10 vol-of-vol
        np.log(vol20 + EPS),                     # 11 vol level
        (logV - vma20) / vstd20,                 # 12 volume z-score
        np.r_[0.0, logV[1:] - logV[:-1]],        # 13 log-volume change
        rsi14 / 100.0 - 0.5,                     # 14 RSI centered at 0
        macd_hist,                               # 15 MACD histogram
        obv_z,                                   # 16 OBV z-score
        r * ((logV - vma20) / vstd20),           # 17 price-volume interaction
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- 4. walk-forward ridge with periodic refit -----
    pred = np.zeros(n)
    beta = mu = sd = None
    last_fit = -10**9

    for i in range(n):
        if i < WARM:
            pred[i] = 0.0
            continue

        if (i - last_fit) >= REFIT or beta is None:
            Xtr = feats[:i]
            ytr = r[1:i + 1]
            mu  = Xtr.mean(axis=0)
            sd  = Xtr.std(axis=0) + EPS
            Xs  = (Xtr - mu) / sd
            Xs  = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
            I   = np.eye(Xs.shape[1]); I[0, 0] = 0.0
            try:
                beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            except np.linalg.LinAlgError:
                beta = np.zeros(Xs.shape[1])
            last_fit = i

        xf   = (feats[i] - mu) / sd
        xf   = np.concatenate([[1.0], xf])
        yhat = SHRINK * float(xf @ beta)
        if yhat >  CLIP: yhat =  CLIP
        if yhat < -CLIP: yhat = -CLIP
        pred[i] = yhat

    return pred.tolist()
"""))

# 6. evaluate (unchanged from sample)
cells.append(code("""# 6. keep the core function unchanged
def target(P, V):
    n, Q = len(P), [0]
    for i in range(1, n):
        Q.append(P[i] / P[i - 1] - 1)
    return Q

def evaluate(p, t, dspl=False):
    p, t = p[1:], t[1:]
    n, e, f = len(t), [], []
    for i in range(1, n):
        e.append(t[i] - p[i - 1])
        f.append(t[i])
    den = np.nanquantile(np.abs(e), 0.5) + 0.5 * np.nanquantile(np.abs(e), 0.9)
    num = np.nanquantile(np.abs(f), 0.5) + 0.5 * np.nanquantile(np.abs(f), 0.9)
    if dspl:
        print(f"\\n\\tbase = {round(num, 3)}  |  abs = {round(den, 3)}  |  rel = {round(1 - den / num, 3)}\\n")
        plt.hist(e, edgecolor='black')
        plt.show()
    else:
        return den, 1 - den / num
"""))

# 7. execute on the loaded stock
cells.append(code("""# 7. execute
# rel > 0 = useful signal, abs as small as possible (lower bound ~ q50(|t|)+0.5*q90(|t|))
p = prediction(P, V, 5)
t = target(P, V)
evaluate(p, t, True)
"""))

# 8. (optional) batch evaluation across all 30 sample stocks
cells.append(code("""# 8. (optional) — verify across all 30 sample stocks
results = []
for k in range(1, 31):
    name = f's{k}'
    fp = base_dir / 'sample_data' / f'{name}.npy'
    if not fp.exists():
        continue
    Ak = np.load(fp, allow_pickle=True)
    Pk = Ak[:, 7].astype(float)
    Vk = Ak[:, 6].astype(float)
    pk = prediction(Pk, Vk, 5)
    tk = target(Pk, Vk)
    ab, rl = evaluate(pk, tk, False)
    results.append({'stk': name, 'n': len(Pk), 'abs': ab, 'rel': rl})

df = pd.DataFrame(results)
print(df.round(4).to_string(index=False))
print(f"\\n  median abs = {df['abs'].median():.4f}    median rel = {df['rel'].median():+.4f}")
print(f"  mean   abs = {df['abs'].mean():.4f}    mean   rel = {df['rel'].mean():+.4f}")
print(f"  % stocks with rel > 0: {(df['rel'] > 0).mean()*100:.1f}%")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("Submission.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} — {len(cells)} cells")
