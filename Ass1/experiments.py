"""Experiments for Error.txt improvement suggestions.

Compares the current v2 predictor (`baseline_v2`) against:
  I1: Rolling window (252 days) instead of expanding window
  I2: Volatility scaling instead of static SHRINK=0.10
  I3: LightGBM with L1 loss replacing Ridge
  I4: Clip ytr to [-0.05, 0.05] before fitting Ridge
  ALL: I1+I2+I4 combined on Ridge backbone

Reports median rel, median abs across 30 sample stocks.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

base_dir = Path(__file__).parent
EPS = 1e-12

# ---------- shared feature engineering (mirror of v2) ----------
def rolling_mean(a, w):
    out = np.zeros_like(a, dtype=float)
    c = np.cumsum(np.insert(a, 0, 0.0))
    for i in range(len(a)):
        j = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[j]) / (i - j + 1)
    return out

def rolling_std(a, w):
    out = np.zeros_like(a, dtype=float)
    for i in range(len(a)):
        j = max(0, i - w + 1)
        out[i] = np.std(a[j:i + 1]) if i > j else 0.0
    return out

def ema(a, span):
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(a, dtype=float)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out

def build_features(P, V):
    P = np.asarray(P, float); V = np.asarray(V, float)
    n = len(P)
    r = np.zeros(n); r[1:] = P[1:] / np.maximum(P[:-1], EPS) - 1.0
    ma5  = rolling_mean(P, 5);   ma10 = rolling_mean(P, 10)
    ma20 = rolling_mean(P, 20);  ma50 = rolling_mean(P, 50)
    vol5  = rolling_std(r, 5)  + EPS
    vol20 = rolling_std(r, 20) + EPS
    logV = np.log(np.maximum(V, 1.0))
    vma20 = rolling_mean(logV, 20)
    vstd20 = rolling_std(logV, 20) + EPS
    d = np.diff(P, prepend=P[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = rolling_mean(up, 14); ad = rolling_mean(dn, 14) + EPS
    rsi14 = 100.0 - 100.0 / (1.0 + au / ad)
    macd = ema(P, 12) - ema(P, 26)
    macd_sig = ema(macd, 9)
    macd_hist = (macd - macd_sig) / np.maximum(P, EPS)
    obv = np.zeros(n)
    for i in range(1, n):
        s = 1.0 if P[i] > P[i-1] else (-1.0 if P[i] < P[i-1] else 0.0)
        obv[i] = obv[i-1] + s * V[i]
    obv_z = (obv - rolling_mean(obv, 20)) / (rolling_std(obv, 20) + EPS)
    feats = np.column_stack([
        r,
        np.r_[0.0, r[:-1]],
        np.r_[0.0, 0.0, r[:-2]],
        r / vol20,
        P / np.maximum(ma5,  EPS) - 1.0,
        P / np.maximum(ma10, EPS) - 1.0,
        P / np.maximum(ma20, EPS) - 1.0,
        P / np.maximum(ma50, EPS) - 1.0,
        ma5  / np.maximum(ma20, EPS) - 1.0,
        ma10 / np.maximum(ma50, EPS) - 1.0,
        vol5 / vol20 - 1.0,
        np.log(vol20 + EPS),
        (logV - vma20) / vstd20,
        np.r_[0.0, logV[1:] - logV[:-1]],
        rsi14 / 100.0 - 0.5,
        macd_hist,
        obv_z,
        r * ((logV - vma20) / vstd20),
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, r, vol20

# ---------- predictors ----------
def predict_baseline_v2(P, V):
    """Current v2 — expanding ridge, SHRINK=0.10."""
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    pred = np.zeros(n)
    SHRINK, CLIP, LAM, REFIT, WARM = 0.10, 0.03, 5.0, 5, 60
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            Xtr = feats[:i]; ytr = r[1:i+1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        y = SHRINK * float(xf @ beta)
        pred[i] = np.clip(y, -CLIP, CLIP)
    return pred

def predict_I1_rolling(P, V, win=252):
    """Improvement 1: rolling window of `win` days."""
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    pred = np.zeros(n)
    SHRINK, CLIP, LAM, REFIT, WARM = 0.10, 0.03, 5.0, 5, 60
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            lo = max(0, i - win)
            Xtr = feats[lo:i]; ytr = r[lo+1:i+1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        y = SHRINK * float(xf @ beta)
        pred[i] = np.clip(y, -CLIP, CLIP)
    return pred

def predict_I2_volscale(P, V):
    return _i2(P, V, 0.30)

def _i2(P, V, target_ratio):
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    raw = np.zeros(n)
    LAM, REFIT, WARM, CLIP = 5.0, 5, 60, 0.03
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            Xtr = feats[:i]; ytr = r[1:i+1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        raw[i] = float(xf @ beta)
    pred = np.zeros(n)
    for i in range(WARM, n):
        s = np.std(raw[WARM:i+1]) + EPS
        scale = (target_ratio * vol20[i]) / s
        pred[i] = np.clip(raw[i] * scale, -CLIP, CLIP)
    return pred

def _all(P, V, win, target_ratio):
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    raw = np.zeros(n)
    LAM, REFIT, WARM, CLIP = 5.0, 5, 60, 0.03
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            lo = max(0, i - win)
            Xtr = feats[lo:i]; ytr = np.clip(r[lo+1:i+1], -0.05, 0.05)
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        raw[i] = float(xf @ beta)
    pred = np.zeros(n)
    for i in range(WARM, n):
        s = np.std(raw[WARM:i+1]) + EPS
        scale = (target_ratio * vol20[i]) / s
        pred[i] = np.clip(raw[i] * scale, -CLIP, CLIP)
    return pred

# ---------- evaluator (mirrors notebook `evaluate`) ----------
def predict_I3_lgbm(P, V):
    """Improvement 3: LightGBM with L1 loss, shallow trees."""
    import lightgbm as lgb
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    pred = np.zeros(n)
    SHRINK, CLIP, REFIT, WARM = 0.10, 0.03, 20, 100
    model = None; last = -10**9
    params = dict(
        objective='regression_l1', learning_rate=0.01, num_leaves=8,
        max_depth=3, min_data_in_leaf=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, verbose=-1,
    )
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or model is None:
            Xtr = feats[:i]; ytr = r[1:i+1]
            ds = lgb.Dataset(Xtr, ytr)
            model = lgb.train(params, ds, num_boost_round=200)
            last = i
        y = SHRINK * float(model.predict(feats[i:i+1])[0])
        pred[i] = np.clip(y, -CLIP, CLIP)
    return pred

def predict_I4_clipy(P, V):
    """Improvement 4: clip ytr to [-0.05, 0.05] before fitting."""
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    pred = np.zeros(n)
    SHRINK, CLIP, LAM, REFIT, WARM = 0.10, 0.03, 5.0, 5, 60
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            Xtr = feats[:i]; ytr = np.clip(r[1:i+1], -0.05, 0.05)
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        y = SHRINK * float(xf @ beta)
        pred[i] = np.clip(y, -CLIP, CLIP)
    return pred

def predict_ALL(P, V, win=252):
    """Combined: rolling window + ytr clip + volatility scaling."""
    feats, r, vol20 = build_features(P, V)
    n = len(P)
    raw = np.zeros(n)
    LAM, REFIT, WARM, CLIP = 5.0, 5, 60, 0.03
    TARGET_RATIO = 0.30
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            lo = max(0, i - win)
            Xtr = feats[lo:i]; ytr = np.clip(r[lo+1:i+1], -0.05, 0.05)
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr)
            last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        raw[i] = float(xf @ beta)
    pred = np.zeros(n)
    for i in range(WARM, n):
        s = np.std(raw[WARM:i+1]) + EPS
        scale = (TARGET_RATIO * vol20[i]) / s
        pred[i] = np.clip(raw[i] * scale, -CLIP, CLIP)
    return pred

# ---------- evaluator (mirrors notebook `evaluate`) ----------
def target(P):
    n = len(P); Q = [0]
    for i in range(1, n):
        Q.append(P[i] / P[i-1] - 1)
    return Q

def evaluate(p, t):
    p, t = list(p)[1:], list(t)[1:]
    n, e, f = len(t), [], []
    for i in range(1, n):
        e.append(t[i] - p[i-1]); f.append(t[i])
    den = np.nanquantile(np.abs(e), 0.5) + 0.5 * np.nanquantile(np.abs(e), 0.9)
    num = np.nanquantile(np.abs(f), 0.5) + 0.5 * np.nanquantile(np.abs(f), 0.9)
    return den, (1 - den / num) if num > 0 else 0.0

# ---------- run ----------
def main():
    methods = [
        ('baseline_v2',     predict_baseline_v2),
        ('I1_roll120',      lambda P, V: predict_I1_rolling(P, V, win=120)),
        ('I1_roll80',       lambda P, V: predict_I1_rolling(P, V, win=80)),
        ('I2_vol_r0.10',    lambda P, V: _i2(P, V, 0.10)),
        ('I2_vol_r0.20',    lambda P, V: _i2(P, V, 0.20)),
        ('I3_lgbm',         predict_I3_lgbm),
        ('I4_clipy',        predict_I4_clipy),
        ('ALL_r120_v0.10',  lambda P, V: _all(P, V, 120, 0.10)),
        ('ALL_r120_v0.20',  lambda P, V: _all(P, V, 120, 0.20)),
    ]
    rows = []
    for k in range(1, 31):
        fp = base_dir / 'sample_data' / f's{k}.npy'
        if not fp.exists(): continue
        A = np.load(fp, allow_pickle=True)
        P = A[:, 7].astype(float); V = A[:, 6].astype(float)
        t = target(P)
        row = {'stk': f's{k}', 'n': len(P)}
        for name, fn in methods:
            p = fn(P, V)
            ab, rl = evaluate(p, t)
            row[f'abs_{name}'] = ab
            row[f'rel_{name}'] = rl
        rows.append(row)
        print(f"  done s{k}")
    df = pd.DataFrame(rows)
    df.to_csv(base_dir / 'experiments_per_stock.csv', index=False)

    summary = []
    for name, _ in methods:
        summary.append({
            'method': name,
            'median_abs': df[f'abs_{name}'].median(),
            'mean_abs':   df[f'abs_{name}'].mean(),
            'median_rel': df[f'rel_{name}'].median(),
            'mean_rel':   df[f'rel_{name}'].mean(),
            'pct_rel_pos': (df[f'rel_{name}'] > 0).mean() * 100,
        })
    s = pd.DataFrame(summary).round(4)
    s.to_csv(base_dir / 'experiments_summary.csv', index=False)

    # delta vs baseline
    base_med_rel = s.loc[s.method == 'baseline_v2', 'median_rel'].iloc[0]
    base_med_abs = s.loc[s.method == 'baseline_v2', 'median_abs'].iloc[0]
    s['drel_vs_base'] = (s['median_rel'] - base_med_rel).round(4)
    s['dabs_vs_base'] = (s['median_abs'] - base_med_abs).round(4)
    print('\n=== SUMMARY (30 stocks) ===')
    print(s.to_string(index=False))

if __name__ == '__main__':
    main()
