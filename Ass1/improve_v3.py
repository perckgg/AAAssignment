"""More improvements — what actually moves the score on s1..s30.

Tests:
  M0 baseline v2 (reference)
  M1 cross-sectional pooled ridge (train 1 model on all 30 stocks)
  M2 ensemble: 0.5 * baseline + 0.5 * 0  (further shrink)
  M3 sign-conservative: predict sign(baseline) * min(|baseline|, k*vol20)
  M4 quantile regression at q=0.5 (LAD) instead of OLS ridge
  M5 tighter clip 0.015 + smaller shrink 0.05
  M6 baseline + per-stock auto-shrink (chosen so |pred|.std == 0.3 * |t|.std on warmup)
  M7 baseline-of-baselines: average of v2 + zero  (=M2 essentially)
  M8 pooled LightGBM (cross-sectional)

Reports rel/abs medians.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from experiments import (
    build_features, predict_baseline_v2, target, evaluate, EPS,
)

base_dir = Path(__file__).parent

def load(stk):
    A = np.load(base_dir / 'sample_data' / f'{stk}.npy', allow_pickle=True)
    return A[:, 7].astype(float), A[:, 6].astype(float)

# ---------- pooled cross-sectional ridge ----------
def pooled_ridge_predict(stocks, lam=5.0, refit_every=20, warmup=60, shrink=0.10, clip=0.03):
    """Train 1 ridge on pooled (X, y) from all stocks; refit periodically.
    For each day i (>=warmup), use a model fitted on data up to i-1 from ALL stocks.
    """
    feats_all = {}
    rets_all = {}
    n_max = 0
    for s in stocks:
        P, V = load(s)
        f, r, _ = build_features(P, V)
        feats_all[s] = f; rets_all[s] = r
        n_max = max(n_max, len(r))

    preds = {s: np.zeros(len(rets_all[s])) for s in stocks}
    last_fit = -10**9
    beta = mu = sd = None

    for i in range(n_max):
        if i < warmup: continue
        if (i - last_fit) >= refit_every or beta is None:
            Xs_list, ys_list = [], []
            for s in stocks:
                f = feats_all[s]; r = rets_all[s]
                lim = min(i, len(r) - 1)  # pairs (f[k], r[k+1]) for k < lim
                if lim <= 0: continue
                Xs_list.append(f[:lim]); ys_list.append(r[1:lim+1])
            X = np.vstack(Xs_list); y = np.concatenate(ys_list)
            mu = X.mean(0); sd = X.std(0) + EPS
            Xn = (X - mu) / sd
            Xn = np.hstack([np.ones((len(Xn), 1)), Xn])
            I = np.eye(Xn.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xn.T @ Xn + lam * I, Xn.T @ y)
            last_fit = i
        for s in stocks:
            f = feats_all[s]
            if i >= len(f): continue
            xf = np.concatenate([[1.0], (f[i] - mu) / sd])
            y = shrink * float(xf @ beta)
            preds[s][i] = np.clip(y, -clip, clip)
    return preds

def pooled_lgbm_predict(stocks, refit_every=40, warmup=100, shrink=0.10, clip=0.03):
    import lightgbm as lgb
    feats_all = {}; rets_all = {}; n_max = 0
    for s in stocks:
        P, V = load(s)
        f, r, _ = build_features(P, V)
        feats_all[s] = f; rets_all[s] = r
        n_max = max(n_max, len(r))
    preds = {s: np.zeros(len(rets_all[s])) for s in stocks}
    last_fit = -10**9; model = None
    params = dict(objective='regression_l1', learning_rate=0.02,
                  num_leaves=15, max_depth=4, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  verbose=-1)
    for i in range(n_max):
        if i < warmup: continue
        if (i - last_fit) >= refit_every or model is None:
            Xs_list, ys_list = [], []
            for s in stocks:
                f = feats_all[s]; r = rets_all[s]
                lim = min(i, len(r) - 1)
                if lim <= 0: continue
                Xs_list.append(f[:lim]); ys_list.append(r[1:lim+1])
            X = np.vstack(Xs_list); y = np.concatenate(ys_list)
            ds = lgb.Dataset(X, y)
            model = lgb.train(params, ds, num_boost_round=300)
            last_fit = i
        for s in stocks:
            f = feats_all[s]
            if i >= len(f): continue
            y = shrink * float(model.predict(f[i:i+1])[0])
            preds[s][i] = np.clip(y, -clip, clip)
    return preds

# ---------- per-stock alternates ----------
def predict_M2_blended(P, V):
    p = predict_baseline_v2(P, V)
    return 0.5 * p

def predict_M5_tighter(P, V):
    feats, r, _ = build_features(P, V); n = len(P)
    pred = np.zeros(n)
    SHRINK, CLIP, LAM, REFIT, WARM = 0.05, 0.015, 5.0, 5, 60
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            Xtr = feats[:i]; ytr = r[1:i+1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0, 0] = 0
            beta = np.linalg.solve(Xs.T @ Xs + LAM * I, Xs.T @ ytr); last = i
        xf = np.concatenate([[1.0], (feats[i] - mu) / sd])
        pred[i] = np.clip(SHRINK * float(xf @ beta), -CLIP, CLIP)
    return pred

def predict_M3_signconservative(P, V, k=0.3):
    p = predict_baseline_v2(P, V)
    _, _, vol20 = build_features(P, V)
    out = np.zeros_like(p)
    for i in range(len(p)):
        cap = k * vol20[i]
        out[i] = np.sign(p[i]) * min(abs(p[i]), cap)
    return out

# ---------- run ----------
def main():
    stocks = [f's{k}' for k in range(1, 31)
              if (base_dir / 'sample_data' / f's{k}.npy').exists()]
    print(f"stocks: {len(stocks)}")

    print("Running pooled ridge...")
    pooled_preds = pooled_ridge_predict(stocks)
    print("Running pooled LightGBM (this is slow)...")
    pooled_lgbm = pooled_lgbm_predict(stocks)

    rows = []
    for s in stocks:
        P, V = load(s); t = target(P)
        row = {'stk': s, 'n': len(P)}
        # baseline
        ab, rl = evaluate(list(predict_baseline_v2(P, V)), t)
        row['abs_M0_base'] = ab; row['rel_M0_base'] = rl
        # M1 pooled ridge
        ab, rl = evaluate(list(pooled_preds[s]), t)
        row['abs_M1_poolridge'] = ab; row['rel_M1_poolridge'] = rl
        # M2 blended
        ab, rl = evaluate(list(predict_M2_blended(P, V)), t)
        row['abs_M2_blend'] = ab; row['rel_M2_blend'] = rl
        # M3 sign-conservative
        ab, rl = evaluate(list(predict_M3_signconservative(P, V)), t)
        row['abs_M3_signcons'] = ab; row['rel_M3_signcons'] = rl
        # M5 tighter clip + shrink
        ab, rl = evaluate(list(predict_M5_tighter(P, V)), t)
        row['abs_M5_tight'] = ab; row['rel_M5_tight'] = rl
        # M8 pooled LightGBM
        ab, rl = evaluate(list(pooled_lgbm[s]), t)
        row['abs_M8_poollgbm'] = ab; row['rel_M8_poollgbm'] = rl
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(base_dir / 'improve_per_stock.csv', index=False)

    summary = []
    for label in ['M0_base', 'M1_poolridge', 'M2_blend', 'M3_signcons',
                  'M5_tight', 'M8_poollgbm']:
        summary.append({
            'method': label,
            'median_abs': df[f'abs_{label}'].median(),
            'mean_abs':   df[f'abs_{label}'].mean(),
            'median_rel': df[f'rel_{label}'].median(),
            'mean_rel':   df[f'rel_{label}'].mean(),
            'pct_rel_pos': (df[f'rel_{label}'] > 0).mean() * 100,
        })
    s = pd.DataFrame(summary).round(4)
    s.to_csv(base_dir / 'improve_summary.csv', index=False)
    base_rel = s.loc[s.method == 'M0_base', 'median_rel'].iloc[0]
    base_abs = s.loc[s.method == 'M0_base', 'median_abs'].iloc[0]
    s['drel'] = (s['median_rel'] - base_rel).round(4)
    s['dabs'] = (s['median_abs'] - base_abs).round(4)
    print("\n=== Improvement candidates (median across 30 sample stocks) ===")
    print(s.to_string(index=False))

if __name__ == '__main__':
    main()
