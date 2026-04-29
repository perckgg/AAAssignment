"""Final tuning: grid over SHRINK + ensembles of pooled & per-stock."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))
from experiments import build_features, predict_baseline_v2, target, evaluate, EPS
from improve_v3 import pooled_ridge_predict, load

base_dir = Path(__file__).parent
stocks = [f's{k}' for k in range(1, 31)
          if (base_dir / 'sample_data' / f's{k}.npy').exists()]

# Baseline cache
base_preds = {}
for s in stocks:
    P, V = load(s)
    base_preds[s] = predict_baseline_v2(P, V)

# Pooled cache (default shrink 0.10)
pool_preds = pooled_ridge_predict(stocks, shrink=0.10)

# Targets cache
tgt = {s: target(load(s)[0]) for s in stocks}

def score(pred_map):
    rels, abss = [], []
    for s in stocks:
        ab, rl = evaluate(list(pred_map[s]), tgt[s])
        rels.append(rl); abss.append(ab)
    return np.median(abss), np.median(rels), np.mean(rels), (np.array(rels) > 0).mean()*100

# ---- 1. Grid over SHRINK on baseline ----
print("Grid over SHRINK applied to baseline:")
print(f"  {'shrink':>6}  {'med_abs':>9}  {'med_rel':>9}  {'mean_rel':>9}  {'pct>0':>6}")
for sh in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.00]:
    pred = {s: base_preds[s] * (sh / 0.10) for s in stocks}  # rescale (baseline already 0.10 * raw)
    # but clip is tied to baseline; for grid just apply scaling factor to the already-shrunk preds
    pred = {s: np.clip(base_preds[s] * (sh / 0.10), -0.03, 0.03) for s in stocks}
    a, mr, ar, pp = score(pred)
    print(f"  {sh:6.2f}  {a:9.5f}  {mr:+9.5f}  {ar:+9.5f}  {pp:6.1f}")

# ---- 2. Ensemble: alpha * baseline + (1-alpha) * pooled ----
print("\nEnsemble baseline + pooled:")
print(f"  {'alpha':>6}  {'med_abs':>9}  {'med_rel':>9}  {'mean_rel':>9}  {'pct>0':>6}")
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    pred = {s: a * base_preds[s] + (1-a) * pool_preds[s] for s in stocks}
    ab, mr, ar, pp = score(pred)
    print(f"  {a:6.2f}  {ab:9.5f}  {mr:+9.5f}  {ar:+9.5f}  {pp:6.1f}")

# ---- 3. Pooled ridge with different shrink ----
print("\nPooled ridge with different shrink:")
print(f"  {'shrink':>6}  {'med_abs':>9}  {'med_rel':>9}  {'mean_rel':>9}  {'pct>0':>6}")
for sh in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    pp_pred = pooled_ridge_predict(stocks, shrink=sh)
    ab, mr, ar, pct = score(pp_pred)
    print(f"  {sh:6.2f}  {ab:9.5f}  {mr:+9.5f}  {ar:+9.5f}  {pct:6.1f}")

# ---- 4. Hard zero baseline reference ----
zero_pred = {s: np.zeros(len(base_preds[s])) for s in stocks}
ab, mr, ar, pp = score(zero_pred)
print(f"\nReference: predict 0 everywhere -> abs={ab:.5f} rel={mr:+.5f} mean_rel={ar:+.5f} pct>0={pp:.1f}")

# ---- 5. Best so far: pooled @ shrink=0.05 + zero blend ----
print("\nBlend pooled(shrink=0.05) with zero:")
pp05 = pooled_ridge_predict(stocks, shrink=0.05)
for a in [0.3, 0.5, 0.7, 1.0]:
    pred = {s: a * pp05[s] for s in stocks}
    ab, mr, ar, pct = score(pred)
    print(f"  scale={a:.2f}  abs={ab:.5f}  rel={mr:+.5f}  mean={ar:+.5f}  pct>0={pct:.1f}")
