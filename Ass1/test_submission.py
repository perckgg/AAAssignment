"""Smoke-test the submission notebook by executing the prediction code in isolation."""
import json, numpy as np, pandas as pd, types
from pathlib import Path

nb = json.loads(Path("Submission.ipynb").read_text(encoding="utf-8"))

# Concatenate all code cells (skip cell 7 which calls evaluate(...,True) -> shows plot)
src = []
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "evaluate(p, t, True)" in s:   # skip the single-stock plot cell
        continue
    if s.startswith("# 8."):           # also skip batch (we'll do our own batch)
        continue
    src.append(s)
code = "\n".join(src)

ns = {"__name__": "__main__"}
exec(code, ns)

# Now batch-evaluate all 30
prediction = ns["prediction"]
target_fn  = ns["target"]
evaluate   = ns["evaluate"]
base_dir   = Path.cwd()

rows = []
for k in range(1, 31):
    fp = base_dir / "sample_data" / f"s{k}.npy"
    if not fp.exists(): continue
    A = np.load(fp, allow_pickle=True)
    P = A[:, 7].astype(float); V = A[:, 6].astype(float)
    p = prediction(P, V, 5)
    t = target_fn(P, V)
    ab, rl = evaluate(p, t, False)
    rows.append({"stk": f"s{k}", "n": len(P), "abs": ab, "rel": rl})
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
print(f"\nmedian abs = {df['abs'].median():.4f}    median rel = {df['rel'].median():+.4f}")
print(f"mean   abs = {df['abs'].mean():.4f}    mean   rel = {df['rel'].mean():+.4f}")
print(f"% rel > 0  = {(df['rel'] > 0).mean()*100:.1f}%")
