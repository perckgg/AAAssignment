"""Verify v2 algorithm on REAL long-history VN stock data scraped via vnstock.

Hypothesis: v2's modest `rel` is bottlenecked by the small (~242-day) sample.
If true, running the same predictor on 5+ years of real data should produce
substantially larger `rel` and unlock improvements (rolling window, LightGBM)
that were ineffective on the short sample.
"""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys, contextlib, io, threading
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# silence vnstock banner
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from vnstock.api.quote import Quote

sys.path.insert(0, str(Path(__file__).parent))
from experiments import (
    predict_baseline_v2, predict_I1_rolling, predict_I3_lgbm, predict_I4_clipy,
    target, evaluate,
)

TICKERS = ["FPT", "VNM", "HPG", "MWG", "VCB",
           "TCB", "VIC", "MSN", "ACB", "GAS",
           "VHM", "SSI", "VRE", "POW", "PNJ"]
START = "2014-01-01"
END   = "2024-12-31"
CACHE = Path(__file__).parent / "vn_cache"
CACHE.mkdir(exist_ok=True)

def fetch(tk):
    fp = CACHE / f"{tk}.csv"
    if fp.exists():
        return pd.read_csv(fp, parse_dates=["time"])

    def _do(src, out):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                q = Quote(symbol=tk, source=src)
                df = q.history(start=START, end=END, interval="1D")
            out["df"] = df
        except Exception as e:
            out["err"] = e

    last_err = None
    for src in ("VCI", "kbs", "msn"):
        out = {}
        th = threading.Thread(target=_do, args=(src, out), daemon=True)
        th.start(); th.join(timeout=30)
        if th.is_alive():
            last_err = TimeoutError(f"{src} >30s")
            continue
        if "err" in out:
            last_err = out["err"]
            continue
        df = out.get("df")
        if df is not None and len(df) > 100:
            df.to_csv(fp, index=False)
            print(f"  fetched {tk} from {src} ({len(df)} rows)")
            return df
    print(f"  ALL sources failed for {tk}: {last_err}")
    return None

def run_window(P, V, last_n, fn):
    if last_n is not None and last_n < len(P):
        P, V = P[-last_n:], V[-last_n:]
    p = fn(P, V)
    t = target(P)
    return evaluate(p, t)

def main():
    rows = []
    for tk in TICKERS:
        try:
            df = fetch(tk)
        except Exception as e:
            print(f"{tk}: fetch failed — {e}")
            continue
        if df is None or len(df) < 300:
            print(f"{tk}: insufficient ({0 if df is None else len(df)})")
            continue
        # vnstock has no adjusted close; `close` is split-adjusted by VCI source
        P = df["close"].to_numpy(dtype=float)
        V = df["volume"].to_numpy(dtype=float)
        n = len(P)
        row = {"ticker": tk, "n_total": n}

        for sz in (250, 500, 1000, 2000, None):
            label = "full" if sz is None else str(sz)
            ab, rl = run_window(P, V, sz, predict_baseline_v2)
            row[f"abs_v2_{label}"] = ab
            row[f"rel_v2_{label}"] = rl

        # on full history, also test the improvements
        ab, rl = run_window(P, V, None, lambda P, V: predict_I1_rolling(P, V, win=252))
        row["abs_roll252"] = ab; row["rel_roll252"] = rl
        ab, rl = run_window(P, V, None, lambda P, V: predict_I1_rolling(P, V, win=504))
        row["abs_roll504"] = ab; row["rel_roll504"] = rl
        ab, rl = run_window(P, V, None, predict_I4_clipy)
        row["abs_clipy"] = ab; row["rel_clipy"] = rl
        try:
            ab, rl = run_window(P, V, None, predict_I3_lgbm)
            row["abs_lgbm"] = ab; row["rel_lgbm"] = rl
        except Exception as e:
            row["abs_lgbm"] = np.nan; row["rel_lgbm"] = np.nan
            print(f"{tk}: lgbm failed — {e}")

        rows.append(row)
        print(f"{tk:5s} n={n:5d}  "
              f"rel(250)={row['rel_v2_250']:+.4f}  "
              f"rel(500)={row['rel_v2_500']:+.4f}  "
              f"rel(1000)={row['rel_v2_1000']:+.4f}  "
              f"rel(2000)={row['rel_v2_2000']:+.4f}  "
              f"rel(full)={row['rel_v2_full']:+.4f}  "
              f"roll252={row['rel_roll252']:+.4f}  "
              f"lgbm={row['rel_lgbm']:+.4f}")

    df_r = pd.DataFrame(rows)
    df_r.to_csv(Path(__file__).parent / "vn_results_per_ticker.csv", index=False)

    summary = []
    cols = [
        ("v2_250",   "abs_v2_250",   "rel_v2_250"),
        ("v2_500",   "abs_v2_500",   "rel_v2_500"),
        ("v2_1000",  "abs_v2_1000",  "rel_v2_1000"),
        ("v2_2000",  "abs_v2_2000",  "rel_v2_2000"),
        ("v2_full",  "abs_v2_full",  "rel_v2_full"),
        ("roll252",  "abs_roll252",  "rel_roll252"),
        ("roll504",  "abs_roll504",  "rel_roll504"),
        ("clipy",    "abs_clipy",    "rel_clipy"),
        ("lgbm",     "abs_lgbm",     "rel_lgbm"),
    ]
    for label, ac, rc in cols:
        summary.append({
            "method":      label,
            "median_abs":  df_r[ac].median(),
            "median_rel":  df_r[rc].median(),
            "mean_rel":    df_r[rc].mean(),
            "pct_rel_pos": (df_r[rc] > 0).mean() * 100,
        })
    s = pd.DataFrame(summary).round(4)
    s.to_csv(Path(__file__).parent / "vn_summary.csv", index=False)

    print("\n=== Summary across VN tickers (full real history vs. synthetic 242-day sample) ===")
    print(s.to_string(index=False))

if __name__ == "__main__":
    main()
