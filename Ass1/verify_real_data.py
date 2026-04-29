"""Verify v2 algorithm on REAL long-history data scraped from Yahoo Finance.

Goal: test whether v2's modest `rel` is caused by the small (242-day) sample,
or by an inherent limit of the model. Strategy:
  1. Download ~10 years of daily bars for liquid US tickers via yfinance.
  2. Run the v2 predictor (same code as the notebook) on each ticker.
  3. Also compute scaled-down windows (last 250 / 500 / 1000 / full) to see
     whether `rel` improves with more history.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

# reuse predictor code
import sys
sys.path.insert(0, str(Path(__file__).parent))
from experiments import predict_baseline_v2, predict_I1_rolling, target, evaluate

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
           "NVDA", "TSLA", "JPM", "V", "WMT",
           "JNJ", "XOM", "PG", "MA", "HD"]
START = "2014-01-01"
END   = "2024-12-31"
CACHE = Path(__file__).parent / "yf_cache"
CACHE.mkdir(exist_ok=True)

def fetch(tk):
    fp = CACHE / f"{tk}.csv"
    if fp.exists():
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    df = yf.download(tk, start=START, end=END, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(fp)
    return df

def windowed_eval(P, V, last_n):
    if last_n is not None and last_n < len(P):
        P, V = P[-last_n:], V[-last_n:]
    p = predict_baseline_v2(P, V)
    t = target(P)
    return evaluate(p, t)

def windowed_eval_roll(P, V, last_n, win=252):
    if last_n is not None and last_n < len(P):
        P, V = P[-last_n:], V[-last_n:]
    p = predict_I1_rolling(P, V, win=win)
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
        if df.empty or "Adj Close" not in df.columns:
            print(f"{tk}: no Adj Close")
            continue
        P = df["Adj Close"].to_numpy(dtype=float)
        V = df["Volume"].to_numpy(dtype=float)
        n = len(P)
        if n < 300:
            print(f"{tk}: too short ({n})")
            continue

        row = {"ticker": tk, "n_total": n}
        for sz in (250, 500, 1000, None):
            label = "full" if sz is None else str(sz)
            ab, rl = windowed_eval(P, V, sz)
            row[f"abs_{label}"] = ab
            row[f"rel_{label}"] = rl
        # rolling on full
        ab, rl = windowed_eval_roll(P, V, None, win=252)
        row["abs_full_roll252"] = ab
        row["rel_full_roll252"] = rl
        rows.append(row)
        print(f"{tk:6s} n={n:5d}  rel(250)={row['rel_250']:+.4f}  "
              f"rel(500)={row['rel_500']:+.4f}  rel(1000)={row['rel_1000']:+.4f}  "
              f"rel(full)={row['rel_full']:+.4f}  rel(full,roll252)={row['rel_full_roll252']:+.4f}")

    df_r = pd.DataFrame(rows)
    df_r.to_csv(Path(__file__).parent / "yf_results_per_ticker.csv", index=False)

    summary = []
    for label in ("250", "500", "1000", "full", "full_roll252"):
        summary.append({
            "window": label,
            "median_abs": df_r[f"abs_{label}"].median(),
            "median_rel": df_r[f"rel_{label}"].median(),
            "mean_rel":   df_r[f"rel_{label}"].mean(),
            "pct_rel_pos": (df_r[f"rel_{label}"] > 0).mean() * 100,
        })
    s = pd.DataFrame(summary).round(4)
    s.to_csv(Path(__file__).parent / "yf_summary.csv", index=False)
    print("\n=== Summary across tickers ===")
    print(s.to_string(index=False))

if __name__ == "__main__":
    main()
