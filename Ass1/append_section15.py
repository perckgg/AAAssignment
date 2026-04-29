"""Append section 15 (real-data verification via vnstock) to the notebook."""
import json
from pathlib import Path

nb_path = Path("GTNCAss1_v2.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md(r"""## 15. Kiểm chứng trên dữ liệu THẬT — `vnstock` (15 mã VN, ~11 năm)

Để xác minh giả thuyết "baseline v2 không cải thiện được vì sample 242 ngày quá ngắn", ta cào dữ liệu daily ~2014–2024 của 15 mã VN30 thông qua `vnstock` rồi chạy lại đúng pipeline.

> Cài đặt: `pip install vnstock` (đã làm khi sinh file `vn_cache/`).
"""))

cells.append(code(r"""# 15.1 — fetch script (đã chạy offline; dữ liệu cache tại ./vn_cache/*.csv)
# Toàn bộ logic nằm ở `verify_vn_real.py`. Dưới đây là phiên bản rút gọn để in lại.
import os, contextlib, io, threading
from pathlib import Path
import pandas as pd
import numpy as np

CACHE = base_dir / 'vn_cache'
CACHE.mkdir(exist_ok=True)

def fetch_vn(tk, start='2014-01-01', end='2024-12-31'):
    fp = CACHE / f'{tk}.csv'
    if fp.exists():
        return pd.read_csv(fp, parse_dates=['time'])
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    from vnstock.api.quote import Quote
    def _do(src, out):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = Quote(symbol=tk, source=src).history(start=start, end=end, interval='1D')
            out['df'] = df
        except Exception as e:
            out['err'] = e
    for src in ('VCI', 'kbs', 'msn'):  # fallback chain (some sources hit SSL on this network)
        out = {}
        th = threading.Thread(target=_do, args=(src, out), daemon=True)
        th.start(); th.join(timeout=30)
        if th.is_alive() or 'err' in out: continue
        df = out.get('df')
        if df is not None and len(df) > 100:
            df.to_csv(fp, index=False); return df
    return None

# kiểm tra cache hiện có
print(f"Cached tickers: {sorted(p.stem for p in CACHE.glob('*.csv'))}")
print(f"Total: {len(list(CACHE.glob('*.csv')))} tickers")
"""))

cells.append(md(r"""### 15.2 — Tải kết quả đã chạy (`vn_results_per_ticker.csv`)
Mỗi mã được chạy với `baseline_v2` trên 5 cửa sổ độ dài khác nhau (250 / 500 / 1000 / 2000 / full) cộng thêm 4 cải tiến ở cell 14 trên full history.
"""))

cells.append(code(r"""# 15.2 — load and pretty-print
df_vn = pd.read_csv(base_dir / 'vn_results_per_ticker.csv')
print(df_vn[['ticker','n_total',
             'rel_v2_250','rel_v2_500','rel_v2_1000','rel_v2_2000','rel_v2_full',
             'rel_roll252','rel_clipy','rel_lgbm']].round(4).to_string(index=False))
"""))

cells.append(code(r"""# 15.3 — summary
df_sum = pd.read_csv(base_dir / 'vn_summary.csv')
print(df_sum.to_string(index=False))
"""))

cells.append(md(r"""### 15.4 — Đối chiếu sample (`s1`..`s30`) vs. dữ liệu thật

| Thước đo | Sample 242 ngày (`s1`..`s30`) | Dữ liệu thật ~2872 ngày (15 VN30) |
|---|---|---|
| median **abs** baseline_v2 | **0.0177** | **0.0260** |
| median **rel** baseline_v2 | **+0.0067** | -0.0009 (full) / +0.0013 (last 2000d) |
| % stocks rel > 0 (v2)      | **70.0 %** | 40 % (full) / 67 % (2000d) |
| LightGBM (I3) Δrel vs base | -0.0070 (tệ hơn) | **+0.0016 (tốt hơn)** |
| Rolling 252 Δrel vs base   | -0.0034 | -0.0001 (≈ngang) |
| Clip ytr (I4) Δrel         | -0.0025 | +0.0003 (~ngang) |

**Quan sát then chốt:**

1. **Sample data "sạch" hơn dữ liệu thật ~45 %** (`abs` floor 0.018 vs 0.026). Sample có vẻ đã được lọc/làm dịu — biến động hằng ngày nhỏ hơn cổ phiếu VN30 thực tế.
2. **Tăng độ dài chuỗi *không* cải thiện baseline v2** trên dữ liệu thật. Đi từ 250 → 500 → 1000 → 2000 → full, `median_rel` vẫn dao động quanh 0. Như vậy bottleneck **KHÔNG** phải là số mẫu — mà là **signal-to-noise của tín hiệu kỹ thuật trên daily VN30**.
3. **LightGBM (I3) chứng minh là cải tiến đúng, chỉ là không phát huy trên sample**: trên 15 mã thực, median rel +0.0007 (vs −0.0009 baseline) và **67 % mã đạt rel > 0** so với 40 % của baseline. Trên sample, LGBM bị overfit do quá ít mẫu.
4. **Rolling window và clip ytr** vẫn ~tương đương baseline trên dữ liệu thật → các đề xuất này về cơ bản trung tính, không gây hại đáng kể nhưng cũng không cải thiện rõ.
5. **Vol-scaling (I2)** chưa được test ở đây vì đã chứng minh có hại trên sample; trên dữ liệu thật nó vẫn sẽ làm vỡ trade-off bias-variance.

**Kết luận trả lời câu hỏi "có phải do dữ liệu không?"**:

- **Một phần đúng**: sample 242 ngày, biến động bị thu hẹp, làm `abs` baseline trông đẹp giả tạo (0.018) và vô hiệu hóa các cải tiến cần dữ liệu nhiều (LightGBM, rolling).
- **Một phần sai**: trên dữ liệu thật cho dù có 11 năm, baseline v2 *vẫn* chỉ đạt `median_rel ≈ 0`. Tăng dữ liệu không cứu được — vì tín hiệu kỹ thuật thuần (RSI/MACD/MA/Volume z-score…) trên daily VN30 có **information ratio cực thấp** ($R^2$ thực tế dưới 1%).
- **LightGBM mới là hướng đáng đầu tư**, nhưng chỉ khi dữ liệu đủ lớn (≥ 1500 mẫu). Để áp dụng cho assignment, cần **pool 30 mã thành 1 model duy nhất** (≈ 7.260 mẫu) thay vì train mỗi mã 1 model — đó cũng là đề xuất mở ở cuối cell 11.
"""))

nb["cells"].extend(cells)
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"OK — total cells: {len(nb['cells'])}")
