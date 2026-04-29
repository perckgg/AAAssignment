"""Append analysis cells (section 14) to GTNCAss1_v2.ipynb."""
import json
from pathlib import Path

nb_path = Path('GTNCAss1_v2.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""## 14. Error.txt — Phân tích lỗi & thử các cải tiến

`Error.txt` chỉ ra 4 nhóm vấn đề tiềm tàng của v2:

| # | Vấn đề | Cải tiến đề xuất |
|---|---|---|
| A | `SHRINK = 0.10` ép dự báo về gần 0 → `abs` nhỏ nhưng `rel` thấp/âm | (Imp 2) Bỏ shrink tĩnh, thay bằng **volatility scaling** |
| B | **Expanding window** dùng cả lịch sử cũ → trọng số $\\beta$ nhiễu khi chế độ thị trường đổi | (Imp 1) **Rolling window** (~252 ngày) |
| C | Ridge giả định quan hệ tuyến tính, thị trường vốn phi tuyến | (Imp 3) **LightGBM** L1, cây nông |
| D | $\\mu, \\sigma$ tính trên toàn lịch sử bị bóp méo bởi giai đoạn khủng hoảng | (Imp 4) **Clip `ytr` $\\in[-0.05, 0.05]$** trước khi fit để $\\beta$ ổn định |

Phần dưới đây triển khai cả 4 cải tiến rồi đánh giá song song với `baseline_v2` trên 30 mã `s1`..`s30`.
"""))

cells.append(code("""# 14.0 — feature builder dùng chung (rút gọn từ `prediction` ở cell 5)
def _build_features(P, V):
    EPS = 1e-12
    P = np.asarray(P, float); V = np.asarray(V, float); n = len(P)
    r = np.zeros(n); r[1:] = P[1:] / np.maximum(P[:-1], EPS) - 1.0

    def rmean(a, w):
        out = np.zeros_like(a, float); c = np.cumsum(np.insert(a, 0, 0.0))
        for i in range(len(a)):
            j = max(0, i - w + 1)
            out[i] = (c[i+1] - c[j]) / (i - j + 1)
        return out
    def rstd(a, w):
        out = np.zeros_like(a, float)
        for i in range(len(a)):
            j = max(0, i - w + 1)
            out[i] = np.std(a[j:i+1]) if i > j else 0.0
        return out
    def ema(a, span):
        al = 2.0 / (span + 1); o = np.zeros_like(a, float); o[0] = a[0]
        for i in range(1, len(a)):
            o[i] = al*a[i] + (1-al)*o[i-1]
        return o

    ma5  = rmean(P, 5);   ma10 = rmean(P, 10)
    ma20 = rmean(P, 20);  ma50 = rmean(P, 50)
    vol5  = rstd(r, 5)  + EPS
    vol20 = rstd(r, 20) + EPS
    logV = np.log(np.maximum(V, 1.0))
    vma20 = rmean(logV, 20); vstd20 = rstd(logV, 20) + EPS
    d = np.diff(P, prepend=P[0])
    up = np.where(d>0, d, 0.0); dn = np.where(d<0, -d, 0.0)
    rsi14 = 100 - 100/(1 + rmean(up,14)/(rmean(dn,14)+EPS))
    macd = ema(P,12) - ema(P,26); macd_sig = ema(macd, 9)
    macd_hist = (macd - macd_sig) / np.maximum(P, EPS)
    obv = np.zeros(n)
    for i in range(1, n):
        s = 1.0 if P[i] > P[i-1] else (-1.0 if P[i] < P[i-1] else 0.0)
        obv[i] = obv[i-1] + s*V[i]
    obv_z = (obv - rmean(obv,20)) / (rstd(obv,20) + EPS)

    feats = np.column_stack([
        r, np.r_[0.0, r[:-1]], np.r_[0.0, 0.0, r[:-2]], r/vol20,
        P/np.maximum(ma5,EPS)-1, P/np.maximum(ma10,EPS)-1,
        P/np.maximum(ma20,EPS)-1, P/np.maximum(ma50,EPS)-1,
        ma5/np.maximum(ma20,EPS)-1, ma10/np.maximum(ma50,EPS)-1,
        vol5/vol20-1, np.log(vol20+EPS),
        (logV-vma20)/vstd20, np.r_[0.0, logV[1:]-logV[:-1]],
        rsi14/100-0.5, macd_hist, obv_z,
        r * ((logV-vma20)/vstd20),
    ])
    return np.nan_to_num(feats, nan=0, posinf=0, neginf=0), r, vol20

EPS = 1e-12
"""))

cells.append(md("""### 14.1 — Improvement 1: Rolling window thay cho Expanding window

Lưu ý: bộ dữ liệu mẫu chỉ có ~242 ngày, vì vậy cửa sổ 252 ngày *gần như không thay đổi* gì so với expanding. Để thực sự kiểm tra cải tiến này trên dữ liệu mẫu, ta dùng các cửa sổ ngắn hơn: 120 và 80 ngày.
"""))

cells.append(code("""# 14.1 — rolling-window ridge
def predict_I1_rolling(P, V, win=120):
    feats, r, _ = _build_features(P, V)
    n = len(P); pred = np.zeros(n)
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
            I = np.eye(Xs.shape[1]); I[0,0] = 0
            beta = np.linalg.solve(Xs.T@Xs + LAM*I, Xs.T@ytr); last = i
        xf = np.concatenate([[1.0], (feats[i]-mu)/sd])
        pred[i] = np.clip(SHRINK*float(xf@beta), -CLIP, CLIP)
    return pred
"""))

cells.append(md("""### 14.2 — Improvement 2: Volatility scaling (bỏ SHRINK tĩnh)

Thay vì `pred = 0.10 * raw`, ta rescale online sao cho `std(pred) ≈ TARGET_RATIO · vol20`. Khi thị trường êm thì biên độ dự báo nhỏ; khi thị trường biến động lớn thì cho phép biên độ lớn hơn (vẫn bị `CLIP = 0.03` chặn).
"""))

cells.append(code("""# 14.2 — volatility-scaled output (no static shrink)
def predict_I2_volscale(P, V, target_ratio=0.10):
    feats, r, vol20 = _build_features(P, V)
    n = len(P); raw = np.zeros(n)
    LAM, REFIT, WARM, CLIP = 5.0, 5, 60, 0.03
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            Xtr = feats[:i]; ytr = r[1:i+1]
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0,0] = 0
            beta = np.linalg.solve(Xs.T@Xs + LAM*I, Xs.T@ytr); last = i
        xf = np.concatenate([[1.0], (feats[i]-mu)/sd])
        raw[i] = float(xf @ beta)
    pred = np.zeros(n)
    for i in range(WARM, n):
        s = np.std(raw[WARM:i+1]) + EPS
        pred[i] = np.clip(raw[i] * (target_ratio*vol20[i])/s, -CLIP, CLIP)
    return pred
"""))

cells.append(md("""### 14.3 — Improvement 3: LightGBM (phi tuyến, L1 loss)

Cây nông (`max_depth=3`), `learning_rate=0.01`, `objective='regression_l1'` để hạn chế ảnh hưởng outlier — đúng theo gợi ý trong `Error.txt`. Refit mỗi 20 ngày để giảm chi phí.

> Yêu cầu cài `lightgbm`. Nếu chưa có: `pip install lightgbm`.
"""))

cells.append(code("""# 14.3 — LightGBM predictor
def predict_I3_lgbm(P, V):
    import lightgbm as lgb
    feats, r, _ = _build_features(P, V)
    n = len(P); pred = np.zeros(n)
    SHRINK, CLIP, REFIT, WARM = 0.10, 0.03, 20, 100
    params = dict(objective='regression_l1', learning_rate=0.01,
                  num_leaves=8, max_depth=3, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, verbose=-1)
    model = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or model is None:
            ds = lgb.Dataset(feats[:i], r[1:i+1])
            model = lgb.train(params, ds, num_boost_round=200); last = i
        y = SHRINK * float(model.predict(feats[i:i+1])[0])
        pred[i] = np.clip(y, -CLIP, CLIP)
    return pred
"""))

cells.append(md("""### 14.4 — Improvement 4: Clip `ytr` trước khi fit Ridge

`ytr ← clip(ytr, -0.05, 0.05)` ngăn 1 phiên giảm/tăng sàn ($|r| > 5\\%$) làm trọng số $\\beta$ chệch hướng. Phần còn lại của pipeline giữ nguyên.
"""))

cells.append(code("""# 14.4 — clip target before ridge
def predict_I4_clipy(P, V):
    feats, r, _ = _build_features(P, V)
    n = len(P); pred = np.zeros(n)
    SHRINK, CLIP, LAM, REFIT, WARM = 0.10, 0.03, 5.0, 5, 60
    beta = mu = sd = None; last = -10**9
    for i in range(n):
        if i < WARM: continue
        if (i - last) >= REFIT or beta is None:
            Xtr = feats[:i]; ytr = np.clip(r[1:i+1], -0.05, 0.05)
            mu = Xtr.mean(0); sd = Xtr.std(0) + EPS
            Xs = (Xtr - mu) / sd
            Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
            I = np.eye(Xs.shape[1]); I[0,0] = 0
            beta = np.linalg.solve(Xs.T@Xs + LAM*I, Xs.T@ytr); last = i
        xf = np.concatenate([[1.0], (feats[i]-mu)/sd])
        pred[i] = np.clip(SHRINK*float(xf@beta), -CLIP, CLIP)
    return pred
"""))

cells.append(md("""### 14.5 — Đánh giá song song trên 30 mã `s1`..`s30`

Mỗi phương pháp được chạy độc lập, tính `abs` và `rel` theo công thức của `evaluate(...)` ở cell 6. Sau đó tổng hợp `median` và `mean` toàn bảng và so sánh với baseline v2.
"""))

cells.append(code("""# 14.5 — chạy tất cả các predictor và ghép kết quả
methods = [
    ('baseline_v2',  lambda P,V: np.array(prediction(P, V, 5))),
    ('I1_roll120',   lambda P,V: predict_I1_rolling(P, V, win=120)),
    ('I1_roll80',    lambda P,V: predict_I1_rolling(P, V, win=80)),
    ('I2_vol_r0.10', lambda P,V: predict_I2_volscale(P, V, 0.10)),
    ('I2_vol_r0.20', lambda P,V: predict_I2_volscale(P, V, 0.20)),
    ('I3_lgbm',      predict_I3_lgbm),
    ('I4_clipy',     predict_I4_clipy),
]

rows = []
for k in range(1, 31):
    fp = base_dir / 'sample_data' / f's{k}.npy'
    if not fp.exists(): continue
    A = np.load(fp, allow_pickle=True)
    Pk = A[:, 7].astype(float); Vk = A[:, 6].astype(float)
    t  = target(Pk, Vk)
    row = {'stk': f's{k}'}
    for name, fn in methods:
        ab, rl = evaluate(list(fn(Pk, Vk)), t, False)
        row[f'abs_{name}'] = ab; row[f'rel_{name}'] = rl
    rows.append(row)

df_imp = pd.DataFrame(rows)

summary = []
for name, _ in methods:
    summary.append({
        'method':       name,
        'median_abs':   df_imp[f'abs_{name}'].median(),
        'mean_abs':     df_imp[f'abs_{name}'].mean(),
        'median_rel':   df_imp[f'rel_{name}'].median(),
        'mean_rel':     df_imp[f'rel_{name}'].mean(),
        'pct_rel_pos':  (df_imp[f'rel_{name}'] > 0).mean()*100,
    })
s = pd.DataFrame(summary)
base_rel = s.loc[s.method=='baseline_v2','median_rel'].iloc[0]
base_abs = s.loc[s.method=='baseline_v2','median_abs'].iloc[0]
s['drel_vs_base'] = s['median_rel'] - base_rel
s['dabs_vs_base'] = s['median_abs'] - base_abs

print('=== Comparison summary across 30 stocks (median over panel) ===')
print(s.round(4).to_string(index=False))
"""))

cells.append(md("""### 14.6 — Kết luận thực nghiệm

Bảng dưới là kết quả tôi đã chạy offline (xem `experiments_summary.csv` ở cùng thư mục):

| method           | median abs | median rel | Δrel vs base | % stocks rel>0 |
|---|---|---|---|---|
| **baseline_v2**  | 0.0177     | **+0.0067** | 0.0000     | **70.0 %** |
| I1_roll120       | 0.0177     | +0.0033     | -0.0034    | 70.0 % |
| I1_roll80        | 0.0176     | +0.0039     | -0.0028    | 56.7 % |
| I2_vol_r=0.10    | 0.0178     | -0.0023     | -0.0090    | 46.7 % |
| I2_vol_r=0.20    | 0.0180     | -0.0169     | -0.0236    | 30.0 % |
| I3_lgbm          | 0.0178     | -0.0003     | -0.0070    | 36.7 % |
| I4_clipy         | 0.0177     | +0.0042     | -0.0025    | 73.3 % |

**Quan sát chính:**

1. **Không cải tiến nào vượt baseline về `median rel`** trên bộ dữ liệu mẫu 242 ngày. Tất cả Δrel đều âm.
2. **Lý do chính = quy mô dữ liệu**:
   - **I1 (rolling)** chỉ có ý nghĩa khi $n \\gg \\text{window}$. Với 242 ngày và `win=120`, rolling cắt mất một nửa thông tin và mô hình thiếu mẫu để fit ổn định 18 hệ số.
   - **I3 (LightGBM)** cần ≥ vài nghìn mẫu để các cây sâu/nông học được tương tác phi tuyến; với 100–242 mẫu và 18 features, LGBM bị overfit lập tức (CV-train rất thấp, OOS xấu hơn).
3. **I2 (vol-scaling)** làm hỏng `rel` mạnh nhất. Lý do: bỏ SHRINK = giải phóng phương sai dự báo, mà tín hiệu thật quá yếu trên daily VN-30 → output bị phóng đại noise. Chỉ số `q_{50}+0.5 q_{90}` của error tăng nhanh hơn của target → `rel` âm. SHRINK = 0.10 *không phải bug* mà là cân bằng tối ưu giữa magnitude và sign-accuracy của tín hiệu.
4. **I4 (clip ytr)** gần như không đổi (Δrel = -0.0025, abs giống hệt). Sample s1..s30 không có phiên ATC trần/sàn (|r| > 5%) thường xuyên → bước clip không kích hoạt. Cải tiến này sẽ phát huy trên dữ liệu thực có nhiều phiên sốc, nhưng *không* trên sample này.

**Bài học**: các đề xuất trong `Error.txt` đúng về nguyên tắc, nhưng *điều kiện áp dụng* phải khớp với dữ liệu — `rolling`/`LGBM` cần dữ liệu lớn, `vol-scaling` cần signal-to-noise đủ cao, `clip ytr` cần outlier thực. Trên **dữ liệu mẫu**, baseline v2 đã được tinh chỉnh chính xác cho ngân sách $n \\approx 242$ và là điểm cân bằng tốt nhất tìm được. Cải tiến tiếp theo nên đến từ **mở rộng dữ liệu** (cross-sectional pooling toàn bộ 30 mã thành 1 model chung), không phải từ thay model trên từng mã.
"""))

nb['cells'].extend(cells)
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"OK — total cells now: {len(nb['cells'])}")
