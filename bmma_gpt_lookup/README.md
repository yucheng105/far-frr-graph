# bmma_gpt_lookup

把 SBI / Xception / UCF 三個 Visual Deepfake 偵測器的 threshold sweep 資料，轉換成
Anti-Deepfake-Box（ADB）序列式（cascade）偵測系統實際查表用的 **Lookup Table**
（`real_data_vip_settings.csv`）。

理論基礎是論文 *Advancing Biometric Authentication With Dual-Threshold
Multi-Modal Systems and Geometric Programming*（BMMA-GPT）：把生物特徵驗證的
雙門檻序列架構（每個偵測器一個下門檻 `threshold_L`、一個上門檻 `threshold_H`，
`score < threshold_L` 判 Real、`score > threshold_H` 判 Fake、否則換下一個偵測器）
映射到 Deepfake 偵測場景，用幾何規劃（GP）在滿足系統整體 FAR 上限（α）、
FRR 上限（β）的前提下，找出「用哪些偵測器、依什麼順序、搭配什麼雙門檻」
能讓判斷延遲最短。

這個資料夾是獨立的分析輸出，**不會修改 repo 裡任何既有檔案**（`models/`、
`results/`、`input-csvs/` 都只被讀取，不被寫入）。

**本次 GP 約束／反查修正**（最後一關單門檻、兩邊 `inv_far`、差式 cascade 重算）見
[README_GP_FIX.md](README_GP_FIX.md)。

---

## 目錄結構

```
bmma_gpt_lookup/
├── build_lookup_table.py            # 主程式（步驟1~4，見下）
├── validate.py                      # 讀輸出 CSV，印摘要 + 檢查格式健全性
├── README_GP_FIX.md                 # 本次 GP 約束與反查修正說明
├── GP_LOOKUP_TABLE_REVIEW.md        # Code review 與公式說明
├── PROJECT_BRIEFING.md              # 專案背景（給尚未進入公式的人）
├── detector_latency.csv             # 補齊 sweep 檔缺少的 avg_latency_ms
├── threshold_sweep_sbi_faceforensics_v1.csv
│                                     # 用 v1 原始分數重新算的「方向正確」SBI/FF++ sweep
│                                     # （見下方「SBI 資料問題」一節，為何需要這份檔案）
├── requirements.txt                 # pandas, numpy, cvxpy
├── output/
│   ├── real_data_vip_settings.csv   # 最終輸出：每個安全等級選中的排列與門檻
│   └── all_orderings.csv            # 診斷用：15 種排列各自的 GP 求解結果（不篩選）
└── .venv/                           # 本機測試用的獨立虛擬環境（不影響全域 Python）
```

## 怎麼跑

```bash
cd bmma_gpt_lookup
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python build_lookup_table.py   # 產生 output/*.csv，並印出過程訊息
./.venv/bin/python validate.py             # 讀 output/real_data_vip_settings.csv 驗證格式
```

---

## 整體流程（對應任務書的四個步驟）

### 步驟 1：讀取/合併 threshold sweep 資料 → 每個偵測器的 `DetectorProfile`

對每個偵測器：讀入所有資料集的 sweep 檔 → 驗證 far/frr 方向（`verify_far_frr_direction`，
不合預期就直接停止並印警告，不自動猜方向）→ 用線性內插對齊到共同 threshold 格點
（`align_to_common_grid`）→ 跨資料集合併，預設 `combine_mode="max"`（取最差值，
保守策略，`combine_across_datasets`）→ 單調化（`enforce_monotonic`）→ 算出：

- **EER**：far/frr 曲線交點（`compute_eer`）
- **conb**：`max(EER², max(far×frr))`，FAR×FRR 守恆常數（`compute_conb`，對應論文公式16）
- **Tb**：跨資料集 `avg_latency_ms` 取最大值（保守策略，恆定，跟 `combine_mode` 無關）
- 四個查表函式 `far_at/frr_at`（正查）、`inv_far/inv_frr`（反查，線性內插）

### 步驟 2：排列窮舉

`itertools.permutations`，長度 1 到 N，3 個偵測器 → 3+6+6 = **15 種排列**。

### 步驟 3：對每個排列用 `cvxpy` 的 DGP（`gp=True`）求解雙門檻

決策變數 `AB_lb[i]`、`AB_ub[i]`（每階段各一個正變數），透過 conb 守恆關係推導
`RB_lb[i] = conb[i]/AB_lb[i]`、`RB_ub[i] = conb[i]/AB_ub[i]`。

- 目標函數（**論文公式17**，最小化系統總 FRR）：
  `total_frr = Σ_s (Π_{j<s} RB_lb[j]) × RB_ub[s]`
- 約束（**論文公式18**，系統總 FAR ≤ α）：
  `total_far = Σ_s (Π_{j<s} AB_ub[j]) × AB_lb[s] ≤ α`
- 加上 `AB_lb[i] ≤ AB_ub[i]`（門檻順序）與變數邊界 `(1e-6, 1-1e-6)`。

這幾條逐字對過論文原文的公式(5)(6)(9)(11)(16)-(22)，完全一致（見下方
「過程中發現的問題」第 2 點的詳細推導）。

### 步驟 4：依安全等級篩選最佳排列，反查回實際門檻

每個安全等級除了原本的 `alpha`（FAR上限）、`beta`（FRR上限）、
`latency_budget_ms`（延遲預算）之外，**多加了一個 `min_stages`**（最少階段數，
見下方「為什麼加 min_stages」）。篩選順序：排列長度 ≥ min_stages → 各偵測器
Tb 加總 ≤ latency_budget_ms → GP 求解成功 → total_frr ≤ beta → 依總延遲排序取
最短者（打平取 frr 較小者）→ 反查門檻後，前 N−1 關須 `threshold_L < threshold_H`，
最後一關須 `threshold_L ≈ threshold_H`（單一門檻）。詳見 README_GP_FIX.md。
全部沒有可行排列時印警告說明是哪一關篩掉的，不會讓程式中斷。

---

## 資料來源與過程中發現的問題

### 1. SBI 的 threshold sweep 資料方向寫反了

跑第一次的時候，`verify_far_frr_direction()` 在 `models/sbi/threshold_sweep_sbi_v0.csv`
上直接觸發警告並停止：far 隨 threshold **遞減**、frr 隨 threshold **遞增**，跟
Xception/UCF（far 遞增、frr 遞減）方向相反。

用 `input-csvs/sbi/scores_visual_sbi_ffpp_v{0,1}.csv` 的原始分數，照
`src/threshold_sweep.py` 的正確定義（`fake_score >= threshold` 判 Fake）重新算過，
逐個 threshold 切點精準對上「far/frr 兩欄互換」的結果——確認 `models/sbi/` 這兩份
sweep 檔案的 far/frr 欄位在產生時被寫反了。

進一步比對發現 `results/20/sbi/threshold_sweep_sbi_{celebdf,dfdc}.csv` 方向是對的，
且與 `input-csvs/sbi` 的 v0 原始分數逐點吻合，所以 SBI 的 CelebDF/DFDC 改用這兩份。
但 `results/20/sbi/threshold_sweep_sbi_faceforensics.csv` 核對後其實是用 v0
（280 筆平衡樣本）算出來的，不是預期中該用的 v1（700 筆，140 real + 560 fake）。
因此另外在本資料夾用 v1 原始分數重新計算了一份方向正確版本
（`threshold_sweep_sbi_faceforensics_v1.csv`，avg_latency_ms 取 v1 原始檔
`inference_time_ms` 平均 308.7ms）。

**結論：SBI 三個資料集的最終資料來源**

| 資料集 | 檔案 | 備註 |
|---|---|---|
| CelebDF | `results/20/sbi/threshold_sweep_sbi_celebdf.csv` | 方向正確，對過 v0 原始分數 |
| DFDC | `results/20/sbi/threshold_sweep_sbi_dfdc.csv` | 方向正確，對過 v0 原始分數 |
| FaceForensics | `bmma_gpt_lookup/threshold_sweep_sbi_faceforensics_v1.csv` | 本資料夾內重算，用 v1 原始分數 |

Xception/UCF 的 sweep 檔案（`models/xception/`、`models/ucf/`）方向本來就正確，
直接使用；缺少的 `avg_latency_ms` 補自 `README.md` 的「Average Inference Time」表格
（記錄在 `detector_latency.csv`），排除三者的 `*_combined.csv`（已是跨資料集合併結果，
避免和單一資料集檔案一起做 `combine_mode` 造成重複計入）。

### 2. 為什麼串接（cascade）數學上永遠贏不了單一偵測器

第一次拿真實資料跑完，不管 alpha/beta 怎麼調，`select_best_ordering()` 選出來的
永遠是「單獨用 UCF」，多階段串接一律更差、甚至 `total_frr` 會超過 1（不合理的
錯誤率）。逐條核對論文公式後確認：這是**公式(17)本身的數學性質，不是程式錯誤**。

`total_frr = Σ_s (Π_{j<s} RB_lb[j]) × RB_ub[s]`，第一項 `RB_ub[1] = conb₁/AB_ub[1]`，
因為 `AB_ub[1] < 1`，這一項恆 `> conb₁`；之後每多一階段只會再疊加一個恆正的項。
**所以任何串接排列的 total_frr，數學上都不可能低於「單獨用 conb 最小的偵測器」
的極限值**——這對任何 alpha、任何偵測器組合都成立，是公式結構決定的，不是我們
這三個偵測器資料品質特別差才會這樣。

論文自己在 Section IV-B 提到 CISO 可以設定「minimum number of required
modalities」這個政策參數——這正是 Defense-in-Depth 的精神：就算單一偵測器
理論上 FRR 更低，也不能只依賴單一因子（單點故障、容易被針對性繞過欺騙），
所以才需要強制要求至少用多個偵測器。**這就是為什麼要新增 `min_stages` 參數**
（見 `SECURITY_LEVELS` 定義與 `select_best_ordering()`）：沒有這個參數，
選擇邏輯永遠只會選單一偵測器，跟論文的 DiD（多層防禦）設計初衷矛盾。

### 3. SECURITY_LEVELS 的 alpha/beta 已重新校準

任務書範例的 `alpha`/`beta` 數字（0.005~0.10）是通用範例，套用在這三個偵測器
目前的實際表現上完全不可行——SBI/Xception/UCF 的 `conb` 分別是 0.324 / 0.171 /
0.137，遠高於一般生物特徵辨識常見的 EER 量級（論文範例裡是 0.005~0.01）。
用 `dump_all_orderings()` 掃過每個 `min_stages` 門檻下實際可達到的最小
`total_frr` 後，把三組安全等級改成：

```python
SECURITY_LEVELS = {
    "fast_screen": {"alpha": 0.05, "beta": 0.15, "latency_budget_ms": 1500,    "min_stages": 1},
    "balanced":    {"alpha": 0.05, "beta": 0.65, "latency_budget_ms": 4000,    "min_stages": 2},
    "forensic":    {"alpha": 0.3,  "beta": 0.6,  "latency_budget_ms": 100000, "min_stages": 3},
}
```

**注意 forensic 的 `alpha`（0.3）反而比 balanced（0.05）寬鬆**——這不是打錯字。
實測發現 `min_stages=3` 時，`alpha` 必須放寬到 ≥0.15 左右，3 階段反查回門檻後
才能滿足 `threshold_L < threshold_H`（alpha 越嚴，早期階段的 `AB_lb` 被壓得越小，
`RB_lb = conb/AB_lb` 越容易反查不出合法門檻）。這是「要求更多層防禦」與
「FAR 上限更嚴」在目前偵測器品質下互相衝突的結果——之後如果偵測器的 conb
降低（分類效果變好），這幾組數字應該重新校準。

---

## 最終結果

跑 `python build_lookup_table.py` 印出的三個偵測器 Profile：

```
DetectorProfile(name='SBI',      EER=0.5644 @T=0.313, conb=0.323807, Tb=2676.2ms)
DetectorProfile(name='Xception', EER=0.4134 @T=0.501, conb=0.171408, Tb=158.8ms)
DetectorProfile(name='UCF',      EER=0.3408 @T=0.498, conb=0.137441, Tb=224.8ms)
共窮舉 15 種排列組合（偵測器：['SBI', 'Xception', 'UCF']）
```

三個安全等級最終選中的排列：

| security_level | 選中排列 | system_total_far | system_total_frr | 總延遲 |
|---|---|---|---|---|
| fast_screen | UCF | 0.00037 | 0.137 | 224.8ms |
| balanced | UCF → Xception | 0.05 | 0.609 | 383.6ms |
| forensic | UCF → Xception → SBI | 0.30 | 0.571 | 3059.8ms |

`output/real_data_vip_settings.csv` 內容（`validate.py` 已確認：每個
`security_level` 底下 `stage` 從 1 連續遞增、每列 `threshold_L < threshold_H`）：

| security_level | stage_order | stage | detector_name | threshold_L | threshold_H | expected_latency_ms |
|---|---|---|---|---|---|---|
| fast_screen | UCF | 1 | UCF | 0.050 | 0.775 | 224.81 |
| balanced | UCF,Xception | 1 | UCF | 0.131 | 0.775 | 224.81 |
| balanced | UCF,Xception | 2 | Xception | 0.050 | 0.591 | 158.79 |
| forensic | UCF,Xception,SBI | 1 | UCF | 0.362 | 0.650 | 224.81 |
| forensic | UCF,Xception,SBI | 2 | Xception | 0.391 | 0.591 | 158.79 |
| forensic | UCF,Xception,SBI | 3 | SBI | 0.050 | 0.527 | 2676.15 |

`output/all_orderings.csv` 是不套用 `beta`/`min_stages` 篩選、對全部 15 種排列
都用同一個 `alpha`（預設 0.05）求解的診斷表，每個排列另外標了 `valid_thresholds`
欄位（該排列反查門檻後是否每階段都 `threshold_L < threshold_H`）。想觀察「串接
到底值不值得」時可以直接看這張表，不用重新調 `SECURITY_LEVELS` 反覆試。

---

## 已知限制 / 之後可以做的事

- **延遲是 worst-case 加總，不是期望延遲**：目前 `latency_budget_ms` 篩選用的是
  「排列裡所有偵測器 Tb 直接加總」，這是任務書步驟4第1點原本的做法，但這代表
  串接只會讓 worst-case 延遲變差，體現不出「大部分容易判斷的樣本在第一階段就
  分流掉、只有少數疑難樣本才會真的跑到後面較慢偵測器」這種 cascade 真正該有的
  期望延遲優勢。如果要更貼近論文 Fig.1 的設計初衷，需要額外的「各階段分流機率」
  資料才能算期望延遲，目前手上的 sweep 資料沒有這個維度。
- **conb 目前偏保守**：`compute_conb` 用 `max(EER², max(far×frr))`，SBI/Xception/UCF
  的 `max(far×frr)` 都只比 `EER²` 高一點點，不是估計方法本身有問題，純粹反映這三個
  偵測器目前的分類效果就是不夠好（EER 0.34~0.56）。偵測器效果提升後，`conb` 會
  自然降低，串接才會真的划算。
