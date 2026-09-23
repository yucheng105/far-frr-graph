# GP 約束與門檻反查修正說明

這份文件說明 `bmma_gpt_lookup` **這一支修改**做了什麼、為什麼要改、跑完之後表長什麼樣子。  
理論推導見 `GP_LOOKUP_TABLE_REVIEW.md`；從頭講專案背景見 `PROJECT_BRIEFING.md`。

**沒有改：** `SECURITY_LEVELS` 的 α / β、`combine_mode="max"`、`compute_conb`、延遲模型。那些是下一步。

---

## 為什麼要改

Lookup Table 是防禦系統要查的「用誰、什麼順序、每關門檻多少」。舊版 GP 解的題，和系統真正在跑的題不一樣：

1. **最後一關仍是雙門檻。** 中間帶（不確定、交給下一關）在最後一關沒有下一關可去。公式卻只對「明確放行」收 FAR、只對「明確擋下」收 FRR，猶豫等於免費。Solver 會把最後一關的 `T_L` 壓到 sweep 左端（0.05），留下巨大灰色地帶。例如舊的 `fast_screen`：UCF 的 `T_L=0.05`、`T_H=0.775`，表上 `total_far≈0.0004`。
2. **反查不對稱。** `T_L` 走 `inv_far`、`T_H` 走 `inv_frr`。conb 是 `max(FAR×FRR)` 上界、不是逐點等式，兩條曲線對不齊就會出現 `T_L > T_H`。超出曲線時還會 `np.clip` 裝成有解。
3. **CSV 印的是 GP 代數值**，不是這對門檻拿去排隊檢查時會量到的 FAR / FRR。

FAR / FRR 定義仍以 `src/threshold_sweep.py` 為準：分數 0 = Real、1 = Fake；`fake_score >= threshold` 判 Fake。FAR（漏抓）隨門檻遞增，FRR（錯殺）隨門檻遞減。

---

## 改了哪些檔案

| 檔案 | 改什麼 |
|---|---|
| `build_lookup_table.py` | GP 約束、反查、差式 cascade 重算 |
| `validate.py` | 最後一關改為檢查 `T_L ≈ T_H`，前 N−1 關仍要 `T_L < T_H` |
| `output/real_data_vip_settings.csv` | 依新模型重產 |
| `output/all_orderings.csv` | 依新模型重產（α=0.05 下目前為空，見下方結果） |

### `solve_gp_thresholds`

新增兩條仍符合 DGP 的約束：

- **最後一關單門檻：** `AB_lb[-1] == AB_ub[-1]` → 落地 `T_L == T_H`。最後一人必須給答案。
- **錯誤率必須是機率：** `AB_lb[i] >= conb[i]`（且 `AB_ub >= conb`）→ `RB <= 1`。否則階段乘積 `Π RB_lb` 可以大於 1，`total_frr` 會爆成 5.88 這種非機率。

另外：`assert problem.is_dgp()`；有 ECOS 就釘 ECOS；`optimal_inaccurate` 時若 `total_far` 超過 α 視為失敗。

公式 17 / 18 的 posynomial 寫法沒有改。

### 門檻反查

- `T_L = inv_far(AB_lb)`、`T_H = inv_far(AB_ub)`。FAR 嚴格遞增 ⇒ `AB_lb <= AB_ub` 自動得到 `T_L <= T_H`。
- 最後一關兩個 AB 相等，反查後取平均，保證單一門檻。
- 目標 FAR 超出該偵測器曲線觀測範圍 → 回傳失敗，**不再 clip**。
- 通過後用差式 cascade 重算系統錯誤率，寫進 CSV：

```text
FAR = Σ_s FAR_s(T_L) · Π_{j<s} [FAR_j(T_H) − FAR_j(T_L)]
FRR = Σ_s FRR_s(T_H) · Π_{j<s} [FRR_j(T_L) − FRR_j(T_H)]
```

這與 `progress_report.md` 的真實 cascade 定義一致。最後一關 `T_L == T_H` 時續傳機率為 0。

β 篩選改看**重算後**的系統 FRR，不再只用 GP 目標函數。

---

## 重跑結果（預期現象，不是回歸失敗）

三個 `DetectorProfile` 的 conb 仍是合併後的數字（尚未修 combine）：SBI 0.324、Xception 0.171、UCF 0.137。

| 安全等級 | α | min_stages | 結果 |
|---|---|---|---|
| `fast_screen` | 0.05 | 1 | **無解。** 單一 UCF 的 conb≈0.137 已大於 α=0.05，加上 `AB >= conb` 後 GP 直接不可行。 |
| `balanced` | 0.05 | 2 | **無解。** 同一個原因。 |
| `forensic` | 0.3 | 3 | **有解：** UCF → Xception → SBI |

`forensic` 落地門檻：

| stage | detector | T_L | T_H |
|---|---|---|---|
| 1 | UCF | 0.319 | 0.416 |
| 2 | Xception | 0.465 | 0.581 |
| 3 | SBI | **0.318** | **0.318**（單一門檻） |

差式重算：系統 FAR = 0.190（≤ 0.3）、FRR = 0.479（≤ 0.6）。最後一關不再出現 `T_L=0.05` 配巨大灰色地帶。

`dump_all_orderings` 仍固定 α=0.05，因此 15 種排列 GP 全部失敗，`all_orderings.csv` 目前是空的。這表示「用現在這把被 pointwise-max 灌水的尺子，5% 漏抓做不到」，**不要先把 α 放寬來硬解**。

---

## 怎麼跑

```bash
cd bmma_gpt_lookup
python -m venv .venv
# Windows: .venv\Scripts\pip install -r requirements.txt
# Linux/macOS: .venv/bin/pip install -r requirements.txt
.venv/Scripts/python build_lookup_table.py    # Windows
.venv/bin/python build_lookup_table.py        # Linux/macOS
.venv/Scripts/python validate.py
```

`validate.py`：空表會 FAIL；有列時最後一關必須 `T_L ≈ T_H`，其餘關 `T_L < T_H`。

---

## 刻意沒改、下一步才做

1. **修 combine：** 現在 `combine_mode="max"` 會把不同資料集的最差 FAR 與最差 FRR 拼成一條不存在的曲線，conb 被灌得比 FF++ 基準（UCF EER≈0.05）大很多。應改選 conb 最大的**整條**資料集曲線，再回頭看 α=0.05 能不能解。
2. **重校 `SECURITY_LEVELS`：** forensic 的 α=0.3 是舊漏洞下試出來的。combine 修完再用新診斷表重設 α / β。
3. **期望延遲、獨立 test set：** 門檻語意對了再做。不要把 `E[L]` 塞進 cvxpy（續傳機率是相減，不是 DGP）。

---

## 給審查的檢查清單

- [ ] 最後一關 `T_L == T_H`（本輪 forensic / SBI 已是 0.318 / 0.318）
- [ ] 前幾關 `T_L < T_H`，中間帶還在
- [ ] `fast_screen` / `balanced` 在 α=0.05 無解，且診斷訊息指向 conb > α，而不是程式崩潰
- [ ] CSV 的 `system_total_far/frr` 是差式 cascade，不是 GP 目標函數（forensic 的 GP FRR 仍可能 > 1，那是上界；落地 FRR 是 0.479）
- [ ] 未改 `models/`、`results/`、`input-csvs/`
