# bmma_gpt_lookup：GP 模型與 Lookup Table Code Review

- **審查對象：** `bmma_gpt_lookup/build_lookup_table.py`（步驟 3：cvxpy DGP；步驟 4：門檻反查與 CSV 匯出）
- **對照輸出：** `output/real_data_vip_settings.csv`、`output/all_orderings.csv`
- **理論依據：** BMMA-GPT 論文公式 17 / 18（posynomial 上界），以及本專案 `progress_report.md` 對真實 cascade 的定義（最後一關必須做出最終決策）
- **FAR / FRR 定義：** 以 `src/threshold_sweep.py` 為準。分數 0 = Real、1 = Fake；`fake_score >= threshold` 判 Fake。FAR（漏抓假片）隨門檻遞增，FRR（錯殺真片）隨門檻遞減。
- **狀態：** 本文件只做審查與理論說明，**尚未修改程式碼**。

---

## 一句話結論

公式 17 / 18 的 posynomial 寫法**符合 DGP**，solver 也接得上。真正會讓 Lookup Table 失真的，是：

1. 最後一關仍允許雙門檻，中間帶樣本沒被算進系統錯誤率，求解器會把 `T_L` 推到 EPS；
2. 反查時下界走 `inv_far`、上界走 `inv_frr`，在 conb 偏鬆時會得到 `T_L > T_H`。

現有 `fast_screen` 已是第一個洞的實例：UCF 的 `T_L = 0.05`（sweep 左端）、`T_H = 0.775`、`total_far ≈ 3.6e-4`。

---

## 問題清單

| ID | 嚴重度 | 區塊 | 問題 | 實測證據 |
|---|---|---|---|---|
| F1 | 高 | GP 語意 / 最後一關 | 最後一關仍是雙門檻，中間帶從 FAR/FRR 消失 | `fast_screen` 的 UCF：`T_L=0.05`、`T_H=0.775`、`total_far≈3.6e-4≈EPS` |
| F2 | 高 | 門檻反查 | `T_L` 用 `inv_far`、`T_H` 用 `inv_frr`，conb 一鬆就 `T_L > T_H` | `all_orderings` 在 `alpha=0.05` 的 6 種三階段排列，`valid_thresholds` 全是 False |
| F3 | 高 | DGP 缺約束 | 未限制 `RB ≤ 1`（即 `AB ≥ conb`），`total_frr` 可爆到 >1 | `SBI→Xception→UCF` 的 `system_total_frr ≈ 5.88` |
| F4 | 高 | conb | `combine_mode=max` 把不同資料集的最差 FAR 與最差 FRR 拼成一條不可能曲線 | FF++ 上 UCF EER≈0.05，合併後 Profile EER=0.34、conb=0.137 |
| F5 | 中 | Lookup Table | CSV 寫的是 GP 模型值，不是實際 `T_L`/`T_H` 在曲線上的系統 FAR/FRR | `inv_*` 會 clip；反查後沒有重算 cascade 錯誤率 |
| F6 | 中 | 門檻反查 | `inv_far` / `inv_frr` 超出曲線範圍時靜默 clip，沒有標記失敗 | 多數 last-stage 的 `T_L` 都被夾到 0.05 |
| F7 | 中 | 診斷輸出 | `dump_all_orderings` 鎖死 `alpha=0.05`，forensic 的 0.3 可行性看不見 | `ALL_ORDERINGS_ALPHA = 0.05`，三階段在診斷表上全是 invalid |
| F8 | 低 | DGP / solver | 接受 `optimal_inaccurate`、未 assert `is_dgp()`、未釘 solver | `problem.solve(gp=True)` 無約束違規檢查 |

以下第 1、2 節把 F1 / F3 與 F2 講清楚（這兩點會直接改變門檻語意，改 code 前必須對齊）。F4–F8 在第 3、4 節較短說明。

---

## 1. DGP 限制式：形式合格，模型語意不完整

### 1.1 哪些寫法是合法的 DGP

決策變數 `AB_lb`、`AB_ub` 宣告 `pos=True`；`RB = conb / AB` 是 monomial；`total_frr` 是 monomial 之和（posynomial）；`total_far ≤ α` 是 posynomial ≤ 正常數。`AB_lb ≤ AB_ub` 等價於 `AB_lb / AB_ub ≤ 1`，同樣合法。

| 表達式 | 形式 | DGP？ | 評語 |
|---|---|---|---|
| min Σ (Π_{j<s} RB_lb[j]) · RB_ub[s] | posynomial | 是 | 論文公式 17，寫法正確 |
| Σ (Π_{j<s} AB_ub[j]) · AB_lb[s] ≤ α | posynomial ≤ 常數 | 是 | 論文公式 18，寫法正確 |
| AB_lb[i] ≤ AB_ub[i] | monomial ≤ monomial | 是 | 等價 T_L ≤ T_H（FAR 遞增） |
| EPS ≤ AB ≤ 1−EPS | monomial 邊界 | 是 | 合法，但下限太鬆（見 F3） |
| AB_lb[N] == AB_ub[N] | monomial 等式 | **缺** | 最後一關應單門檻 |
| AB[i] ≥ conb[i]  ⟺  RB[i] ≤ 1 | monomial ≤ 1 | **缺** | 沒加的話 RB 可 >1，三階段 FRR 爆掉 |

缺的不是「寫成非凸」，而是少了兩條**仍是 DGP** 的約束。

### 1.2 實務上，一個雙門檻偵測器在做什麼

對應論文的 dual-threshold、也對應本專案 `progress_report.md`：

- `score < T_L`：判 Real（放行 / Accept）
- `score > T_H`：判 Fake（拒絕 / Reject）
- `T_L ≤ score ≤ T_H`：**不決定，交給下一關**

中間帶存在的唯一理由，是後面還有偵測器。因此中間帶的機率 = 進到下一關的機率。

真實 cascade（差式，不是 GP 鬆弛）為：

```text
FAR = Σ_s  FAR_s(T_L^s)  · Π_{j<s} [ FAR_j(T_H^j) − FAR_j(T_L^j) ]
FRR = Σ_s  FRR_s(T_H^s)  · Π_{j<s} [ FRR_j(T_L^j) − FRR_j(T_H^j) ]
```

並且：**最後一關必須迫使系統做出最終決策**，即 `T_L^(N) = T_H^(N)`。若最後一關還留中間帶，那些樣本沒有下一關可去，驗證系統等於對一部分輸入交白卷。

`T_L = T_H = T*` 時，最後一關續傳機率為 0，進到最後一關的人全部被單一門檻切成 Real 或 Fake。這是「序列決策必須終止」的定義，不是實作偏好。

### 1.3 論文 GP 在鬆弛什麼（階段乘積從哪來）

差式 `FAR(T_H) − FAR(T_L)` 是兩個 posynomial 相減，**不是 DGP**。論文為了讓問題能進幾何規劃，改用上界：

```text
FAR(T_H) − FAR(T_L)  ≤  FAR(T_H) = AB_ub
FRR(T_L) − FRR(T_H)  ≤  FRR(T_L) = RB_lb
```

於是「能活到第 s 關」的機率，被正向項的階段乘積取代：

| 路徑 | 真實續傳 | GP 用來當上界的階段乘積 |
|---|---|---|
| 假片（FAR） | Π_{j<s} (AB_ub^j − AB_lb^j) | Π_{j<s} AB_ub^j |
| 真片（FRR） | Π_{j<s} (RB_lb^j − RB_ub^j) | Π_{j<s} RB_lb^j |

公式 17 / 18：

```text
total_frr = Σ_s  (Π_{j<s} RB_lb^j)  · RB_ub^s
total_far = Σ_s  (Π_{j<s} AB_ub^j)  · AB_lb^s   ≤ α
```

每一項都是：**（前面各關「不決定」的乘積）×（本關真正做出 Accept 或 Reject 的機率）**。

這個上界要成立，前提是：每一關被算進 FAR / FRR 的，都是真正做了決定的樣本。中間帶只能出現在「後面還有關卡」的階段。

### 1.4 少掉 `AB_lb[N] == AB_ub[N]`，求解器鑽的洞

最後一關的貢獻是：

- FAR 項：`(Π_{j<N} AB_ub^j) · AB_lb^N`
- FRR 項：`(Π_{j<N} RB_lb^j) · RB_ub^N`

不對稱：最後一關 FAR 只看下門檻 `AB_lb`，FRR 只看上門檻 `RB_ub = conb / AB_ub`。中間帶 `AB_ub − AB_lb` **沒有被算成任何錯誤**。

沒有最後一關等式時，兩個旋鈕可以往反方向轉：

1. 把 `AB_lb^N` 壓到 `1e-6` → 最後一關 FAR 項 ≈ 0，α 約束幾乎免費滿足。
2. 把 `AB_ub^N` 拉到 `1 − 1e-6` → `RB_ub^N ≈ conb`，最後一關 FRR 被壓到只剩 conb。

Solver 最小化的是公式 17 那個**代數式**，不是部署後量到的 FRR。它會找到：最後一關留一個巨大灰色地帶，公式裡的 FAR 趨近 0、FRR 趨近 conb。這正是現在的 `fast_screen`。

白話：公式只對「明確放行」收 FAR、只對「明確拒絕」收 FRR；最後一關的猶豫樣本兩邊都不付錢。優化器當然把猶豫區開到最大。

單一偵測器（N=1）時這個洞最大：根本沒有下一關，卻仍在解雙門檻。論文 GP 是在給**會終止的驗證流程**找門檻；少了最後一關等式，解的是另一個問題——允許對一部分人不出答案的系統。

對應關係：

- 第 1 到 N−1 關：雙門檻合法，中間帶 = 續傳。
- 第 N 關：必須 `T_L = T_H`，也就是 `AB_lb[N] == AB_ub[N]`（FAR 遞增 ⇒ 同一個 T）。這是 monomial 等式，仍是 DGP。

### 1.5 為什麼少 `RB ≤ 1`（`AB ≥ conb`）會爆炸

守恆關係 `RB = conb / AB`。`AB`、`RB` 在論文裡是錯誤**機率**，必須落在 (0, 1)。若 `AB_lb < conb`，則 `RB_lb > 1`，它不再是機率，只是一個可以大於 1 的正數。

公式 17 的階段乘積會把這個數一路乘下去。例如 SBI 的 conb ≈ 0.32，solver 為了省 α 把第一關 `AB_lb` 設成 0.01，則 `RB_lb = 32`。語意上等於在說「真片活過第一關的機率 ≤ 32」。之後：

- 第 2 關 FRR 項：`32 × RB_ub[2]`
- 第 3 關 FRR 項：`32 × RB_lb[2] × RB_ub[3]`

這就是三階段 `total_frr` 可以到 5.88 的原因：不是影片被拒了 588%，是「活到後關的上界」已經不是機率。

對照兩條路徑：

- FAR 路徑 `Π AB_ub`：每一項本身 ≤ 1，乘積愈乘愈小。
- FRR 路徑 `Π RB_lb`：只有 `RB_lb ≤ 1` 時才有同樣性質。少了這條，乘積可以愈乘愈大。

為什麼 solver 會把 `AB_lb` 壓小？因為公式 18 是各關 FAR 項加總 ≤ α。最後一關壓 `AB_lb` **完全不罰 FRR**（最後一關 FRR 看的是 `RB_ub` 不是 `RB_lb`）。前關壓 `AB_lb` 會讓後面 FRR 變差，有 trade-off；但只要 conb 偏大、α 又緊，`RB_lb` 仍常被推出 (0, 1)。

**副作用必須講清楚：** 以目前合併後 UCF 的 conb ≈ 0.137 來說，加上 `AB ≥ conb` 後，連「單一 UCF、α=0.05」都會不可行（0.137 已經超過 α）。這不代表約束是錯的，而代表**現在的 conb 已經大到連機率單純形都裝不進 α**。約束暴露的是模型與資料的矛盾。

另外建議：`assert problem.is_dgp()`、拒絕 `optimal_inaccurate`（除非 `total_far` 仍滿足 α）、釘死 solver（ECOS / Clarabel）。

---

## 2. 門檻反查不對稱

現況（`_reverse_map_thresholds`）：

```text
T_L = inv_far(AB_lb)          ← 走真實 FAR 曲線
T_H = inv_frr(RB_ub)
    = inv_frr(conb / AB_ub)   ← 走被 conb 放大的 FRR
```

### 2.1 論文想對齊的是同一條雙曲線上的兩個點

每個門檻 T 對應曲線上一個點 `(FAR(T), FRR(T))`。論文用

```text
FAR(T) · FRR(T)  ≈  conb    ⇒    FRR(T)  ≈  conb / FAR(T)
```

把 DET 收成雙曲線，GP 才能寫 monomial。若 conb 在 `T_L` 和 `T_H` **都是等式**，四種反查會得到同一對門檻：

```text
T_L = inv_far(AB_lb) = inv_frr(RB_lb)
T_H = inv_far(AB_ub) = inv_frr(RB_ub)
```

現在用的是「下界走 FAR、上界走 FRR」。conb 是等式時沒問題；conb 是**上界**時會系統性地把 `T_H` 往左推。

### 2.2 衝突的幾何（為什麼會 `T_L > T_H`）

本專案 `conb = max_T (FAR(T) · FRR(T))`，所以對任何 T：

```text
FRR_實際(T)  ≤  conb / FAR(T)
```

GP 交給反查的 `RB_ub = conb / AB_ub` 是**高估的 FRR**。

兩條路去找 `T_H`：

1. FAR 一致：`T_H^FAR = inv_far(AB_ub)`，此處實際 FRR ≤ `RB_ub`。
2. FRR 一致：`T_H^FRR = inv_frr(RB_ub)`，要的是實際 FRR **等於**那個被高估的值。

FRR 隨門檻遞減：要達到更高的 FRR，門檻必須更低。因此 `T_H^FRR ≤ T_H^FAR`。conb 愈鬆，`T_H` 被往左拉得愈厲害。`T_L` 卻仍依真實 FAR 曲線。

當 α 很緊、前關 `AB_ub` 也被壓小時，`RB_ub` 會大到超過曲線上最大的 FRR，`inv_frr` 再靜默 clip 到 sweep 左端（T=0.05）。於是出現 `all_orderings` 裡第一關 `T_L=0.19`、`T_H=0.05`。

不是 FAR 遞增、FRR 遞減壞了，是**兩次反查各自對齊一條曲線，而被一條過高的雙曲線硬綁在一起**。

現有處理：`select_best_ordering` 會丟棄 `T_L ≥ T_H` 的候選，這步是對的，但只檢查順序、不檢查 clip、也不重算錯誤率。診斷還把「α 解不出來」和「β 太嚴」混在同一句。

### 2.3 兩邊都改 `inv_far`，能保證 `T_L ≤ T_H` 嗎？

能。前提只有已經 enforce 過的單調性：FAR 隨 T 不遞減（程式還加了極小遞增量，變成嚴格遞增），以及 GP 約束 `AB_lb ≤ AB_ub`。

嚴格遞增函數的反函數仍嚴格遞增：

```text
AB_lb ≤ AB_ub    ⇒    inv_far(AB_lb) ≤ inv_far(AB_ub)
```

這保證的是**門檻順序**，不依賴 conb。最後一關若再加上 `AB_lb == AB_ub`，會得到 `T_L = T_H`，與「最後一關單一門檻」一致。

建議：clip 時視為反查失敗；通過後用 `far_at` / `frr_at`（最好用差式 cascade 公式）重算再寫入 CSV。

### 2.4 這樣還能保證 FRR ≤ β 嗎？

在三個條件同時成立時：**可以保證論文那條 GP 上界仍然 ≥ 真實 cascade FRR**，因此 GP 的 `total_frr ≤ β` 仍蘊涵真實 FRR ≤ β。

條件：

1. conb 仍是 `FAR · FRR` 的上界（現在的 `max` 滿足；它偏保守，不是偏樂觀）；
2. 反查沒有 clip（GP 要的 FAR 落在曲線觀測範圍內）；
3. 最後一關確實單一門檻，系統對每個輸入都做完決定。

理由：`T_H = inv_far(AB_ub)` 時，實際 `FAR(T_H) = AB_ub`，且

```text
FRR_實際(T_H)  ≤  conb / AB_ub  =  RB_ub^GP
```

同理 `FRR_實際(T_L) ≤ RB_lb^GP`。公式 17 用的每個因子都 ≥ 真實對應機率。真實 cascade 還用更緊的差式，所以：

```text
FRR_真實 cascade  ≤  total_frr^GP  ≤  β
```

β 不會因為改走 `inv_far` 而變得比較鬆。實際 FRR 只會低於或等於 GP 宣稱的值。

會讓 β 保證斷掉的是另一個方向：

- conb **低估**乘積（例如直接改用 EER²）→ GP 以為 FRR 很小 → 篩過 β，實際上線超過 β；
- `inv_*` 發生 clip → 落地門檻達不到 GP 要的 AB / RB；
- 最後一關仍留灰色地帶 → 公式 17 根本沒在為那個系統負責。

兩邊 `inv_far` **不會犧牲 β 的保守性**；它放棄的是「把 GP 的 `RB_ub` 逐點對到 FRR 曲線」，換來門檻順序良定義。論文的初衷是 conb 很緊、雙曲線幾乎貼著 DET，那時兩種反查本來就會重合。現在 conb 偏高，硬把上界對到 FRR 曲線，才是在違背「這條雙曲線只是上界」。

若要既保順序、又讓 CSV 數字對得上 β：反查出 `(T_L, T_H)` 之後，用真實曲線（最好用差式 cascade 公式）再算一次系統 FRR，用那一個數字去跟 β 比。GP 負責在 DGP 裡找到候選；β 是對真實系統的約束，應該用真實系統的錯誤率來驗。

---

## 3. conb 為什麼那麼緊（F4）

`conb = max(EER², max(FAR × FRR))` 作為 GP 需要的單一正常數，方向是對的。數字被灌水的是前面的 **pointwise max 合併**，不是這一行 max。

| 來源 | UCF EER | UCF conb | 意義 |
|---|---|---|---|
| FaceForensics 基準（任務書） | ≈ 0.05 @ T=0.79 | ≈ 0.0025（EER²） | 單資料集真實水準 |
| 合併後 DetectorProfile | 0.341 @ T=0.498 | 0.137 | GP 實際在優化的偵測器 |
| SBI 合併後 | 0.564 | 0.324 | 三階段幾乎被這顆拖死 |

同一門檻 T，資料集 A 可能是 FAR=0.40 / FRR=0.05，B 是 FAR=0.05 / FRR=0.34。pointwise max 得到 FAR=0.40 / FRR=0.34，乘積 0.136——沒有任何真實資料集這麼差。`enforce_monotonic` 的 accumulate 還會再抬一次。

不要先把 conb 改成 EER² 來「放寬門檻」：那會讓 GP 樂觀，實測 FRR 可能超過 β。較穩的改法：改選 **conb 最大的那一條完整資料集曲線**，不要逐點拼 FAR 與 FRR。若還要再鬆，只在操作窗（例如 FAR、FRR 都在 0.02–0.8）算 max product。

順序應是：先修最後一關與反查，再動 combine。

---

## 4. 期望延遲擴充性

延遲目前不在 GP 裡，只在 `select_best_ordering` 當篩選鍵，這點對擴充有利。問題是欄位名叫 `expected_latency_ms`，值卻是跨資料集 max 的 `Tb`，排序用 Σ Tb（假設每層都跑完）。

真實期望延遲 `E[L] = T1 + P(進第2層)·T2 + …`，而 `P(continue) = FAR(T_H) − FAR(T_L)` 是兩個 posynomial 相減，**不是 DGP**。

建議：GP 維持 `min FRR s.t. FAR ≤ α`；新增 LatencyModel（WorstCase / Expected），在反查之後用真實曲線與 class prior 算 `E[L]` 再排序。CSV 分開 `worst_case_ms` 與 `expected_ms`。若一定要在 GP 內限制延遲，只能用 `AB_ub` 當續傳上界，那仍是 worst-case。

---

## 建議修改順序（尚未實作）

| 順序 | 改什麼 | 目的 |
|---|---|---|
| 1 | 最後一關 `AB_lb == AB_ub`，並限制 `RB ≤ 1` | 堵住 GP 把 `T_L` 推到 EPS 的洞 |
| 2 | `T_L`、`T_H` 都用 `inv_far`；clip 當失敗；重算真實 FAR/FRR 再寫 CSV | 反查強健、lookup 數字可信 |
| 3 | combine 改選最差整條資料集曲線 | conb 不再被假曲線灌水 |
| 4 | `dump_all_orderings` 依各 security_level 的 α 各跑一次 | forensic 校準看得到 |
| 5 | LatencyModel 介面，CSV 分開 `worst_case_ms` / `expected_ms` | 之後加期望延遲不必動 GP |

第 1、2 步會直接改變 `real_data_vip_settings.csv` 的門檻，需要先對齊理論再動 code。第 1 步加上 `RB ≤ 1` 後，以目前 conb 量級，`α = 0.05` 可能全面不可行——那是資料／合併策略的問題浮上來，不是約束加錯。

---

## 5. 接下來可能要做什麼

建議先跟寫 GP 的朋友對齊第 1、2 節理論，再動 code。實作時依下面順序，不要先改 conb 或 α 來「讓表看起來比較好看」。

### Phase 0：對齊（不改 code）

- [ ] 確認最後一關必須 `T_L = T_H`（`progress_report.md` 已寫死；對應 GP 的 `AB_lb[N] == AB_ub[N]`）
- [ ] 確認反查改兩邊 `inv_far` 的前提：conb 當上界，不當逐點等式
- [ ] 確認 `RB ≤ 1` 要加，即使目前 conb 會讓 `α = 0.05` 暫時無解——那是下一步要修 combine，不是把約束拿掉

### Phase 1：修 GP 語意（F1、F3）

改 `solve_gp_thresholds`：

- [ ] 最後一關 `ab_lb[-1] == ab_ub[-1]`
- [ ] `ab_lb[i] >= conb[i]`（等價 `RB <= 1`）
- [ ] `assert problem.is_dgp()`；`optimal_inaccurate` 要檢查 `total_far` 仍 ≤ α；solver 釘死

跑完後預期：`fast_screen` 的 UCF 不再出現 `T_L=0.05`、`T_H=0.775` 這種巨大灰色地帶；`α=0.05` 可能全面無可行解。這是正常現象，進入 Phase 3 前不要先把 α 放寬來硬解。

### Phase 2：修反查與 Lookup 數字（F2、F5、F6）

改 `_reverse_map_thresholds` / `select_best_ordering`：

- [ ] `T_L = inv_far(AB_lb)`、`T_H = inv_far(AB_ub)`；最後一關兩者相等
- [ ] 超出曲線範圍改為反查失敗，不要 `np.clip` 裝成有解
- [ ] 反查成功後，用真實曲線重算系統 FAR/FRR 再寫 CSV（優先用 `progress_report.md` 的差式 cascade，不要只抄 GP 代數值）
- [ ] 診斷拆開：GP 不可行 / β 失敗 / 反查失敗

### Phase 3：讓 conb 回到真實資料集（F4）

- [ ] `combine_mode` 改為「選 conb 最大的那一條完整曲線」，不要逐點 max FAR 再 max FRR
- [ ] 重新印三個 `DetectorProfile` 的 EER / conb，對照 FF++ 基準（UCF EER≈0.05、Xception≈0.06）
- [ ] 此時再回頭看 Phase 1 的 `α=0.05` 是否恢復可行

可選、較後面：只在操作窗（例如 FAR、FRR ∈ [0.02, 0.8]）算 max product。不要先改成 `EER²`，那會讓 β 保證變樂觀。

### Phase 4：重新校準安全等級並重產表（F7）

模型語意正確、conb 合理之後，才動 `SECURITY_LEVELS`：

- [ ] `dump_all_orderings` 依 `fast_screen` / `balanced` / `forensic` 各自的 α 各跑一次
- [ ] 用新的最小 `total_frr` 重設 β；forensic 的 α 是否仍需 0.3 以新數字為準，不要沿用舊校準
- [ ] 重跑 `build_lookup_table.py` + `validate.py`，產出新的 `real_data_vip_settings.csv`

### Phase 5：之後再說（非修 bug）

- [ ] LatencyModel：CSV 分開 `worst_case_ms` / `expected_ms`；期望延遲在反查後算，不要塞進 DGP
- [ ] 用與校準資料不重疊的 test set 評估上線後的系統 FAR/FRR（腳本開頭的資料洩漏警告）

### 不建議現在做的事

- 先把 forensic 的 α 再放寬、或把 conb 改成 EER²，只為了讓三階段「有解」
- 在最後一關語意修正前，解讀現有 lookup 表的 `system_total_far` / `system_total_frr` 為真實系統表現
- 把期望延遲寫進 cvxpy（`AB_ub − AB_lb` 不是 DGP）
