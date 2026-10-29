# 曲線、C 與 GP（以 FaceForensics++ 為例）

這份說明講三件事怎麼串起來：影片分數怎麼變成曲線、曲線怎麼算出常數 C、GP 怎麼用 C。平行式融合不用曲線上的 C，它直接數每支影片的分數。

C 不是專題要交的答案。查表上的順序、門檻、實際 FAR、FRR 和延遲才是。C 是 GP 開算之前，從曲線上讀出來的輸入。

## 曲線是什麼

曲線不是圖檔。它是一張 CSV：同一個偵測器、同一批影片，把門檻從 0.05、0.10 一直放到 1.00。每一列只有三個有用的欄位：門檻、FAR、FRR。

分數來自更前面的檔案。一支 FaceForensics++ 影片經過 UCF，得到一列 `fake_score`（0 到 1，愈高愈像偽片）和真偽標籤。FaceForensics++ 這批是 140 支真片、560 支偽片。Xception、SBI 也各跑一次，所以同一支影片有三個分數。GP 不讀這些原始分數。

`threshold_sweep.py` 把原始分數收成曲線。它不重新跑模型，只對每個門檻數一次：

- FAR = 偽片裡分數**低於**這個門檻、因而被放行的支數 / 560
- FRR = 真片裡分數**達到**這個門檻、因而被擋下的支數 / 140

門檻愈高，愈不容易被判成偽片，FAR 上升、FRR 下降。這整張表就是「門檻拉高時，兩種錯誤怎麼變」。

UCF 在 FaceForensics++ 上的前幾列（`results/20/ucf/threshold_sweep_ucf_faceforensics.csv`）：

| 門檻 | FAR | 大約是幾支偽片 | FRR | 大約是幾支真片 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.001786 | 1/560 | 0.828571 | 116/140 |
| 0.10 | 0.003571 | 2/560 | 0.750000 | 105/140 |
| 0.20 | 0.003571 | 2/560 | 0.542857 | 76/140 |

門檻 0.05 幾乎把所有真片都擋下來（FRR 83%），偽片幾乎不會漏（FAR 0.2%）。門檻拉到 0.20，漏抓仍約 2 支偽片，誤擋降到 76 支真片。GP 之後選門檻，就是在這張表上選一列或兩列。

## 為什麼一個資料集一條曲線

同一個偵測器、同一個門檻，換一批影片，比例會變。UCF、門檻都是 0.20：

| 資料集 | FAR | FRR |
|---|---:|---:|
| FaceForensics++ | 0.4%（約 2/560） | 54%（約 76/140） |
| Celeb-DF-v2 | 5.0% | 65% |
| DFDC | 8.5% | 69% |

所以 UCF 有三條曲線，不是一條。三個偵測器一共九條。這週若只做 FaceForensics++，GP 只讀其中三條：

| 偵測器 | 使用的曲線檔 |
|---|---|
| UCF | `exhaustive-search/threshold-sweep/results/20/ucf/threshold_sweep_ucf_faceforensics.csv` |
| Xception | `exhaustive-search/threshold-sweep/results/20/xception/threshold_sweep_xception_faceforensics.csv` |
| SBI | `bmma_gpt_lookup/threshold_sweep_sbi_faceforensics_v1.csv` |

不要用檔名帶 `combined` 的曲線。那個檔是三個資料集已經併過的結果。現在的 `build_lookup_table.py` 還會把三條曲線在每個門檻上取最差的 FAR 和最差的 FRR，再拼成一條新的。以門檻 0.20 的 UCF 為例，FAR 會取到 DFDC 的 8.5%，FRR 會取到三條裡比較高的那個。沒有任何一批影片同時是這兩個數字。FaceForensics++ 的 GP 不要吃這條拼出來的表。

## 一條曲線、一個 C

FaceForensics++ 上，每個偵測器的那張門檻表各算出一個 C。

C 的算法與 `build_lookup_table.py` 的 `compute_conb()` 相同。對這一張表：

1. 每個門檻算一次 `FAR × FRR`。
2. 取出其中最大的乘積。
3. 找出 FAR 與 FRR 最接近的那一點，稱為 EER，再算 `EER²`。
4. C 取第 2 步和第 3 步裡比較大的那個。

取最大，是為了讓下面這句在每個門檻都成立：

```text
真實 FRR ≤ C / FAR
```

所以 C 是上界用的常數，不是每一格乘積的平均。程式裡的變數名稱是 `conb`。

用 `results/20/` 的 FaceForensics++ 曲線（間距 0.05）計算：

| 偵測器 | 最大的 FAR×FRR | 出現在哪個門檻 | EER | EER² | C |
|---|---:|---:|---:|---:|---:|
| UCF | 0.003214 | 0.60（FAR 0.0250，FRR 0.1286） | 0.0531 | 0.002815 | 0.003214 |
| Xception | 0.004082 | 0.70（FAR 0.0571，FRR 0.0714） | 0.0624 | 0.003895 | 0.004082 |
| SBI | 0.040319 | 0.20（FAR 0.1946，FRR 0.2071） | 0.2006 | 0.040249 | 0.040319 |

這三個 C 都等於最大乘積，因為 `EER²` 比那個乘積小。EER 用的是 FAR 與 FRR 交點的線性內插，和 `compute_eer()` 相同。

SBI 請用 `bmma_gpt_lookup/threshold_sweep_sbi_faceforensics_v1.csv`。`results/20/sbi/threshold_sweep_sbi_faceforensics.csv` 這次算出的 C 與 v1 相同，仍以 v1 為準。UCF、Xception 用各自 FaceForensics++ 的 threshold sweep，不要用檔名帶 `combined` 的曲線。

這三個 C 只屬於 FaceForensics++。Celeb-DF 的 UCF 大約是 0.087，DFDC 大約是 0.109，不能拿來排 FaceForensics++ 的隊伍。

## C 在 GP 裡做什麼

GP 要選的是每一關的門檻。它先選這一關的 FAR，再用該偵測器的 C 換成 FRR：

```text
這一關的 FRR = 這一關的 C / 這一關的 FAR
```

有了這一步，系統的 FAR 上界和 FRR 上界都可以寫成求解器接受的式子。cvxpy 再呼叫 ECOS，在「系統 FAR 上界 ≤ 指定上限」的前提下，把系統 FRR 上界壓低。

C 愈小，同樣的 FAR 對應到的 FRR 愈低。FaceForensics++ 上 UCF 的 C 比 SBI 小很多，所以在公式裡 UCF 比較容易被選進會影響錯誤率的位置。SBI 又慢，延遲是另一個篩選，不寫在 C 裡面。

GP 不會把 C 再算出一個新的值。三個 C 在開算前就固定了。

## 一次排隊用幾個 C

一次 GP 只處理一種排隊，而且三條曲線都必須是 FaceForensics++。

| 排隊 | 用到的 C | FaceForensics++ 的數值 |
|---|---|---|
| 只用 UCF | 1 個 | 0.003214 |
| UCF → Xception | 2 個 | 0.003214、0.004082 |
| UCF → Xception → SBI | 3 個 | 0.003214、0.004082、0.040319 |

15 種排法就是 1 關 3 種、2 關 6 種、3 關 6 種。FaceForensics++ 因此跑 15 次 GP，共用上面這三個 C，不是每次重新估計 C，也不是 15 個 C。

## GP 算出來之後還要做什麼

求解器交出的是每一關的 FAR 上界、下界，以及整條隊伍的 FAR 上界、FRR 上界。接著做兩件 GP 外面的事：

1. **反查門檻。** 用該偵測器的 FAR 曲線，把 FAR 下界查成下門檻、FAR 上界查成上門檻。最後一關兩個 FAR 相等，所以兩個門檻相等。
2. **用曲線重算真正要寫進查表的 FAR、FRR。** 上界是求解時用的保守數字。查表寫的是門檻套回 FaceForensics++ 曲線後的結果。兩個數字都留：上界說明求解器保證了什麼，重算值才是這批影片上的成績。

期望延遲也在這時候算。某一關偽片會進下一關的比例，大約是 `FAR(上門檻) − FAR(下門檻)`；真片則大約是 `FRR(下門檻) − FRR(上門檻)`。這個減法不能放進 GP。

## 現在直接跑主程式會得到哪一組 C

`build_lookup_table.py` 的 `main()` 仍把三個資料集的曲線逐點取最差再拼起來。那樣估出來的 C 大約是 UCF 0.137、Xception 0.171、SBI 0.324，不是上面 FaceForensics++ 的 0.003、0.004、0.040。

只做 FaceForensics++ 時，`build_detector_profile()` 一次只傳這個資料集的一個 sweep 檔，不要做跨資料集合併。三個 Profile 的 C 應對上表。對不上時，先停止，不要接著解 15 種排列。
