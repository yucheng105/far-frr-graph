# FaceForensics++：FAR×FRR 的緊密保守上界 C

這一段只做 FaceForensics++（140 支真片、560 支偽片）。每個偵測器先有一條曲線，再從曲線算出一個 C。C 是 GP 的輸入，不是這段要交的查表。

圖已經產生在 `bmma_gpt_lookup/output/ffpp_conb/`。重畫的方式在最後一節。

## 投影片 1　這段在做什麼

標題：FaceForensics++ 上，每個偵測器的 FAR×FRR 上界 C

可以念的文字：

我們先不排隊伍。FaceForensics++ 上，UCF、Xception、SBI 各有一張門檻表。每一格是一個門檻下的 FAR 和 FRR。這張表叫做這條曲線。

我們要從每條曲線取出一個常數 C，讓所有門檻都滿足

```text
FAR × FRR ≤ C
```

也就是

```text
FRR ≤ C / FAR
```

GP 之後會用第二式，把「選一個 FAR」換成「對應的 FRR」。這段先把三個 C 算出來，並用圖說明它是上界、而且已經不能再調小。

## 投影片 2　曲線是哪來的

標題：曲線是門檻表，不是重新跑出來的分數

資料來源：

| 偵測器 | 檔案 |
|---|---|
| UCF | `exhaustive-search/threshold-sweep/results/20/ucf/threshold_sweep_ucf_faceforensics.csv` |
| Xception | `exhaustive-search/threshold-sweep/results/20/xception/threshold_sweep_xception_faceforensics.csv` |
| SBI | `bmma_gpt_lookup/threshold_sweep_sbi_faceforensics_v1.csv` |

這些檔是 `threshold_sweep.py` 把每支影片的 `fake_score` 數出來的。分數愈高愈像偽片。對每個門檻：

- FAR = 偽片分數低於門檻的支數 / 560
- FRR = 真片分數達到門檻的支數 / 140

舉 UCF、門檻 0.20：FAR 是 0.003571，約 2/560；FRR 是 0.542857，約 76/140。門檻拉高，FAR 變大、FRR 變小。GP 讀的是整張表。

不要用檔名有 `combined` 的曲線。那是三個資料集拼過的結果。

## 投影片 3　什麼叫做緊密、什麼叫做保守

標題：C 是貼住最高點的那條上界

保守：C 取成所有門檻裡最大的 `FAR × FRR`（若 `EER²` 更大才改用 `EER²`）。所以沒有任何一格的乘積超過 C。用 `FRR = C / FAR` 估計這一關的 FRR 時，不會比曲線上的實測更樂觀。

緊密：因為 C 就是那個最大乘積，至少有一個門檻的乘積等於 C。C 再調小，那一格就會破界。所以在「單一常數上界」這個形式裡，它是最小的一條。

它不是說每個門檻的乘積都等於 C。UCF 的乘積大約落在 0.001 到 0.0032，只有門檻 0.60 碰到 0.003214，其他門檻都在線下面。

EER 是 FAR 與 FRR 曲線的交點。三個偵測器的 `EER²` 都小於最大乘積，所以這次 C 全部等於最大乘積。算法和 `build_lookup_table.py` 的 `compute_conb()` 相同。

## 投影片 4　C 的做法

標題：一條曲線算出一個 C

對 FaceForensics++ 的一張門檻表：

1. 每個門檻計算 `FAR × FRR`。門檻 1.00 若 FRR 為 0，乘積是 0，仍留在表上，但不拿來當 C。
2. 取出這些乘積的最大值，並記下那個門檻。
3. 找 FAR 與 FRR 的交點，得到 EER，計算 `EER²`。交點落在兩格中間時做線性內插，和 `compute_eer()` 一樣。
4. C = `max(最大乘積, EER²)`。

FaceForensics++ 的結果：

| 偵測器 | 最大乘積出現的門檻 | 該點 FAR | 該點 FRR | EER | EER² | C |
|---|---:|---:|---:|---:|---:|---:|
| UCF | 0.60 | 0.0250（14/560） | 0.1286（18/140） | 0.0531 | 0.002815 | 0.003214 |
| Xception | 0.70 | 0.0571（32/560） | 0.0714（10/140） | 0.0624 | 0.003895 | 0.004082 |
| SBI | 0.20 | 0.1946（109/560） | 0.2071（29/140） | 0.2006 | 0.040249 | 0.040319 |

三個 C 都來自最大乘積。UCF 最小，SBI 大約是 UCF 的 12 倍。GP 用 `FRR = C / FAR` 時，同樣的 FAR，SBI 對應的 FRR 上界會高很多。

這三個數只屬於 FaceForensics++。Celeb-DF、DFDC 要各算各的，不能把這三個 C 拿去排別的資料集。

## 投影片 5　圖：乘積與水平線

放這張圖：`output/ffpp_conb/product_vs_threshold.png`

圖說：

橫軸是門檻，縱軸是該門檻的 `FAR × FRR`。虛線是 C。紅點是乘積最大的那一格，它貼在虛線上。其餘的點都在虛線下面，所以這條線是上界；因為紅點碰到線，C 不能再降低。

口頭可以補：UCF 的紅點在門檻 0.60，Xception 在 0.70，SBI 在 0.20。左右兩端乘積掉下去，是因為其中一個錯誤率接近 0，不代表那個門檻比較好。

## 投影片 6　圖：雙曲線

放這張圖：`output/ffpp_conb/far_frr_bound.png`

圖說：

橫軸 FAR、縱軸 FRR，每個點是一個門檻。虛線是 `FRR = C / FAR`。GP 用的就是這條虛線，不是上一張圖的水平線。

實測點落在虛線上或虛線下方。紅點是乘積最大的門檻，它貼住虛線。這表示用 C 換算 FRR 時，得到的是上界：真實 FRR 不會比 `C / FAR` 更高。

SBI 的點在高 FAR 那邊離虛線較遠，代表那裡的乘積比 C 小，上界比較鬆。鬆緊不影響「它仍是上界」，只影響 GP 用這條公式時會不會偏保守。

## 投影片 7　這段先停在三個 C

標題：GP 還沒開始排隊伍

可以念的文字：

FaceForensics++ 現在有三個輸入：UCF 的 C 是 0.003214，Xception 是 0.004082，SBI 是 0.040319。

下一階段才做 GP。只用 UCF 時，GP 用 1 個 C。兩個偵測器排隊用 2 個。三個都排用 3 個。15 種排法共用這三個數，不會每次重算 C。GP 解出的門檻、系統 FAR、系統 FRR 不在這一段。

## 圖怎麼生成

在 `bmma_gpt_lookup` 目錄執行：

```powershell
python plot_ffpp_conb.py
```

腳本只讀上面三個 CSV，不跑偵測器，也不跑 GP。它會：

1. 讀每個門檻的 FAR、FRR。
2. 用與 `compute_eer()`、`compute_conb()` 相同的規則計算 EER 和 C。
3. 写出三個檔案：
   - `output/ffpp_conb/ffpp_conb_table.csv`：投影片 4 的數字
   - `output/ffpp_conb/product_vs_threshold.png`：投影片 5
   - `output/ffpp_conb/far_frr_bound.png`：投影片 6

曲線檔若更新，重新執行這支程式，投影片上的 C 改成新的 `ffpp_conb_table.csv`，圖直接換檔。不要手改圖上的數字。
