# 最新進度說明

# Project Files
- 所有結果皆在 `results/20/` 中
- 每個permutation 的 CSV 命名為`cascade_threshold_sweep_detectors_dataset.csv`

# Cascade Threshold Sweep CSV 格式
| 欄位名稱 (Field) | 說明 (Description) |
| :--- | :--- |
| **sbi_stage** | sbi 所在的關卡 |
| **ucf_stage** | ucf 所在的關卡 |
| **xception_stage** | xception 所在的關卡 |
| **sbi_upper_threshold** | sbi 的上門檻 |
| **sbi_lower_threshold** | sbi 的下門檻 |
| **ucf_upper_threshold** | ucf 的上門檻 |
| **ucf_lower_threshold** | ucf 的下門檻 |
| **xception_upper_threshold** | xception 的上門檻 |
| **xception_lower_threshold** | xception 的下門檻 |
| **FAR** | 系統 FAR 值 |
| **FRR** | 系統 FRR 值 |

## 備註：
### 關於 Stages
- 第一關 stage = 1，第二關 stage 2
- 沒有使用到該偵測器的話，stage = -1
### 關於 上下門檻
- 如果上下門檻相等的話，進入下一關的機率等於 0。自然不會進入下一關。

# FAR、FRR 計算方法
Assume a cascade has `N` stages. For stage `s`, let:

- `l_s` be the lower threshold of stage `s`
- `u_s` be the upper threshold of stage `s`
- `FAR_s(t)` be the FAR of stage `s` at threshold `t`
- `FRR_s(t)` be the FRR of stage `s` at threshold `t`

Each stage must satisfy:

```math
l_s \le u_s
```

The last stage must force a final decision:

```math
l_N = u_N
```

The total FAR is:

```math
FAR =
\sum_{s=1}^{N}
FAR_s(l_s) \cdot
\prod_{j=1}^{s-1}
\left[
FAR_j(u_j) - FAR_j(l_j)
\right]
```

The total FRR is:

```math
FRR =
\sum_{s=1}^{N}
FRR_s(u_s) \cdot
\prod_{j=1}^{s-1}
\left[
FRR_j(l_j) - FRR_j(u_j)
\right]
```

# 資料筆數
|關卡數|筆數|
|----|----|
|單關|20|
|兩關|4200|
|三關|882000|