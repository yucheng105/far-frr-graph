# FAR-FRR Graph for UCF and Xception

# 最新進度說明
[progress_report.md](progress_report.md)

# 真假分數
本專題設定
- 分數 0 為 **真**
- 分數 1 為 **假**

## Project Files
- 所有結果皆在 `results/20/` 中

# Average Inference Time
## UCF
|Dataset|Time(ms)|
|----           |----   |
|CelebDF        |166.85     |
|DFDC           |170.81     |
|FaceForensics  |224.81     |
|Combined(Micro)       |199.42    |
|Combined(Macro)       |176.84     |

## Xception
|Dataset|Time(ms)|
|----           |----   |
|CelebDF        |158.79     |
|DFDC           |106.22     |
|FaceForensics  |114.70     |
|Combined(Micro)       |124.01    |
|Combined(Macro)       |111.82     |

- **Micro**：每個類別平均時間的平均
- **Macro**：所有測試的平均時間

# FAR、FRR Chart for UCF
## UCF
![UCF](results/plots/1000/ucf/threshold_sweep_ucf_combined.png)
## Xception
![Xception](results/plots/1000/xception/threshold_sweep_xception_combined.png)


# EER (FAR == FRR)
## UCF
|Dataset            |Threshold  |FAR        |FRR        | 
|----               |----       |----       |----       |
|CelebDF            |0.463      |0.294118   |0.297753   |
|DFDC               |0.509      |0.329427   |0.330022   |
|FaceForensics      |0.79       |0.051786   |0.05       |
|Combined           |0.530     |0.298875    |0.298519   |
## Xception
|Dataset            |Threshold  |FAR        |FRR        | 
|----               |----       |----       |----       |
|CelebDF            |0.541      |0.273529   |0.275281   |
|DFDC               |0.483      |0.352867   |0.349892   |
|FaceForensics      |0.707      |0.062500   |0.057143   |
|Combined           |0.496      |0.313165   |0.31523    |


# EER 點
[results/min_eer_entries](results/min_eer_entries.csv)


# Exhaustive Search for Accuracy-Oriented and Latency-Oriented Selection
| 分類 | 選取標準 |
| :--- | :--- |
| **Accuracy-Oriented** | FAR Limit 之下，FRR 最低的 Permutation 與其 Threshold 組合 |
| **Latency-Oriented** | FAR Limit 之下，推論時間最短的 Permutation 與其 Threshold 組合 |

## 流程
![Flow](./img/FAR-FRR_tradeoff.drawio.png)