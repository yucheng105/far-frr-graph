# Latency-oriented global selection

從 `exhaustive-search/far-frr-tradeoff/results/20/` 中，對每個 dataset 與
`FAR Limit` 比較所有 detector permutation，只保留總推論時間最短的
detector + threshold 組合。

## Latency 定義

依照本階段假設，permutation 的 latency 是所有已使用 detector 的平均推論時間總和：

```text
Permutation latency = sum(each used detector's average inference time)
```

推論時間讀取自專案根目錄的：

- `sbi_average_inference_time.csv`
- `ucf_average_inference_time.csv`
- `xception_average_inference_time.csv`

`combined` 預設使用 `Combined(Macro)`，因為該值是所有測試樣本的平均時間。

## 選擇規則

1. 候選列必須滿足 `FAR <= FAR Limit`。
2. 優先選擇 permutation latency 最低的候選列。
3. 若 latency 相同，優先選擇 FRR 較低的候選列。
4. 若 FRR 仍相同，優先選擇實際 FAR 較低的候選列。
5. 其餘完全相同的情況以來源檔案路徑字典順序決定，使結果可重現。

輸出保留 FAR–FRR trade-off 的所有欄位，並在 `FAR Limit` 前新增
`Inference Time (ms)`。

## 執行

```powershell
python exhaustive-search/latency-oriented-selection/src/select_lowest_latency.py
```

若 combined 要改用 Micro：

```powershell
python exhaustive-search/latency-oriented-selection/src/select_lowest_latency.py --combined-average micro
```

結果會寫入 `exhaustive-search/latency-oriented-selection/results/20/`。
