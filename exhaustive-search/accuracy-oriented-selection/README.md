# Accuracy-oriented global selection

從 `exhaustive-search/far-frr-tradeoff/results/20/` 中，對每個 dataset 與
`FAR Limit` 比較所有 detector permutation，只保留 FRR 最低的 detector
+ threshold 組合。

## 選擇規則

1. 候選列必須滿足 `FAR <= FAR Limit`。
2. 跨所有 detector permutation 選擇 FRR 最低的候選列。
3. 若 FRR 相同，優先選擇實際 FAR 較低的候選列。
4. 若 FRR 與 FAR 都相同，以來源檔案路徑字典順序決定，使結果可重現。

輸出表頭與 FAR–FRR trade-off 結果完全相同，不新增欄位。Detector
組合可由 `sbi_stage`、`ucf_stage` 與 `xception_stage` 識別。

## 執行

```powershell
python exhaustive-search/accuracy-oriented-selection/src/select_global_best.py
```

結果會寫入 `exhaustive-search/accuracy-oriented-selection/results/20/`。
