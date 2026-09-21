# FAR–FRR trade-off

從 `threshold-sweep/results/20/` 的每個 cascade threshold sweep CSV 中，找出各個
`FAR Limit` 下 FRR 最低的組合。

## 選擇規則

- `FAR Limit` 預設從 `0.000` 到 `1.000`，間隔為 `0.005`。
- 候選列必須嚴格滿足 `FAR <= FAR Limit`。
- 在所有合格候選列中，只保留 FRR 最低的一列。
- 如果多列的 FRR 相同，優先保留實際 FAR 較低的一列；若仍相同，保留原始 CSV 中較早出現的一列。
- 若某個 `FAR Limit` 下沒有任何合格候選列，該 limit 不會輸出。

輸出沿用原始 CSV 欄位，並在 `FAR` 前新增 `FAR Limit` 欄位。目錄結構會對應
原始的 permutation 目錄，輸出檔名為
`far_frr_tradeoff_<permutation>_<dataset>.csv`。

## 執行

```powershell
python far-frr-tradeoff/src/calculate_tradeoff.py
```

要改變間隔時：

```powershell
python far-frr-tradeoff/src/calculate_tradeoff.py --far-step 0.01
```
