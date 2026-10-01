# Parallel Fusion Exhaustive Search

This program performs the same weight-and-threshold optimization as the MILP,
but only on a fixed grid.

For `step = 0.05`, the interval count is 20 and the inclusive values are:

```text
0.00, 0.05, ..., 0.95, 1.00
```

That gives 21 grid values, 231 valid alpha triples, and 4,851
alpha/threshold combinations per dataset.

The decision rule and objective are:

```text
q = alpha_X * Xception + alpha_U * UCF + alpha_S * SBI
prediction = 1 if q >= beta else 0
objective = FAR + FRR
```

Run all datasets from the project root:

```powershell
python parallel-fusion/exhaustive/run_exhaustive.py
```

Run one dataset or change the grid step:

```powershell
python parallel-fusion/exhaustive/run_exhaustive.py --dataset DFDC --step 0.05
```

Results are written to `parallel-fusion/exhaustive/results/`:

- `<dataset>_exhaustive_grid.csv`: all 4,851 evaluated combinations.
- `<dataset>_best_predictions.csv`: per-sample predictions for the best point.
- `exhaustive_best_summary.csv`: the best point for every selected dataset.

If several rows have the same minimum `FAR+FRR`, the summary prefers the row
with the smallest `abs(FAR-FRR)` and then uses parameter order for a stable,
deterministic choice. The complete grid CSV retains every tied optimum.
