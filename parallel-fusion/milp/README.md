# Parallel Fusion MILP

This directory contains the merged detector scores and a mixed-integer linear
program that optimizes each dataset independently.

## Model

The conceptual formulation uses the three detector weights, the overall
threshold, and one binary prediction per sample. The implementation uses an
equivalent, tighter binary error indicator per sample: `z_j = 1` exactly when
sample `j` is misclassified. This reduces two Big-M rows per sample to one and
makes the full DFDC model practical for HiGHS. Predictions are recovered from
the optimized threshold and verified against every error indicator. The
detector weights are constrained to sum to one. The objective is:

```text
minimize FAR + FRR
```

`q_j` is substituted directly into the Big-M constraints, so it does not need
to be stored as a separate solver variable. `M = 1` is valid because all input
scores, weights, and the threshold are in `[0, 1]`.

The upper Big-M constraint uses a small configurable `epsilon`. Therefore:

```text
q_j >= beta  -> Fake (1)
q_j <  beta  -> Real (0)
```

## Run

Install the dependencies if necessary:

```powershell
python -m pip install -r parallel-fusion/requirements.txt
```

Run all datasets from the project root:

```powershell
python parallel-fusion/run_milp.py
```

Run only one dataset:

```powershell
python parallel-fusion/run_milp.py --dataset DFDC
```

Useful optional arguments:

```text
--epsilon 1e-6
--time-limit SECONDS
--mip-rel-gap VALUE
--solver-log
```

With no time limit, HiGHS continues until it proves global optimality. Minimum
classification-error MILPs can require substantial branch-and-bound time. If a
time limit is supplied, the CSV still records the best feasible solution, but
`optimal` is `False` and `mip_gap` reports the remaining proof gap; the program
does not describe that solution as optimal.

Results are written under `parallel-fusion/milp-results/`:

- `milp_summary.csv`: weights, threshold, FAR, FRR, and solver status.
- `<dataset>_milp_predictions.csv`: per-sample fusion score, prediction, and
  error type.

## Rebuild merged inputs

```powershell
python parallel-fusion/merge_scores.py
```
