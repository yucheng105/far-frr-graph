# Parallel Fusion MILP

This directory contains the merged detector scores and a mixed-integer linear
program that optimizes each dataset independently.

## Model

The formulation uses the three detector weights, the overall threshold, and
one binary prediction `e_j` per sample. The detector weights are constrained
to sum to one. The objective is:

```text
minimize FAR + FRR
```

`q_j` is substituted directly into the Big-M constraints, so it does not need
to be stored as a separate solver variable. `M = 1` is valid because all input
scores, weights, and the threshold are in `[0, 1]`.

Each sample uses both strict Big-M indicator rows, with `M = 1` and a small
configurable `epsilon` (default `1e-6`):

```text
q_j - beta >= -M(1 - e_j)
q_j - beta <= -epsilon + (M + epsilon)e_j
```

Therefore there is no ambiguous band in which the solver can choose whichever
label improves the objective:

```text
q_j >= beta  -> Fake (1)
q_j <= beta - epsilon -> Real (0)
beta - epsilon < q_j < beta -> infeasible
```

The epsilon-sized infeasible interval is the finite-precision replacement for
the strict inequality `q_j < beta`; it is not a region where either label is
allowed.

## Run

Install the dependencies if necessary:

```powershell
python -m pip install -r parallel-fusion/milp/requirements.txt
```

Run all datasets from the project root:

```powershell
python parallel-fusion/milp/run_milp.py
```

Run only one dataset:

```powershell
python parallel-fusion/milp/run_milp.py --dataset DFDC
```

Useful optional arguments:

```text
--epsilon 1e-6
--time-limit SECONDS
--mip-rel-gap VALUE
--solver-log
```

## Start from an exhaustive-search incumbent

SciPy's `milp` interface does not expose a native `x0`/MIP-start argument. The
runner can still validate the best grid point and add its objective as a hard
incumbent cutoff, so branch-and-bound searches only solutions that are at least
as good as the grid result:

```powershell
python parallel-fusion/milp/run_milp.py `
  --grid-start parallel-fusion/exhaustive/results/exhaustive_best_summary.csv `
  --time-limit 300
```

The alpha values and beta are re-evaluated from the fusion input CSV; the FAR
and FRR written in the supplied summary are not trusted. If the grid point is
not feasible under the strict epsilon margin, the run stops with an error. If
HiGHS does not find a new incumbent within the time limit, the validated grid
solution is retained in the output with `solution_source=grid_seed`.

With no time limit, HiGHS continues until it proves global optimality. Minimum
classification-error MILPs can require substantial branch-and-bound time. If a
time limit is supplied, the CSV still records the best feasible solution, but
`optimal` is `False` and `mip_gap` reports the remaining proof gap; the program
does not describe that solution as optimal.

Results are written under `parallel-fusion/milp/milp-results/`:

- `milp_summary.csv`: weights, threshold, FAR, FRR, and solver status.
- `<dataset>_milp_predictions.csv`: per-sample fusion score, prediction, and
  error type.

The optimizer diagnostics are printed to the terminal and stored in
`milp_summary.csv` as `best_bound`, `mip_gap`, `node_count`, and
`termination_reason`.

## Rebuild merged inputs

```powershell
python parallel-fusion/merge_scores.py
```
