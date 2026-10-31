# Parallel Fusion Baselines

Run from the project root:

```powershell
python parallel-fusion/baseline/run_baseline.py
python parallel-fusion/baseline/run_baseline.py --dataset DFDC
```

Requires NumPy (also used by the existing exhaustive-search implementation).
Inputs default to the merged `parallel-fusion/<dataset>_fusion.csv` files.
Use `--input-dir` and `--output-dir` to override the directories.

Both baselines use fixed thresholds without fitting on the evaluation labels:

- `score_average`: arithmetic mean of Xception, UCF, and SBI fake scores,
  using exact equal weights (1/3). Predict Fake when the mean is >= 0.5.
- `majority_vote`: each detector votes Fake when its fake score is >= 0.5;
  predict Fake with at least two Fake votes. Three detectors cannot tie.

Labels are Real=0 and Fake=1. Metrics follow the project's cascade convention:

- FAR = Fake predicted Real / number of Fake samples.
- FRR = Real predicted Fake / number of Real samples.
- Accuracy = correct predictions / all samples.

**Comparison note:** the current `exhaustive` and `milp` implementations use
the opposite FAR/FRR naming convention. Swap their FAR and FRR columns when
comparing individual rates with these baselines; FAR+FRR is unaffected.
Those implementations are not modified by this baseline runner.

`results/baseline_summary.csv` contains both methods for each selected dataset,
including thresholds, class counts, error counts, FAR, FRR, FAR+FRR, and accuracy.
Each `<dataset>_<method>_predictions.csv` records the source scores, individual
votes, mean score, vote count, prediction, and error type. Repeated sample IDs
remain separate rows identified by their one-based sample index.
The summary is replaced on each run and contains only the selected datasets.
