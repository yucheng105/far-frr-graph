#!/usr/bin/env python3
"""Evaluate fixed-threshold mean-score and majority-vote baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "exhaustive"))
from exhaustive_search import read_fusion_csv

DATASETS = ("Celeb-DF-v2", "DFDC", "FaceForensics++")
DETECTORS = ("xception", "ucf", "sbi")


def evaluate(scores, labels):
    """Return both predictions; equality at 0.5 counts as Fake."""
    votes = (scores >= 0.5).astype(np.int8)
    means = scores.mean(axis=1)
    return means, votes, {
        "score_average": (means >= 0.5).astype(np.int8),
        "majority_vote": (votes.sum(axis=1) >= 2).astype(np.int8),
    }


def metrics(labels, predictions):
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    if not n_real or not n_fake:
        raise ValueError("Both Real and Fake samples are required")
    false_accepts = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    false_rejections = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    far = false_accepts / n_fake
    frr = false_rejections / n_real
    return dict(n_real=n_real, n_fake=n_fake, false_accepts=false_accepts,
                false_rejections=false_rejections, far=far, frr=frr,
                far_plus_frr=far + frr,
                accuracy=float(np.mean(labels == predictions)))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for dataset in DATASETS if args.dataset == "all" else (args.dataset,):
        data = read_fusion_csv(args.input_dir / f"{dataset}_fusion.csv")
        if data.dataset != dataset:
            raise ValueError(f"Expected dataset {dataset}, got {data.dataset}")
        means, votes, methods = evaluate(data.scores, data.labels)
        for method, predictions in methods.items():
            rates = metrics(data.labels, predictions)
            summary.append(dict(dataset=dataset, method=method,
                                alpha_xception=1 / 3 if method == "score_average" else "",
                                alpha_ucf=1 / 3 if method == "score_average" else "",
                                alpha_sbi=1 / 3 if method == "score_average" else "",
                                score_threshold=0.5,
                                minimum_fake_votes=2 if method == "majority_vote" else "",
                                **rates))
            rows = []
            for i, prediction in enumerate(predictions):
                label = int(data.labels[i])
                error = "correct" if prediction == label else (
                    "false_accept" if label == 1 else "false_rejection")
                rows.append(dict(dataset=dataset, sample_index=i + 1,
                                 sample_id=data.sample_ids[i],
                                 **{"label(ground_truth)": label},
                                 **{f"{name}_fake_score": data.scores[i, j]
                                    for j, name in enumerate(DETECTORS)},
                                 **{f"{name}_vote": int(votes[i, j])
                                    for j, name in enumerate(DETECTORS)},
                                 mean_fake_score=means[i],
                                 fake_vote_count=int(votes[i].sum()),
                                 prediction=int(prediction), error_type=error))
            write_csv(args.output_dir / f"{dataset}_{method}_predictions.csv", rows)
            print(f"{dataset} / {method}: FAR={rates['far']:.6f}, "
                  f"FRR={rates['frr']:.6f}, Accuracy={rates['accuracy']:.6f}")
    write_csv(args.output_dir / "baseline_summary.csv", summary)


if __name__ == "__main__":
    main()
