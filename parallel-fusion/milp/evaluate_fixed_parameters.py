"""Evaluate the user-supplied, rounded FF++ parameters without optimization.

Uses the MILP convention: FAR = real predicted fake / real count;
FRR = fake predicted real / fake count. No inference latency is measured.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = (0.5487, 0.4401, 0.0112)
BETA = 0.7368
SCORES = ('xception_fake_score', 'ucf_fake_score', 'sbi_fake_score')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = ROOT / 'fixed-ffpp-results'
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    # FF++ is also evaluated to check the rounded source parameters.
    for dataset in ('FaceForensics++', 'Celeb-DF-v2', 'DFDC'):
        with (ROOT.parent / f'{dataset}_fusion.csv').open(
            encoding='utf-8-sig', newline=''
        ) as stream:
            samples = list(csv.DictReader(stream))
        counts, errors = [0, 0], [0, 0]
        predictions = []
        for sample in samples:
            if sample['dataset'] != dataset:
                raise ValueError('Unexpected dataset')
            label = int(sample['label(ground_truth)'])
            if label not in (0, 1):
                raise ValueError('Invalid label')
            scores = [float(sample[column]) for column in SCORES]
            if not all(0 <= score <= 1 for score in scores):
                raise ValueError('Invalid score')
            score = sum(w * s for w, s in zip(WEIGHTS, scores))
            prediction = int(score >= BETA)
            counts[label] += 1
            errors[label] += prediction != label
            predictions.append(dict(
                **sample, fusion_score=score, prediction=prediction,
                error_type=('correct' if prediction == label else
                            'false_accept' if label == 0 else 'false_reject')))
        far, frr = errors[0] / counts[0], errors[1] / counts[1]
        summary.append(dict(
            dataset=dataset, alpha_xception=WEIGHTS[0], alpha_ucf=WEIGHTS[1],
            alpha_sbi=WEIGHTS[2], beta=BETA, far=far, frr=frr,
            far_plus_frr=far + frr, false_accepts=errors[0],
            false_rejections=errors[1], n_real=counts[0], n_fake=counts[1]))
        write_csv(output / f'{dataset}_fixed_predictions.csv', predictions)
        print(f'{dataset}: FAR={far:.10f} ({errors[0]}/{counts[0]}), '
              f'FRR={frr:.10f} ({errors[1]}/{counts[1]}), sum={far + frr:.10f}')
    write_csv(output / 'fixed_parameters_summary.csv', summary)


if __name__ == '__main__':
    main()
