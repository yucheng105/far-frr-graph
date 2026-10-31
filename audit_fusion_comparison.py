"""Re-evaluate reported cascade configurations on aligned per-sample scores.

Writes a separate audit; never overwrites existing experiment results.
Rates use the cascade convention: FAR=fake accepted as real, FRR=real rejected.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DETECTORS = ('sbi', 'ucf', 'xception')


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def evaluate(samples, config):
    order = sorted((d for d in DETECTORS if int(config[d + '_stage']) > 0),
                   key=lambda d: int(config[d + '_stage']))
    errors = [0, 0]
    counts = [0, 0]
    for sample in samples:
        label = int(sample['label(ground_truth)'])
        counts[label] += 1
        for d in order:
            score = float(sample[d + '_fake_score'])
            lower = float(config[d + '_lower_threshold'])
            upper = float(config[d + '_upper_threshold'])
            if score < lower:
                prediction = 0
                break
            if score >= upper:
                prediction = 1
                break
        else:
            raise ValueError('Cascade left a sample undecided')
        errors[label] += prediction != label
    return dict(n_real=counts[0], n_fake=counts[1],
                fake_accepted_as_real=errors[1], real_rejected_as_fake=errors[0],
                empirical_far=errors[1] / counts[1],
                empirical_frr=errors[0] / counts[0],
                empirical_sum=errors[1] / counts[1] + errors[0] / counts[0])


def main():
    results = []
    for short, name in [('faceforensics', 'FaceForensics++'),
                        ('celebdf', 'Celeb-DF-v2'), ('dfdc', 'DFDC')]:
        samples = read(ROOT / 'parallel-fusion' / (name + '_fusion.csv'))
        path = ROOT / 'exhaustive-search/accuracy-oriented-selection/results/20' / (
            'accuracy_oriented_selection_' + short + '.csv')
        configs = read(path)
        best = min(configs, key=lambda r: float(r['FAR']) + float(r['FRR']))
        empirical = evaluate(samples, best)
        reported_sum = {'faceforensics': .029498, 'celebdf': .317119, 'dfdc': .507143}[short]
        reported = next(c for c in configs if abs(float(c['FAR']) + float(c['FRR']) - reported_sum) < 1e-9)
        # Check all selected frontier configurations too. This is not a fresh
        # exhaustive search over the empirical objective.
        frontier_best = min((evaluate(samples, c)['empirical_sum'] for c in configs))
        prediction_path = ROOT / 'parallel-fusion/milp/milp-results' / (
            name + '_milp_predictions.csv')
        predictions = read(prediction_path)
        expected = [(s['sample_id'], s['label(ground_truth)']) for s in samples]
        actual = [(s['sample_id'], s['label(ground_truth)']) for s in predictions]
        assert expected == actual, 'MILP sample alignment differs'
        milp_errors = [0, 0]
        for p in predictions:
            y = int(p['label(ground_truth)'])
            milp_errors[y] += int(p['prediction']) != y
        milp_sum = milp_errors[0] / empirical['n_real'] + milp_errors[1] / empirical['n_fake']
        results.append(dict(dataset=name, selected_configuration=best,
                            user_reported_configuration=reported,
                            user_reported_estimated_sum=reported_sum,
                            user_reported_configuration_empirical=evaluate(samples, reported),
                            estimated_sum=float(best['FAR']) + float(best['FRR']),
                            **empirical, best_empirical_sum_on_selected_frontier=frontier_best,
                            saved_milp_prediction_sum=milp_sum))
    output = ROOT / 'fusion_comparison_audit.json'
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
