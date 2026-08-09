"""Calculate cascade FAR/FRR for every detector permutation."""
from itertools import product
from pathlib import Path
import csv


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "20"

DETECTORS = ["sbi", "ucf", "xception"]
DATASETS = ["celebdf", "dfdc", "faceforensics", "combined"]

FIELDNAMES = [
    "sbi_stage",
    "ucf_stage",
    "xception_stage",
    "sbi_upper_threshold",
    "sbi_lower_threshold",
    "ucf_upper_threshold",
    "ucf_lower_threshold",
    "xception_upper_threshold",
    "xception_lower_threshold",
    "FAR",
    "FRR",
]


def format_number(value):
    if value == -1:
        return -1

    return round(value, 6)


def read_threshold_sweep(detector, dataset):
    csv_path = RESULTS_DIR / detector / f"threshold_sweep_{detector}_{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Threshold sweep CSV not found: {csv_path}")

    rates_by_threshold = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"Threshold", "FAR", "FRR"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            threshold = float(row["Threshold"])
            rates_by_threshold[threshold] = {
                "FAR": float(row["FAR"]),
                "FRR": float(row["FRR"]),
            }

    if not rates_by_threshold:
        raise ValueError(f"{csv_path} does not contain any threshold rows")

    return rates_by_threshold


def get_permutation_dirs():
    permutation_dirs = []

    for path in RESULTS_DIR.iterdir():
        if not path.is_dir():
            continue

        detectors = path.name.split("_")
        if all(detector in DETECTORS for detector in detectors):
            permutation_dirs.append(path)

    return sorted(permutation_dirs, key=lambda path: (len(path.name.split("_")), path.name))


def get_threshold_pairs(thresholds, is_last_stage):
    if is_last_stage:
        return [(threshold, threshold) for threshold in thresholds]

    return [
        (lower_threshold, upper_threshold)
        for lower_threshold in thresholds
        for upper_threshold in thresholds
        if lower_threshold <= upper_threshold
    ]


def calculate_cascade_far_frr(detector_permutation, threshold_by_detector, sweeps):
    total_far = 0
    total_frr = 0
    previous_far_middle_probability = 1
    previous_frr_middle_probability = 1

    for detector in detector_permutation:
        lower_threshold, upper_threshold = threshold_by_detector[detector]
        sweep = sweeps[detector]

        far_at_lower = sweep[lower_threshold]["FAR"]
        far_at_upper = sweep[upper_threshold]["FAR"]
        frr_at_lower = sweep[lower_threshold]["FRR"]
        frr_at_upper = sweep[upper_threshold]["FRR"]

        total_far += far_at_lower * previous_far_middle_probability
        total_frr += frr_at_upper * previous_frr_middle_probability

        previous_far_middle_probability *= far_at_upper - far_at_lower
        previous_frr_middle_probability *= frr_at_lower - frr_at_upper

    return total_far, total_frr


def build_output_row(detector_permutation, threshold_by_detector, far, frr):
    row = {}

    for detector in DETECTORS:
        if detector in detector_permutation:
            row[f"{detector}_stage"] = detector_permutation.index(detector) + 1
            lower_threshold, upper_threshold = threshold_by_detector[detector]
            row[f"{detector}_upper_threshold"] = format_number(upper_threshold)
            row[f"{detector}_lower_threshold"] = format_number(lower_threshold)
        else:
            row[f"{detector}_stage"] = -1
            row[f"{detector}_upper_threshold"] = -1
            row[f"{detector}_lower_threshold"] = -1

    row["FAR"] = format_number(far)
    row["FRR"] = format_number(frr)

    return row


def write_permutation_dataset_csv(permutation_dir, dataset):
    detector_permutation = permutation_dir.name.split("_")
    sweeps = {
        detector: read_threshold_sweep(detector, dataset)
        for detector in detector_permutation
    }

    threshold_options = []
    for index, detector in enumerate(detector_permutation):
        thresholds = sorted(sweeps[detector])
        is_last_stage = index == len(detector_permutation) - 1
        threshold_options.append(get_threshold_pairs(thresholds, is_last_stage))

    output_path = (
        permutation_dir
        / f"cascade_threshold_sweep_{permutation_dir.name}_{dataset}.csv"
    )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for threshold_combination in product(*threshold_options):
            threshold_by_detector = dict(zip(detector_permutation, threshold_combination))
            far, frr = calculate_cascade_far_frr(
                detector_permutation,
                threshold_by_detector,
                sweeps,
            )
            writer.writerow(
                build_output_row(
                    detector_permutation,
                    threshold_by_detector,
                    far,
                    frr,
                )
            )

    print(f"Wrote {output_path}")


def main():
    permutation_dirs = get_permutation_dirs()
    if not permutation_dirs:
        raise FileNotFoundError(f"No permutation directories found in {RESULTS_DIR}")

    for permutation_dir in permutation_dirs:
        for dataset in DATASETS:
            write_permutation_dataset_csv(permutation_dir, dataset)


if __name__ == "__main__":
    main()
