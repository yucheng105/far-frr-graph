from itertools import permutations
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


def get_detector_permutations():
    detector_permutations = []

    for length in range(1, len(DETECTORS) + 1):
        detector_permutations.extend(permutations(DETECTORS, length))

    return detector_permutations


def write_empty_csv(csv_path):
    if csv_path.exists():
        print(f"Skipped existing file: {csv_path}")
        return

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

    print(f"Wrote {csv_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for detector_permutation in get_detector_permutations():
        permutation_name = "_".join(detector_permutation)
        permutation_dir = RESULTS_DIR / permutation_name
        permutation_dir.mkdir(parents=True, exist_ok=True)

        for dataset in DATASETS:
            csv_path = (
                permutation_dir
                / f"cascade_threshold_sweep_{permutation_name}_{dataset}.csv"
            )
            write_empty_csv(csv_path)


if __name__ == "__main__":
    main()
