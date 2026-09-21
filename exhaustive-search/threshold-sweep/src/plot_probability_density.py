from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SPLITS = ["20", "1000"]
DATASET_SLUGS = {
    "CelebDF": "celebdf",
    "DFDC": "dfdc",
    "FaceForensics": "faceforensics",
    "Combined": "combined",
}
# SBI is only evaluated on CelebDF and DFDC, so FF++ and the combined sweep
# are intentionally left out of its chart list.
MODEL_DATASETS = {
    "ucf": ["CelebDF", "DFDC", "FaceForensics", "Combined"],
    "xception": ["CelebDF", "DFDC", "FaceForensics", "Combined"],
    "sbi": ["CelebDF", "DFDC"],
}


def read_threshold_sweep(csv_path):
    thresholds = []
    fars = []
    frrs = []

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
            thresholds.append(float(row["Threshold"]))
            fars.append(float(row["FAR"]))
            frrs.append(float(row["FRR"]))

    if not thresholds:
        raise ValueError(f"{csv_path} does not contain any threshold rows")

    return thresholds, fars, frrs


def sweep_csv_path(split_name, model_name, dataset_name):
    filename = f"threshold_sweep_{model_name}_{DATASET_SLUGS[dataset_name]}.csv"
    return RESULTS_DIR / split_name / model_name / filename


def setup_axes(title):
    plt.figure(figsize=(9, 5.5))
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Probability")
    plt.xlim(0, )
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)


def save_current_figure(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")


def plot_single_chart(dataset_name, split_name, model_name, csv_path):
    thresholds, fars, frrs = read_threshold_sweep(csv_path)

    setup_axes(f"FAR / FRR Threshold Sweep - {dataset_name} ({split_name} splits)")
    plt.plot(thresholds, fars, label="FAR", linewidth=2.2, color="#d62728")
    plt.plot(thresholds, frrs, label="FRR", linewidth=2.2, color="#1f77b4")
    plt.legend()

    output_path = (
        PLOTS_DIR
        / split_name
        / model_name
        / csv_path.with_suffix(".png").name
    )
    save_current_figure(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot FAR/FRR threshold sweeps from results/<split>/<model>/."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_DATASETS),
        default=sorted(MODEL_DATASETS),
        help="Models to plot (default: all).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=SPLITS,
        help="Threshold-grid sizes to plot (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for split_name in args.splits:
        for model_name in args.models:
            for dataset_name in MODEL_DATASETS[model_name]:
                csv_path = sweep_csv_path(split_name, model_name, dataset_name)
                if not csv_path.exists():
                    print(f"Skipping missing sweep: {csv_path}")
                    continue

                plot_single_chart(dataset_name, split_name, model_name, csv_path)


if __name__ == "__main__":
    main()
