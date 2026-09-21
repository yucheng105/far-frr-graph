from pathlib import Path
import csv


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "input-csvs" / "xception"
OUTPUT_FILE = ROOT_DIR / "xception_average_inference_time.csv"
TIME_COLUMN = "inference_time_ms"

DATASET_FILES = [
    ("CelebDF", INPUT_DIR / "scores_visual_xception_ft_celebdf_v0.csv"),
    ("DFDC", INPUT_DIR / "scores_visual_xception_dfdc_v0.csv"),
    ("FaceForensics", INPUT_DIR / "scores_visual_xception_ft_v0.csv"),
]


def read_valid_rows(csv_path):
    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"dataset", TIME_COLUMN}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            if row.get("status", "ok").strip().lower() != "ok":
                continue

            try:
                inference_time = float(row[TIME_COLUMN])
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "category": row["dataset"].strip(),
                    "inference_time_ms": inference_time,
                }
            )

    if not rows:
        raise ValueError(f"{csv_path} does not contain any valid inference time rows")

    return rows


def average(values):
    values = list(values)
    return sum(values) / len(values)


def round_time(value):
    return round(value, 2)


def main():
    output_rows = []
    all_rows = []

    for dataset_name, csv_path in DATASET_FILES:
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        rows = read_valid_rows(csv_path)
        all_rows.extend(rows)
        dataset_average = average(row["inference_time_ms"] for row in rows)
        output_rows.append(
            {
                "Dataset": dataset_name,
                "Time(ms)": round_time(dataset_average),
            }
        )

    category_times = {}
    for row in all_rows:
        category_times.setdefault(row["category"], []).append(row["inference_time_ms"])

    # Micro: average the mean inference time of each dataset category.
    combined_micro = average(average(times) for times in category_times.values())

    # Macro: average the inference time across all test rows.
    combined_macro = average(row["inference_time_ms"] for row in all_rows)

    output_rows.extend(
        [
            {"Dataset": "Combined(Micro)", "Time(ms)": round_time(combined_micro)},
            {"Dataset": "Combined(Macro)", "Time(ms)": round_time(combined_macro)},
        ]
    )

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Dataset", "Time(ms)"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
