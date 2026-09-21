from pathlib import Path
import csv


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "input-csvs" / "sbi"

SCORE_COLUMN = "fake_score"
LABEL_COLUMN = "label"
THRESHOLDS = [index / 20 for index in range(1,21)]

DATASET_FILES = [
    (
        "sbi_celebdf",
        INPUT_DIR / "scores_visual_sbi_celebdf_v0.csv",
        ROOT_DIR / "threshold_sweep_sbi_celebdf.csv",
    ),
    (
        "sbi_dfdc",
        INPUT_DIR / "scores_visual_sbi_dfdc_v0.csv",
        ROOT_DIR / "threshold_sweep_sbi_dfdc.csv",
    ),
    (
        "sbi_faceforensics",
        INPUT_DIR / "scores_visual_sbi_v0.csv",
        ROOT_DIR / "threshold_sweep_sbi_faceforensics.csv",
    ),
]


def read_valid_rows(csv_path):
    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {LABEL_COLUMN, SCORE_COLUMN}
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
                label = int(row[LABEL_COLUMN])
                fake_score = float(row[SCORE_COLUMN])
            except (TypeError, ValueError):
                continue

            if label not in {0, 1}:
                continue

            rows.append({"label": label, "fake_score": fake_score})

    if not rows:
        raise ValueError(f"{csv_path} does not contain any valid score rows")

    return rows


def calculate_far_frr(rows, threshold):
    real_rows = [row for row in rows if row["label"] == 0]
    fake_rows = [row for row in rows if row["label"] == 1]

    if not real_rows:
        raise ValueError("Cannot calculate FRR because there are no real rows")
    if not fake_rows:
        raise ValueError("Cannot calculate FAR because there are no fake rows")

    # fake_score >= threshold means the model predicts fake.
    false_accepts = sum(row["fake_score"] < threshold for row in fake_rows)
    false_rejects = sum(row["fake_score"] >= threshold for row in real_rows)

    far = false_accepts / len(fake_rows)
    frr = false_rejects / len(real_rows)

    return far, frr


def format_threshold(threshold):
    return f"{threshold:.3f}".rstrip("0").rstrip(".")


def format_rate(rate):
    return round(rate, 6)


def build_sweep_rows(rows):
    sweep_rows = []

    for threshold in THRESHOLDS:
        far, frr = calculate_far_frr(rows, threshold)
        sweep_rows.append(
            {
                "Threshold": format_threshold(threshold),
                "FAR": format_rate(far),
                "FRR": format_rate(frr),
            }
        )

    return sweep_rows


def write_sweep_csv(output_path, rows):
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Threshold", "FAR", "FRR"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    combined_rows = []

    for _, input_path, output_path in DATASET_FILES:
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        rows = read_valid_rows(input_path)
        combined_rows.extend(rows)
        write_sweep_csv(output_path, build_sweep_rows(rows))
        print(f"Wrote {output_path}")

    combined_output_path = ROOT_DIR / "threshold_sweep_sbi_combined.csv"
    write_sweep_csv(combined_output_path, build_sweep_rows(combined_rows))
    print(f"Wrote {combined_output_path}")


if __name__ == "__main__":
    main()
