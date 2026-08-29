from pathlib import Path
import csv


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "min_eer_entries.csv"

SKIP_FILENAMES = {OUTPUT_FILE.name}

FIELDNAMES = [
    "source csv",
    "EER",
    "FAR",
    "FRR",
    "sbi_stage",
    "ucf_stage",
    "xception_stage",
    "sbi_upper_threshold",
    "sbi_lower_threshold",
    "ucf_upper_threshold",
    "ucf_lower_threshold",
    "xception_upper_threshold",
    "xception_lower_threshold",
]


def calculate_selection_gap(far, frr):
    return abs(far - frr)


def calculate_eer(far, frr):
    return (far + frr) / 2


def iter_result_csvs():
    for csv_path in sorted(RESULTS_DIR.rglob("*.csv")):
        if csv_path.name in SKIP_FILENAMES:
            continue

        yield csv_path


def find_min_eer_entry(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"FAR", "FRR"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            return None

        min_gap = None
        min_entry = None

        for row in reader:
            try:
                far = float(row["FAR"])
                frr = float(row["FRR"])
            except (TypeError, ValueError):
                continue

            selection_gap = calculate_selection_gap(far, frr)
            if min_gap is None or selection_gap < min_gap:
                min_gap = selection_gap
                min_entry = dict(row)

        if min_entry is None:
            return None

        far = float(min_entry["FAR"])
        frr = float(min_entry["FRR"])

        return {
            "source csv": str(csv_path.relative_to(ROOT_DIR)),
            "EER": round(calculate_eer(far, frr), 12),
            "FAR": min_entry["FAR"],
            "FRR": min_entry["FRR"],
            "sbi_stage": min_entry.get("sbi_stage", -1),
            "ucf_stage": min_entry.get("ucf_stage", -1),
            "xception_stage": min_entry.get("xception_stage", -1),
            "sbi_upper_threshold": min_entry.get("sbi_upper_threshold", -1),
            "sbi_lower_threshold": min_entry.get("sbi_lower_threshold", -1),
            "ucf_upper_threshold": min_entry.get("ucf_upper_threshold", -1),
            "ucf_lower_threshold": min_entry.get("ucf_lower_threshold", -1),
            "xception_upper_threshold": min_entry.get("xception_upper_threshold", -1),
            "xception_lower_threshold": min_entry.get("xception_lower_threshold", -1),
        }


def main():
    summary_rows = []

    for csv_path in iter_result_csvs():
        min_eer_entry = find_min_eer_entry(csv_path)
        if min_eer_entry is None:
            print(f"Skipped {csv_path}")
            continue

        summary_rows.append(min_eer_entry)
        print(f"Processed {csv_path}")

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
