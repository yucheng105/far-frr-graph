"""為三個資料集、三個偵測器各畫一張 C 的計算圖。

每張圖三欄：
1. 門檻掃描：FAR、FRR 對門檻，並標出 EER
2. 每一格的 FAR×FRR、EER²，以及取較大者得到的 C
3. 實測點與 FRR = C / FAR

C 的算法與 build_lookup_table.compute_conb 相同：
    C = max(EER², max(FAR×FRR))
不讀 combined 曲線，也不重跑偵測器。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SWEEP = ROOT / "exhaustive-search" / "threshold-sweep" / "results" / "20"
OUT = HERE / "output" / "conb_all"

DATASETS = [
    {
        "key": "faceforensics",
        "label": "FaceForensics++",
        "files": {
            "UCF": SWEEP / "ucf" / "threshold_sweep_ucf_faceforensics.csv",
            "Xception": SWEEP / "xception" / "threshold_sweep_xception_faceforensics.csv",
            "SBI": HERE / "threshold_sweep_sbi_faceforensics_v1.csv",
        },
    },
    {
        "key": "celebdf",
        "label": "Celeb-DF-v2",
        "files": {
            "UCF": SWEEP / "ucf" / "threshold_sweep_ucf_celebdf.csv",
            "Xception": SWEEP / "xception" / "threshold_sweep_xception_celebdf.csv",
            "SBI": SWEEP / "sbi" / "threshold_sweep_sbi_celebdf.csv",
        },
    },
    {
        "key": "dfdc",
        "label": "DFDC",
        "files": {
            "UCF": SWEEP / "ucf" / "threshold_sweep_ucf_dfdc.csv",
            "Xception": SWEEP / "xception" / "threshold_sweep_xception_dfdc.csv",
            "SBI": SWEEP / "sbi" / "threshold_sweep_sbi_dfdc.csv",
        },
    },
]

DETECTORS = ["UCF", "Xception", "SBI"]
COLORS = {"UCF": "#1f77b4", "Xception": "#ff7f0e", "SBI": "#2ca02c"}


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"空的曲線檔：{path}")

    def column(row: dict[str, str], *names: str) -> float:
        for name in names:
            if name in row and row[name] != "":
                return float(row[name])
        raise KeyError(f"{path} 缺少欄位 {names}，現有欄位 {sorted(row)}")

    threshold = np.array([column(row, "Threshold", "threshold") for row in rows])
    far = np.array([column(row, "FAR", "far") for row in rows])
    frr = np.array([column(row, "FRR", "frr") for row in rows])
    order = np.argsort(threshold)
    return threshold[order], far[order], frr[order]


def compute_eer(threshold: np.ndarray, far: np.ndarray, frr: np.ndarray) -> tuple[float, float]:
    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        index = int(np.argmin(np.abs(diff)))
        return float((far[index] + frr[index]) / 2.0), float(threshold[index])

    index = int(sign_changes[0])
    gap = diff[index] - diff[index + 1]
    frac = diff[index] / gap if gap != 0 else 0.0
    eer_threshold = threshold[index] + frac * (threshold[index + 1] - threshold[index])
    eer_far = far[index] + frac * (far[index + 1] - far[index])
    eer_frr = frr[index] + frac * (frr[index + 1] - frr[index])
    return float((eer_far + eer_frr) / 2.0), float(eer_threshold)


def analyze(path: Path) -> dict:
    threshold, far, frr = load_curve(path)
    product = far * frr
    eer, eer_threshold = compute_eer(threshold, far, frr)
    max_index = int(np.argmax(product))
    eer_squared = float(eer**2)
    max_product = float(product[max_index])
    conb = max(eer_squared, max_product)
    return {
        "threshold": threshold,
        "far": far,
        "frr": frr,
        "product": product,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_squared": eer_squared,
        "max_index": max_index,
        "max_product": max_product,
        "conb": conb,
        "bound_source": "EER²" if eer_squared >= max_product else "最大乘積",
    }


def plot_one(dataset_label: str, detector: str, curve: dict, destination: Path) -> None:
    color = COLORS[detector]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    axes[0].plot(curve["threshold"], curve["far"], marker="o", color="#d62728", label="FAR")
    axes[0].plot(curve["threshold"], curve["frr"], marker="o", color="#1f77b4", label="FRR")
    axes[0].axvline(curve["eer_threshold"], color="#444444", linestyle=":", label=f"EER @ {curve['eer_threshold']:.2f}")
    axes[0].set_title("門檻掃描")
    axes[0].set_xlabel("門檻")
    axes[0].set_ylabel("錯誤率")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(curve["threshold"], curve["product"], marker="o", color=color, label="FAR × FRR")
    axes[1].axhline(curve["eer_squared"], color="#9467bd", linestyle=":", label=f"EER² = {curve['eer_squared']:.4f}")
    axes[1].axhline(curve["conb"], color="#444444", linestyle="--", label=f"C = {curve['conb']:.4f}")
    index = curve["max_index"]
    axes[1].scatter(
        [curve["threshold"][index]],
        [curve["product"][index]],
        s=70,
        color="#d62728",
        zorder=3,
        label="最大乘積",
    )
    axes[1].set_title(f"C 取 {curve['bound_source']}")
    axes[1].set_xlabel("門檻")
    axes[1].set_ylabel("FAR × FRR")
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    usable = (curve["far"] > 0) & (curve["frr"] > 0)
    axes[2].scatter(curve["far"][usable], curve["frr"][usable], color=color, label="各門檻實測", zorder=3)
    if np.any(usable):
        y_top = float(curve["frr"][usable].max()) * 1.25
        far_line = np.linspace(float(curve["far"][usable].min()), float(curve["far"][usable].max()), 400)
        bound = curve["conb"] / far_line
        visible = bound <= y_top
        axes[2].plot(far_line[visible], bound[visible], color="#444444", linestyle="--", label="FRR = C / FAR")
        axes[2].set_ylim(0, y_top)
    if curve["far"][index] > 0 and curve["frr"][index] > 0:
        axes[2].scatter(
            [curve["far"][index]],
            [curve["frr"][index]],
            s=70,
            color="#d62728",
            zorder=4,
            label="乘積最大",
        )
    axes[2].set_title("上界曲線")
    axes[2].set_xlabel("FAR")
    axes[2].set_ylabel("FRR")
    axes[2].set_xlim(left=0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{dataset_label} · {detector} · C = {curve['conb']:.4f}（{curve['bound_source']}）")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for dataset in DATASETS:
        for detector in DETECTORS:
            path = dataset["files"][detector]
            if not path.is_file():
                raise FileNotFoundError(path)
            curve = analyze(path)
            figure_path = OUT / f"{dataset['key']}_{detector.lower()}.png"
            plot_one(dataset["label"], detector, curve, figure_path)

            detail_path = OUT / f"{dataset['key']}_{detector.lower()}_thresholds.csv"
            with detail_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["threshold", "far", "frr", "far_times_frr"])
                for threshold, far, frr, product in zip(
                    curve["threshold"], curve["far"], curve["frr"], curve["product"]
                ):
                    writer.writerow([f"{threshold:.4f}", f"{far:.6f}", f"{frr:.6f}", f"{product:.6f}"])

            index = curve["max_index"]
            summary_rows.append(
                {
                    "dataset": dataset["label"],
                    "detector": detector,
                    "source_file": str(path.relative_to(ROOT)),
                    "eer": f"{curve['eer']:.6f}",
                    "eer_threshold": f"{curve['eer_threshold']:.4f}",
                    "eer_squared": f"{curve['eer_squared']:.6f}",
                    "max_product": f"{curve['max_product']:.6f}",
                    "max_product_threshold": f"{curve['threshold'][index]:.2f}",
                    "max_product_far": f"{curve['far'][index]:.6f}",
                    "max_product_frr": f"{curve['frr'][index]:.6f}",
                    "C": f"{curve['conb']:.6f}",
                    "C_source": curve["bound_source"],
                    "figure": str(figure_path.relative_to(ROOT)),
                }
            )
            source_ascii = "eer2" if curve["bound_source"] == "EER²" else "max_product"
            print(
                f"{dataset['label']} {detector}: C={curve['conb']:.6f} "
                f"from {source_ascii}  eer2={curve['eer_squared']:.6f} "
                f"max={curve['max_product']:.6f}"
            )

    summary_path = OUT / "conb_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
