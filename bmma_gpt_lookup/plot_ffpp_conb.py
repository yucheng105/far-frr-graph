"""畫 FaceForensics++ 的 FAR×FRR 上界圖。

讀現成的 threshold sweep，不重跑偵測器。
C 的算法與 build_lookup_table.compute_conb 相同：
    C = max(EER^2, max(FAR×FRR))
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "output" / "ffpp_conb"

SOURCES = [
    (
        "UCF",
        ROOT
        / "exhaustive-search"
        / "threshold-sweep"
        / "results"
        / "20"
        / "ucf"
        / "threshold_sweep_ucf_faceforensics.csv",
    ),
    (
        "Xception",
        ROOT
        / "exhaustive-search"
        / "threshold-sweep"
        / "results"
        / "20"
        / "xception"
        / "threshold_sweep_xception_faceforensics.csv",
    ),
    (
        "SBI",
        HERE / "threshold_sweep_sbi_faceforensics_v1.csv",
    ),
]

N_REAL = 140
N_FAKE = 560


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"空的曲線檔：{path}")

    def column(row: dict[str, str], *names: str) -> float:
        for name in names:
            if name in row and row[name] != "":
                return float(row[name])
        raise KeyError(f"{path} 缺少欄位 {names}")

    threshold = np.array([column(row, "Threshold", "threshold") for row in rows])
    far = np.array([column(row, "FAR", "far") for row in rows])
    frr = np.array([column(row, "FRR", "frr") for row in rows])
    order = np.argsort(threshold)
    return threshold[order], far[order], frr[order]


def compute_eer(threshold: np.ndarray, far: np.ndarray, frr: np.ndarray) -> tuple[float, float]:
    """與 build_lookup_table.compute_eer 相同：FAR、FRR 交點，必要時線性內插。"""
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


def nearest_counts(rate: float, total: int) -> int:
    return int(round(rate * total))


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    curves = []
    for name, path in SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)
        threshold, far, frr = load_curve(path)
        product = far * frr
        eer, eer_threshold = compute_eer(threshold, far, frr)
        max_index = int(np.argmax(product))
        conb = float(max(eer**2, product[max_index]))
        curves.append(
            {
                "name": name,
                "path": path,
                "threshold": threshold,
                "far": far,
                "frr": frr,
                "product": product,
                "eer": eer,
                "eer_threshold": eer_threshold,
                "max_index": max_index,
                "conb": conb,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "ffpp_conb_table.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "detector",
                "conb",
                "eer",
                "eer_squared",
                "eer_threshold",
                "max_product",
                "max_product_threshold",
                "max_product_far",
                "max_product_frr",
                "bound_source",
            ]
        )
        for curve in curves:
            index = curve["max_index"]
            source = "EER^2" if curve["eer"] ** 2 >= curve["product"][index] else "max(FAR*FRR)"
            writer.writerow(
                [
                    curve["name"],
                    f"{curve['conb']:.6f}",
                    f"{curve['eer']:.6f}",
                    f"{curve['eer'] ** 2:.6f}",
                    f"{curve['eer_threshold']:.4f}",
                    f"{curve['product'][index]:.6f}",
                    f"{curve['threshold'][index]:.2f}",
                    f"{curve['far'][index]:.6f}",
                    f"{curve['frr'][index]:.6f}",
                    source,
                ]
            )

    colors = {"UCF": "#1f77b4", "Xception": "#ff7f0e", "SBI": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
    for axis, curve in zip(axes, curves):
        axis.plot(
            curve["threshold"],
            curve["product"],
            marker="o",
            color=colors[curve["name"]],
            label="FAR × FRR",
        )
        axis.axhline(curve["conb"], color="#444444", linestyle="--", label=f"C = {curve['conb']:.4f}")
        index = curve["max_index"]
        axis.scatter(
            [curve["threshold"][index]],
            [curve["product"][index]],
            s=80,
            color="#d62728",
            zorder=3,
            label="最大乘積",
        )
        axis.set_title(curve["name"])
        axis.set_xlabel("門檻")
        axis.set_xlim(0, 1)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("FAR × FRR")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("FaceForensics++：每個門檻的 FAR×FRR，水平線是上界 C")
    fig.tight_layout()
    product_path = OUT / "product_vs_threshold.png"
    fig.savefig(product_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for axis, curve in zip(axes, curves):
        usable = (curve["far"] > 0) & (curve["frr"] > 0)
        axis.scatter(
            curve["far"][usable],
            curve["frr"][usable],
            color=colors[curve["name"]],
            label="各門檻的實測點",
            zorder=3,
        )
        y_top = float(curve["frr"][usable].max()) * 1.25
        far_line = np.linspace(float(curve["far"][usable].min()), float(curve["far"][usable].max()), 400)
        bound = curve["conb"] / far_line
        visible = bound <= y_top
        axis.plot(far_line[visible], bound[visible], color="#444444", linestyle="--", label="FRR = C / FAR")
        axis.set_ylim(0, y_top)
        index = curve["max_index"]
        if curve["far"][index] > 0 and curve["frr"][index] > 0:
            axis.scatter(
                [curve["far"][index]],
                [curve["frr"][index]],
                s=80,
                color="#d62728",
                zorder=4,
                label="乘積最大的門檻",
            )
        axis.set_title(f"{curve['name']}  C={curve['conb']:.4f}")
        axis.set_xlabel("FAR")
        axis.set_xlim(left=0)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("FRR")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("FaceForensics++：實測點落在 FRR = C / FAR 上或下方")
    fig.tight_layout()
    bound_path = OUT / "far_frr_bound.png"
    fig.savefig(bound_path, dpi=160)
    plt.close(fig)

    print(f"Wrote {summary_path}")
    print(f"Wrote {product_path}")
    print(f"Wrote {bound_path}")
    for curve in curves:
        index = curve["max_index"]
        print(
            f"{curve['name']}: C={curve['conb']:.6f} "
            f"EER={curve['eer']:.4f} @ {curve['eer_threshold']:.3f} "
            f"max product={curve['product'][index]:.6f} @ T={curve['threshold'][index]:.2f} "
            f"FAR={curve['far'][index]:.4f} ({nearest_counts(curve['far'][index], N_FAKE)}/{N_FAKE}) "
            f"FRR={curve['frr'][index]:.4f} ({nearest_counts(curve['frr'][index], N_REAL)}/{N_REAL})"
        )


if __name__ == "__main__":
    main()
