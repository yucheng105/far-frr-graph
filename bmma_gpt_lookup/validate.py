"""validate.py

讀取 build_lookup_table.py 輸出的 real_data_vip_settings.csv，針對每個
security_level 印出摘要（選中的排列、系統總FAR、系統總FRR、總延遲），
方便人工檢查結果是否合理（例如 alpha 越嚴格，選中的排列是否確實使用了
泛化能力較強的偵測器）。

同時檢查兩項格式健全性：
  - 同一個 security_level 底下的 stage 應從 1 開始連續遞增
  - 每一列 threshold_L < threshold_H 皆須成立

=== 資料洩漏提醒 ===
本腳本只檢查 lookup table 內部的一致性，並不能取代在獨立、與校準資料
不重疊的 test set 上做最終系統效能評估。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def validate_lookup_table(csv_path: Path) -> bool:
    """驗證 lookup table 格式並印出每個安全等級的摘要。回傳是否全數通過檢查。"""
    df = pd.read_csv(csv_path)
    all_ok = True

    for level, group in df.groupby("security_level", sort=False):
        group = group.sort_values("stage")
        stage_order = group["stage_order"].iloc[0]
        total_far = group["system_total_far"].iloc[0]
        total_frr = group["system_total_frr"].iloc[0]
        total_latency = group["expected_latency_ms"].sum()

        print(f"\n=== 安全等級: {level} ===")
        print(f"  選中排列: {stage_order}")
        print(f"  system_total_far = {total_far:.6f}")
        print(f"  system_total_frr = {total_frr:.6f}")
        print(f"  總延遲 = {total_latency:.1f} ms")
        print(
            group[["stage", "detector_name", "threshold_L", "threshold_H", "expected_latency_ms"]].to_string(
                index=False
            )
        )

        level_ok = True

        expected_stages = list(range(1, len(group) + 1))
        actual_stages = group["stage"].tolist()
        if actual_stages != expected_stages:
            print(f"  [FAIL] stage 應從 1 連續遞增，實際為 {actual_stages}")
            level_ok = False

        bad_rows = group[group["threshold_L"] >= group["threshold_H"]]
        if not bad_rows.empty:
            print(f"  [FAIL] 有 {len(bad_rows)} 列 threshold_L >= threshold_H")
            level_ok = False

        print("  [OK]" if level_ok else "  [FAIL]")
        all_ok = all_ok and level_ok

    return all_ok


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parent / "output" / "real_data_vip_settings.csv"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    if not path.exists():
        raise SystemExit(f"找不到檔案：{path}，請先跑 build_lookup_table.py")
    ok = validate_lookup_table(path)
    sys.exit(0 if ok else 1)
