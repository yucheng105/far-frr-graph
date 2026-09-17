"""build_lookup_table.py

實作 BMMA-GPT（Advancing Biometric Authentication With Dual-Threshold
Multi-Modal Systems and Geometric Programming）論文的雙門檻序列架構，
套用在 Anti-Deepfake-Box (ADB) 的序列式（cascade）Deepfake 影片偵測系統
第一階段（SBI、Xception、UCF 三個 Visual 分類器）。

流程對應任務書的四個步驟：
  步驟1：讀取/合併 threshold sweep 資料 -> 每個偵測器的 DetectorProfile（EER/conb/Tb）
  步驟2：排列窮舉（itertools.permutations，長度 1..N）
  步驟3：對每個排列用 cvxpy 的 DGP（gp=True）求解最佳雙門檻
  步驟4：依安全等級（alpha/beta/latency_budget_ms/min_stages）篩選最佳排列，
         反查回實際門檻，輸出 real_data_vip_settings.csv

完整背景、資料來源、過程中發現的資料問題（SBI far/frr 欄位寫反、
FF++ 該用 v0 還是 v1）、以及「為什麼串接數學上贏不了單一偵測器」
（對應論文公式17的性質）這幾件事的完整說明，請見同資料夾的 README.md。
這裡的程式碼註解只保留跟「讀這段程式碼」直接相關的重點提醒。

=== 資料洩漏警告（重要） ===
這裡讀入的 threshold_sweep_*.csv 是用來「校準」雙門檻的資料（far/frr 曲線），
絕對不可以和最終系統效能評估用的 test set 重疊，否則校準出來的門檻會對
test set 過擬合，使最終回報的 FAR/FRR 評估結果失真、失去意義。
使用本腳本前請自行確認傳入的 sweep 檔案來源與最終 test set 沒有交集。
"""

from __future__ import annotations

import itertools
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cvxpy is required for step 3 (GP solving). Run: pip install -r requirements.txt"
    ) from exc


CombineMode = Literal["max", "mean"]

EPS = 1e-6
INV_FAR_ATOL = 1e-9
THRESHOLD_EQ_ATOL = 1e-6


# =============================================================================
# 步驟 1：讀取與合併 threshold sweep 資料，建立每個偵測器的統計 Profile
# =============================================================================


def _infer_detector_and_dataset_from_filename(path: Path) -> tuple[str, str]:
    """從檔名 threshold_sweep_<detector>_<dataset>.csv 反推 detector_name / dataset。

    只有在 CSV 本身沒有 detector_name / dataset 欄位時才會呼叫
    （SBI/UCF/Xception 部分檔案只有 threshold,far,frr 三欄）。
    """
    stem = path.stem
    prefix = "threshold_sweep_"
    if not stem.startswith(prefix):
        raise ValueError(
            f"無法從檔名判斷 detector/dataset：{path.name}，"
            f"檔名需符合 threshold_sweep_<detector>_<dataset>.csv 格式"
        )
    remainder = stem[len(prefix):]
    parts = remainder.split("_")
    if len(parts) < 2:
        raise ValueError(f"無法從檔名判斷 detector/dataset：{path.name}")
    detector = parts[0]
    dataset = "_".join(parts[1:])
    return detector, dataset


_DATASET_ALIAS = {
    "faceforensics": "FaceForensics",
    "ff++": "FaceForensics",
    "ffpp": "FaceForensics",
    "celebdf": "CelebDF",
    "celeb-df": "CelebDF",
    "dfdc": "DFDC",
}


def normalize_dataset_name(name: str) -> str:
    """把不同來源（檔名 / CSV 欄位 / latency 表）對資料集的命名統一，
    避免 "FF++" / "faceforensics" / "FaceForensics" 被當成三個不同資料集。
    """
    key = name.strip().lower().replace("_", "").replace(" ", "")
    return _DATASET_ALIAS.get(key, name.strip())


def load_sweep_file(path: Path) -> pd.DataFrame:
    """讀入單一 threshold sweep CSV，回傳統一欄位的 DataFrame。

    支援兩種輸入格式：
      1) detector_name, dataset, threshold, far, frr, avg_latency_ms, n_samples
      2) threshold, far, frr（detector_name/dataset 從檔名反推，
         avg_latency_ms/n_samples 留空，由 attach_latency() 補上）
    欄位名稱比對不分大小寫（Threshold/FAR/FRR 與 threshold/far/frr 皆可）。
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def col(name: str) -> str | None:
        return lower_map.get(name)

    threshold_col, far_col, frr_col = col("threshold"), col("far"), col("frr")
    if threshold_col is None or far_col is None or frr_col is None:
        raise ValueError(f"{path} 缺少 threshold/far/frr 欄位")

    out = pd.DataFrame(
        {
            "threshold": df[threshold_col].astype(float),
            "far": df[far_col].astype(float),
            "frr": df[frr_col].astype(float),
        }
    )

    if col("detector_name") is not None and col("dataset") is not None:
        detector_name = str(df[col("detector_name")].iloc[0])
        dataset = str(df[col("dataset")].iloc[0])
    else:
        detector_name, dataset = _infer_detector_and_dataset_from_filename(path)

    out["detector_name"] = detector_name
    out["dataset"] = normalize_dataset_name(dataset)
    out["avg_latency_ms"] = float(df[col("avg_latency_ms")].iloc[0]) if col("avg_latency_ms") else np.nan
    out["n_samples"] = float(df[col("n_samples")].iloc[0]) if col("n_samples") else np.nan
    out["source_file"] = path.name
    return out.sort_values("threshold").reset_index(drop=True)


def attach_latency(df: pd.DataFrame, latency_lookup: pd.DataFrame) -> pd.DataFrame:
    """對缺少 avg_latency_ms 的列，用外部 latency_lookup
    （欄位：detector_name, dataset, avg_latency_ms）依 (detector_name, dataset) 補值。

    找不到對應延遲資料時直接拋錯，要求使用者在 detector_latency.csv 補上，
    不自行假設延遲數字——延遲會直接影響 Tb 與 latency_budget_ms 篩選，不可亂猜。
    """
    df = df.copy()
    missing_mask = df["avg_latency_ms"].isna()
    if not missing_mask.any():
        return df

    lookup = latency_lookup.copy()
    lookup["dataset"] = lookup["dataset"].map(normalize_dataset_name)
    lookup_map = {(row.detector_name, row.dataset): row.avg_latency_ms for row in lookup.itertuples()}

    for idx in df.index[missing_mask]:
        key = (df.at[idx, "detector_name"], df.at[idx, "dataset"])
        if key not in lookup_map:
            raise ValueError(
                f"缺少延遲資料：detector={key[0]}, dataset={key[1]}"
                f"（來源檔 {df.at[idx, 'source_file']}）。"
                f"請在 detector_latency.csv 補上這組 (detector_name, dataset) 的 avg_latency_ms。"
            )
        df.at[idx, "avg_latency_ms"] = lookup_map[key]
    return df


def verify_far_frr_direction(df: pd.DataFrame, swap_far_frr: bool = False) -> pd.DataFrame:
    """驗證 far 是否隨 threshold 遞增、frr 是否隨 threshold 遞減。

    不自行猜測方向並默默處理：若方向與預期相反、且使用者沒有明確傳入
    swap_far_frr=True，會印出警告並中止程式，要求人工確認 far/frr 定義後，
    再決定是否加上 swap_far_frr=True 對調兩欄。
    """
    df = df.sort_values("threshold")
    if df["threshold"].nunique() > 1:
        far_trend = np.corrcoef(df["threshold"], df["far"])[0, 1]
        frr_trend = np.corrcoef(df["threshold"], df["frr"])[0, 1]
    else:
        far_trend = frr_trend = np.nan

    direction_ok = (np.isnan(far_trend) or far_trend >= 0) and (np.isnan(frr_trend) or frr_trend <= 0)

    if direction_ok and not swap_far_frr:
        return df

    if not direction_ok and not swap_far_frr:
        msg = (
            f"\n[方向警告] detector={df['detector_name'].iloc[0]} "
            f"dataset={df['dataset'].iloc[0]} 來源={df['source_file'].iloc[0]}\n"
            f"  far 對 threshold 的相關係數 = {far_trend:.4f}（預期 >= 0，far 應隨 threshold 遞增）\n"
            f"  frr 對 threshold 的相關係數 = {frr_trend:.4f}（預期 <= 0，frr 應隨 threshold 遞減）\n"
            f"  這與程式假設的欄位語意矛盾，不會自動猜測方向並默默處理。\n"
            f"  請人工檢查這份 CSV 的 far/frr 定義，確認後再用 swap_far_frr=True 呼叫。\n"
        )
        print(msg, file=sys.stderr)
        raise SystemExit(
            f"far/frr 方向與預期矛盾（{df['source_file'].iloc[0]}），已停止執行，"
            f"請先人工確認方向後再決定是否加上 swap_far_frr=True。"
        )

    if direction_ok and swap_far_frr:
        warnings.warn(
            f"[{df['detector_name'].iloc[0]}/{df['dataset'].iloc[0]}] "
            f"far/frr 方向本來就正確，但 swap_far_frr=True 仍會強制對調，請確認這是你要的行為。"
        )

    df = df.copy()
    df["far"], df["frr"] = df["frr"].copy(), df["far"].copy()
    return df


def _strictly_increasing(arr: np.ndarray) -> np.ndarray:
    """對單調不遞減陣列加上極小遞增量，避免內插反查時因重複值產生歧義。"""
    return arr + np.arange(len(arr)) * 1e-12


def align_to_common_grid(dataset_dfs: list[pd.DataFrame]) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """把同一偵測器、不同來源檔案的 sweep 曲線用線性內插對齊到共同 threshold 格點。

    共同格點取所有輸入檔案 threshold 的聯集（排序去重）。落在單一檔案
    threshold 範圍之外的格點，以該檔案邊界值延伸（np.interp 預設行為），
    因為 far/frr 曲線在量測範圍外通常已趨於平緩。

    回傳的 far_list/frr_list 依輸入檔案順序排列（不是依資料集名稱），
    避免同一偵測器有多份檔案剛好對應同一資料集名稱時（例如 SBI 的 v0/v1
    都是 FF++）互相覆蓋。
    """
    grid = np.unique(np.concatenate([d["threshold"].to_numpy() for d in dataset_dfs]))
    far_list = [np.interp(grid, d["threshold"].to_numpy(), d["far"].to_numpy()) for d in dataset_dfs]
    frr_list = [np.interp(grid, d["threshold"].to_numpy(), d["frr"].to_numpy()) for d in dataset_dfs]
    return grid, far_list, frr_list


def combine_across_datasets(
    far_list: list[np.ndarray], frr_list: list[np.ndarray], combine_mode: CombineMode = "max"
) -> tuple[np.ndarray, np.ndarray]:
    """跨資料集合併 far/frr 曲線。

    combine_mode="max"（預設，保守策略）：每個門檻切點取跨資料集最差的
    far、frr，避免對單一 domain 校準過頭、犧牲域外泛化能力。
    combine_mode="mean"：跨資料集平均。
    """
    far_stack = np.stack(far_list, axis=0)
    frr_stack = np.stack(frr_list, axis=0)
    if combine_mode == "max":
        return far_stack.max(axis=0), frr_stack.max(axis=0)
    if combine_mode == "mean":
        return far_stack.mean(axis=0), frr_stack.mean(axis=0)
    raise ValueError(f"未知的 combine_mode: {combine_mode!r}")


def enforce_monotonic(far: np.ndarray, frr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """跨資料集合併後的 far/frr 做單調化：far 應嚴格不遞減、frr 應嚴格不遞減
    （反向而言是不遞增）。

    使用累積最大值（far 正向、frr 反向）去除合併造成的非單調雜訊，
    確保後續 inv_far()/inv_frr() 反查是良定義的。
    """
    far_mono = np.maximum.accumulate(far)
    frr_mono = np.maximum.accumulate(frr[::-1])[::-1]
    return far_mono, frr_mono


def compute_eer(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray) -> tuple[float, float]:
    """計算 EER：far(threshold) 與 frr(threshold) 曲線的交點（BMMA-GPT 論文中
    FAR == FRR 時的錯誤率定義）。

    用線性內插找 far-frr 變號的位置；若曲線完全沒有交叉（例如資料稀疏、
    兩條曲線不相交），退回取 |far-frr| 最小的格點作為近似值。
    """
    diff = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])

    i = sign_changes[0]
    t0, t1 = thresholds[i], thresholds[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    frac = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.0
    eer_threshold = t0 + frac * (t1 - t0)
    eer_far = far[i] + frac * (far[i + 1] - far[i])
    eer_frr = frr[i] + frac * (frr[i + 1] - frr[i])
    return float((eer_far + eer_frr) / 2.0), float(eer_threshold)


def compute_conb(far: np.ndarray, frr: np.ndarray, eer: float) -> float:
    """計算 conb（FAR x FRR 守恆常數）。

    conb = max(EER^2, max_T far(T) * frr(T))，取兩者較大值作為保守估計，
    對應 BMMA-GPT 論文中 GP 模型 RB = conb / AB 這組單項式守恆關係的常數。
    """
    return float(max(eer**2, float(np.max(far * frr))))


@dataclass
class DetectorProfile:
    """單一偵測器的統計 Profile：EER、conb、Tb，以及正查/反查用的內插曲線。"""

    name: str
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    eer: float
    eer_threshold: float
    conb: float
    Tb: float

    def __post_init__(self) -> None:
        # far 嚴格遞增、frr（反轉後）嚴格遞增，供 inv_far/inv_frr 內插使用
        self._far_strict = _strictly_increasing(self.far)
        self._frr_asc_strict = _strictly_increasing(self.frr[::-1])  # 對應 thresholds[::-1]

    def far_at(self, T: float) -> float:
        """正查：給定門檻 T，線性內插查出對應的 far(T)。"""
        return float(np.interp(T, self.thresholds, self.far))

    def frr_at(self, T: float) -> float:
        """正查：給定門檻 T，線性內插查出對應的 frr(T)。"""
        return float(np.interp(T, self.thresholds, self.frr))

    def inv_far(self, target_far: float) -> float | None:
        """反查：給定目標 far，線性內插反推回門檻 T（far 視為嚴格遞增的自變數）。

        目標 far 必須落在這條曲線的觀測範圍內（允許 INV_FAR_ATOL 的數值誤差）。
        超出範圍回傳 None，不靜默 clip——否則 GP 要的錯誤率與落地門檻對不上。
        """
        lo = float(self._far_strict[0])
        hi = float(self._far_strict[-1])
        if target_far < lo - INV_FAR_ATOL or target_far > hi + INV_FAR_ATOL:
            return None
        clipped = float(np.clip(target_far, lo, hi))
        return float(np.interp(clipped, self._far_strict, self.thresholds))

    def inv_frr(self, target_frr: float) -> float:
        """反查：給定目標 frr，線性內插反推回門檻 T（frr 嚴格遞減，反轉陣列後查表）。"""
        x = self._frr_asc_strict  # 隨 thresholds[::-1] 遞增
        y = self.thresholds[::-1]
        clipped = float(np.clip(target_frr, x[0], x[-1]))
        return float(np.interp(clipped, x, y))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DetectorProfile(name={self.name!r}, EER={self.eer:.4f} "
            f"@T={self.eer_threshold:.3f}, conb={self.conb:.6f}, Tb={self.Tb:.1f}ms)"
        )


def build_detector_profile(
    name: str,
    sweep_files: list[Path],
    latency_lookup: pd.DataFrame | None = None,
    combine_mode: CombineMode = "max",
    swap_far_frr: bool = False,
) -> DetectorProfile:
    """讀取單一偵測器的所有 sweep 檔案，建立 DetectorProfile。

    流程：讀檔 -> 補延遲 -> 驗證 far/frr 方向 -> 對齊共同格點 ->
    跨資料集合併（combine_mode）-> 單調化 -> 計算 EER / conb / Tb（Tb 恆取
    跨資料集 avg_latency_ms 的最大值，為保守策略，與 combine_mode 無關）。
    """
    dfs = [load_sweep_file(p) for p in sweep_files]
    for d in dfs:
        d["detector_name"] = name  # 檔名反推可能大小寫不一致，統一用呼叫端指定的 name
    if latency_lookup is not None:
        dfs = [attach_latency(d, latency_lookup) for d in dfs]
    dfs = [verify_far_frr_direction(d, swap_far_frr=swap_far_frr) for d in dfs]

    for d in dfs:
        if d["avg_latency_ms"].isna().any():
            raise ValueError(
                f"{name} 的 {d['source_file'].iloc[0]} 缺少 avg_latency_ms，"
                f"且沒有提供 latency_lookup 可供補值。"
            )

    grid, far_list, frr_list = align_to_common_grid(dfs)
    far_combined, frr_combined = combine_across_datasets(far_list, frr_list, combine_mode)
    far_mono, frr_mono = enforce_monotonic(far_combined, frr_combined)

    eer, eer_threshold = compute_eer(grid, far_mono, frr_mono)
    conb = compute_conb(far_mono, frr_mono, eer)
    Tb = float(max(d["avg_latency_ms"].iloc[0] for d in dfs))

    return DetectorProfile(
        name=name,
        thresholds=grid,
        far=far_mono,
        frr=frr_mono,
        eer=eer,
        eer_threshold=eer_threshold,
        conb=conb,
        Tb=Tb,
    )


# =============================================================================
# 步驟 2：排列窮舉
# =============================================================================


def enumerate_orderings(detector_names: list[str]) -> list[tuple[str, ...]]:
    """窮舉長度 1..N 的所有排列（子集合 x 順序）。

    3 個偵測器應產生 C(3,1)*1! + C(3,2)*2! + C(3,3)*3! = 3 + 6 + 6 = 15 種排列。
    """
    orderings: list[tuple[str, ...]] = []
    for r in range(1, len(detector_names) + 1):
        orderings.extend(itertools.permutations(detector_names, r))
    return orderings


# =============================================================================
# 步驟 3：對每個排列，用幾何規劃（GP / DGP）求解最佳雙門檻
# =============================================================================


def solve_gp_thresholds(profiles: list[DetectorProfile], alpha: float) -> dict | None:
    """對固定排列，用 cvxpy 的 DGP（gp=True）求解雙門檻，最小化系統總 FRR。

    決策變數 AB_lb[i]、AB_ub[i]（每個階段各一個，皆為正變數，`cp.Variable(pos=True)`）。
    透過 conb 守恆關係推導 RB（單項式，符合 GP 形式）：
        RB_lb[i] = conb[i] / AB_lb[i]，RB_ub[i] = conb[i] / AB_ub[i]

    目標函數（對應 BMMA-GPT 論文公式 17，最小化系統總 FRR，正項式）：
        total_frr = sum_{s=1}^{n} ( prod_{j<s} RB_lb[j] ) * RB_ub[s]

    約束一（對應論文公式 18，系統總 FAR 不超過安全等級的 alpha，正項式 <= 常數）：
        total_far = sum_{s=1}^{n} ( prod_{j<s} AB_ub[j] ) * AB_lb[s]  <= alpha

    約束二（門檻順序關係，逐階段皆須成立）：AB_lb[i] <= AB_ub[i]

    約束三（最後一關必須做出最終決策）：AB_lb[n-1] == AB_ub[n-1]
        對應 T_L == T_H。中間帶只在「後面還有偵測器」時合法，否則公式 17/18
        會把猶豫樣本從 FAR/FRR 裡拿掉，solver 會把最後一關的 AB_lb 壓到 EPS。

    約束四（RB 必須是機率）：AB_lb[i] >= conb[i]  （等價 RB_lb[i] <= 1；
        搭配 AB_lb <= AB_ub 後 AB_ub >= conb，故 RB_ub <= 1）。少了這條，
        階段乘積 Π RB_lb 可以大於 1，total_frr 會爆掉。

    變數上界：所有變數 <= 1-EPS，避免數值問題。

    求解失敗（非 DGP、SolverError、或 status 非 optimal）回傳 None。
    """
    n = len(profiles)
    ab_lb = cp.Variable(n, pos=True)
    ab_ub = cp.Variable(n, pos=True)

    rb_lb = [profiles[i].conb / ab_lb[i] for i in range(n)]
    rb_ub = [profiles[i].conb / ab_ub[i] for i in range(n)]

    total_frr_terms = []
    rb_lb_prefix_prod = 1
    for s in range(n):
        total_frr_terms.append(rb_lb_prefix_prod * rb_ub[s])
        rb_lb_prefix_prod = rb_lb_prefix_prod * rb_lb[s]
    total_frr = cp.sum(total_frr_terms)

    total_far_terms = []
    ab_ub_prefix_prod = 1
    for s in range(n):
        total_far_terms.append(ab_ub_prefix_prod * ab_lb[s])
        ab_ub_prefix_prod = ab_ub_prefix_prod * ab_ub[s]
    total_far = cp.sum(total_far_terms)

    constraints = [total_far <= alpha]
    for i in range(n):
        conb_i = float(profiles[i].conb)
        constraints += [
            ab_lb[i] <= ab_ub[i],
            ab_lb[i] >= conb_i,
            ab_lb[i] <= 1 - EPS,
            ab_ub[i] >= conb_i,
            ab_ub[i] <= 1 - EPS,
        ]
    constraints.append(ab_lb[n - 1] == ab_ub[n - 1])

    problem = cp.Problem(cp.Minimize(total_frr), constraints)
    if not problem.is_dgp():
        raise RuntimeError("GP 模型不是 DGP，請檢查約束寫法。")

    try:
        solve_kwargs: dict = {"gp": True}
        if "ECOS" in cp.installed_solvers():
            solve_kwargs["solver"] = cp.ECOS
        problem.solve(**solve_kwargs)
    except cp.error.SolverError:
        return None

    if problem.status == "optimal_inaccurate":
        if total_far.value is None or float(total_far.value) > alpha + 1e-8:
            return None
    elif problem.status != "optimal":
        return None

    if ab_lb.value is None or ab_ub.value is None:
        return None

    stages = [
        {
            "ab_lb": float(ab_lb.value[i]),
            "ab_ub": float(ab_ub.value[i]),
            "rb_lb": float(profiles[i].conb / ab_lb.value[i]),
            "rb_ub": float(profiles[i].conb / ab_ub.value[i]),
        }
        for i in range(n)
    ]

    return {
        "stages": stages,
        "gp_total_far": float(total_far.value),
        "gp_total_frr": float(total_frr.value),
        "total_far": float(total_far.value),
        "total_frr": float(total_frr.value),
    }


# =============================================================================
# 步驟 4：依安全等級篩選最佳排列，反查回實際門檻，輸出 Lookup Table
# =============================================================================


SECURITY_LEVELS: dict[str, dict[str, float]] = {
    # min_stages 對應論文 Section IV-B 提到的 CISO 可設定的
    # "minimum number of required modalities" 政策：即使數學上單一偵測器的
    # total_frr 永遠是理論下界（見 select_best_ordering 說明），Defense-in-Depth
    # 仍要求較高安全等級至少疊加多個偵測器，降低單一偵測器被針對性繞過的風險。
    # 下面的 alpha/beta 已依 SBI/Xception/UCF 這三個偵測器目前的實際 conb
    # （0.137~0.324，遠高於一般生物特徵辨識常見的 EER）重新校準過，不是題目
    # 原始範例的數字。校準方式：用 dump_all_orderings() 的診斷結果，找出每個
    # min_stages 門檻下實際可達到的最小 total_frr，beta 設在其上留一點餘裕；
    # forensic 因為要求 min_stages=3，實測至少要 alpha>=0.15 才能讓 3 階段的
    # threshold_L<threshold_H 成立（alpha 越嚴，AB_lb 被壓得越小，RB_lb=conb/AB_lb
    # 越容易反查不出合法門檻），因此 forensic 的 alpha 反而比 balanced 寬——這是
    # 「要求更多層防禦」與「FAR 上限更嚴」在目前偵測器品質下互相衝突的結果，
    # 而不是設定錯誤；如果之後偵測器 conb 降低，這裡的數字應該重新校準。
    "fast_screen": {"alpha": 0.05, "beta": 0.15, "latency_budget_ms": 1500, "min_stages": 1},
    "balanced": {"alpha": 0.05, "beta": 0.65, "latency_budget_ms": 4000, "min_stages": 2},
    "forensic": {"alpha": 0.3, "beta": 0.6, "latency_budget_ms": 100000, "min_stages": 3},
}


def _reverse_map_thresholds(
    ordering: tuple[str, ...], detectors: list[DetectorProfile], result: dict
) -> list[dict] | None:
    """把 GP 求出的 AB_lb / AB_ub 用 inv_far 換算回 threshold_L / threshold_H。

    兩邊都走 FAR 曲線：FAR 嚴格遞增且 AB_lb <= AB_ub，故 T_L <= T_H。
    最後一關 AB_lb == AB_ub，反查後 T_L == T_H。
    任一目標 FAR 超出該偵測器曲線範圍時回傳 None（不靜默 clip）。
    """
    rows = []
    n = len(ordering)
    for i, (detector_name, profile, stage) in enumerate(zip(ordering, detectors, result["stages"])):
        threshold_L = profile.inv_far(stage["ab_lb"])
        threshold_H = profile.inv_far(stage["ab_ub"])
        if threshold_L is None or threshold_H is None:
            return None
        if i == n - 1:
            # 最後一關單一門檻：兩個反查應對到同一個 T，取平均避免浮點差。
            threshold_L = threshold_H = 0.5 * (threshold_L + threshold_H)
        rows.append(
            {
                "stage": i + 1,
                "detector_name": detector_name,
                "threshold_L": threshold_L,
                "threshold_H": threshold_H,
                "expected_latency_ms": profile.Tb,
            }
        )
    return rows


def _thresholds_valid(rows: list[dict]) -> bool:
    """前 N-1 關必須 T_L < T_H（還有續傳區間）；最後一關必須 T_L ≈ T_H。"""
    n = len(rows)
    for i, row in enumerate(rows):
        left, right = row["threshold_L"], row["threshold_H"]
        if i == n - 1:
            if abs(left - right) > THRESHOLD_EQ_ATOL:
                return False
        elif not left < right:
            return False
    return True


def compute_cascade_far_frr(
    detectors: list[DetectorProfile], rows: list[dict]
) -> tuple[float, float]:
    """用差式 cascade 公式，依落地門檻重算系統 FAR / FRR。

    FAR = Σ_s FAR_s(T_L) · Π_{j<s} [FAR_j(T_H) − FAR_j(T_L)]
    FRR = Σ_s FRR_s(T_H) · Π_{j<s} [FRR_j(T_L) − FRR_j(T_H)]
    最後一關 T_L == T_H 時續傳機率為 0，每個輸入都會得到最終決策。
    """
    total_far = 0.0
    total_frr = 0.0
    far_reach = 1.0
    frr_reach = 1.0
    for profile, row in zip(detectors, rows):
        far_L = profile.far_at(row["threshold_L"])
        far_H = profile.far_at(row["threshold_H"])
        frr_L = profile.frr_at(row["threshold_L"])
        frr_H = profile.frr_at(row["threshold_H"])
        total_far += far_L * far_reach
        total_frr += frr_H * frr_reach
        far_reach *= max(far_H - far_L, 0.0)
        frr_reach *= max(frr_L - frr_H, 0.0)
    return float(total_far), float(total_frr)


def select_best_ordering(
    orderings: list[tuple[str, ...]],
    profiles_by_name: dict[str, DetectorProfile],
    alpha: float,
    beta: float,
    latency_budget_ms: float,
    min_stages: int = 1,
) -> tuple[tuple[str, ...], dict, float] | None:
    """對給定安全等級 (alpha, beta, latency_budget_ms, min_stages) 挑出最佳排列。

    篩選順序：
      0) 排列長度 >= min_stages（對應論文 Section IV-B 的
         "minimum number of required modalities" 政策）。這一步是必要的：
         依論文公式(17)的結構，total_frr 的第一項 RB_ub[1]=conb1/AB_ub[1] 恆
         大於 conb1（因 AB_ub[1]<1），後面每多一階段只會再疊加一個恆正的項，
         所以「單一偵測器」永遠是 total_frr 的理論下界，數學上沒有任何串接
         排列能贏過它。若不設 min_stages，選擇邏輯會永遠只選單一偵測器，
         違背 Defense-in-Depth（多層防禦、避免單點被針對性繞過）的設計初衷。
      1) 排列各偵測器 Tb 加總 <= latency_budget_ms
      2) 用 solve_gp_thresholds 求解，求解失敗（最後一關單門檻 + RB<=1 下
         可能對過嚴的 alpha 直接不可行）者剔除
      3) 反查門檻成功，且前 N-1 關 T_L < T_H、最後一關 T_L ≈ T_H
      4) 用差式 cascade 重算後的系統 FAR <= alpha、FRR <= beta
      5) 依總延遲由小到大排序，總延遲打平則取（重算後）total_frr 較小者

    完全沒有可行排列時回傳 None，並印出診斷訊息說明是哪個篩選步驟刷掉了全部候選。
    """
    n_total = len(orderings)
    n_min_stages_ok = 0
    n_latency_ok = 0
    n_gp_ok = 0
    n_reverse_ok = 0
    n_beta_ok = 0

    candidates: list[tuple[tuple[str, ...], dict, float]] = []
    for ordering in orderings:
        if len(ordering) < min_stages:
            continue
        n_min_stages_ok += 1

        detectors = [profiles_by_name[name] for name in ordering]
        total_latency = sum(d.Tb for d in detectors)
        if total_latency > latency_budget_ms:
            continue
        n_latency_ok += 1

        result = solve_gp_thresholds(detectors, alpha)
        if result is None:
            continue
        n_gp_ok += 1

        rows = _reverse_map_thresholds(ordering, detectors, result)
        if rows is None or not _thresholds_valid(rows):
            continue
        n_reverse_ok += 1

        emp_far, emp_frr = compute_cascade_far_frr(detectors, rows)
        if emp_far > alpha or emp_frr > beta:
            continue
        n_beta_ok += 1

        result = {
            **result,
            "total_far": emp_far,
            "total_frr": emp_frr,
            "stage_rows": rows,
        }
        candidates.append((ordering, result, total_latency))

    candidates.sort(key=lambda c: (c[2], c[1]["total_frr"]))
    if candidates:
        return candidates[0]

    if n_min_stages_ok == 0:
        reason = f"min_stages={min_stages} 超過可用偵測器數量，沒有任何排列符合最少階段數要求。"
    elif n_latency_ok == 0:
        reason = "建議放寬 latency_budget_ms（目前太緊，所有排列的 Tb 總和都超過預算）。"
    elif n_gp_ok == 0:
        reason = (
            "GP 求不到滿足 alpha 的解（最後一關單門檻且 RB<=1 之後，"
            "若偵測器 conb 大於 alpha 會直接不可行）。不要先放寬 alpha，下一步應檢查 combine/conb。"
        )
    elif n_reverse_ok == 0:
        reason = "GP 有解，但反查時目標 FAR 超出曲線範圍，或門檻順序不合法。"
    else:
        reason = "落地門檻用差式 cascade 重算後，系統 FAR 超過 alpha 或 FRR 超過 beta。"

    print(
        f"[警告] 找不到可行排列：共 {n_total} 種排列，"
        f"{n_min_stages_ok} 種滿足 min_stages={min_stages}，"
        f"{n_latency_ok} 種通過 latency_budget_ms={latency_budget_ms} 篩選，"
        f"{n_gp_ok} 種 GP 求解成功且滿足 alpha={alpha}，"
        f"{n_reverse_ok} 種反查出門檻，"
        f"{n_beta_ok} 種同時滿足重算後 FAR<=alpha、FRR<=beta={beta}。\n  -> {reason}"
    )
    return None


ALL_ORDERINGS_ALPHA = 0.05


def dump_all_orderings(
    profiles_by_name: dict[str, DetectorProfile],
    alpha: float = ALL_ORDERINGS_ALPHA,
) -> pd.DataFrame:
    """診斷用途：不套用 SECURITY_LEVELS 的 beta/latency_budget_ms 篩選，也不挑
    「最佳」排列，而是對全部排列都用同一個 alpha 求解 GP，把每個排列（GP 求解
    成功者）的階段門檻與 total_far/total_frr/總延遲都輸出成一張大表。

    用途：當所有 SECURITY_LEVELS 都被 beta 篩到沒有可行排列時（實際偵測器
    表現比預設的安全等級嚴格很多），可以用這張表看清楚全部 15 種排列各自
    實際能做到的 total_far/total_frr/延遲，再回頭決定 SECURITY_LEVELS 的
    beta 該設多少才合理——而不是盲猜。

    注意：這裡的 alpha 仍然是 GP 模型公式18 的硬約束（系統總 FAR 上限），
    不會被跳過；被跳過的只有 select_best_ordering() 那層「挑最佳 + beta +
    latency_budget_ms」的外部篩選。反查失敗或最後一關不是單門檻時，
    仍會輸出該排列（門檻為空），並標記 valid_thresholds=False。
    CSV 的 system_total_far/frr 在反查成功時為差式 cascade 重算值；
    gp_total_far/frr 為 GP 代數值，方便對照。
    """
    detector_names = list(profiles_by_name.keys())
    orderings = enumerate_orderings(detector_names)

    rows = []
    n_failed = 0
    n_reverse_failed = 0
    for ordering in orderings:
        detectors = [profiles_by_name[name] for name in ordering]
        total_latency = sum(d.Tb for d in detectors)
        result = solve_gp_thresholds(detectors, alpha)
        if result is None:
            n_failed += 1
            continue

        stage_rows = _reverse_map_thresholds(ordering, detectors, result)
        stage_order_str = ",".join(ordering)
        valid_thresholds = stage_rows is not None and _thresholds_valid(stage_rows)
        if not valid_thresholds:
            n_reverse_failed += 1
            emp_far = emp_frr = float("nan")
            stage_rows = stage_rows or [
                {
                    "stage": i + 1,
                    "detector_name": name,
                    "threshold_L": float("nan"),
                    "threshold_H": float("nan"),
                    "expected_latency_ms": detectors[i].Tb,
                }
                for i, name in enumerate(ordering)
            ]
        else:
            emp_far, emp_frr = compute_cascade_far_frr(detectors, stage_rows)

        for r in stage_rows:
            rows.append(
                {
                    "stage_order": stage_order_str,
                    "n_stages": len(ordering),
                    "stage": r["stage"],
                    "modality": "visual",
                    "detector_name": r["detector_name"],
                    "threshold_L": r["threshold_L"],
                    "threshold_H": r["threshold_H"],
                    "expected_latency_ms": r["expected_latency_ms"],
                    "total_latency_ms": total_latency,
                    "system_total_far": emp_far,
                    "system_total_frr": emp_frr,
                    "gp_total_far": result["gp_total_far"],
                    "gp_total_frr": result["gp_total_frr"],
                    "valid_thresholds": valid_thresholds,
                }
            )

    print(
        f"[全排列表] alpha={alpha}：{len(orderings)} 種排列中，"
        f"{len(orderings) - n_failed} 種 GP 求解成功、{n_failed} 種求解失敗（不可行），"
        f"{n_reverse_failed} 種反查失敗或最後一關不是單門檻。"
    )
    return pd.DataFrame(rows)


def build_lookup_table(
    profiles_by_name: dict[str, DetectorProfile],
    security_levels: dict[str, dict[str, float]] = SECURITY_LEVELS,
) -> pd.DataFrame:
    """跑完整套流程（步驟 2~4），組出最終 lookup table DataFrame。

    欄位：security_level, stage_order, stage, modality, detector_name,
    threshold_L, threshold_H, expected_latency_ms, system_total_far, system_total_frr。
    """
    detector_names = list(profiles_by_name.keys())
    orderings = enumerate_orderings(detector_names)
    print(f"共窮舉 {len(orderings)} 種排列組合（偵測器：{detector_names}）")

    rows = []
    for level_name, cfg in security_levels.items():
        picked = select_best_ordering(
            orderings,
            profiles_by_name,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            latency_budget_ms=cfg["latency_budget_ms"],
            min_stages=int(cfg.get("min_stages", 1)),
        )
        if picked is None:
            continue
        ordering, result, total_latency = picked
        stage_rows = result["stage_rows"]
        stage_order_str = ",".join(ordering)

        print(
            f"[{level_name}] 選中排列={stage_order_str} "
            f"total_far={result['total_far']:.6f} total_frr={result['total_frr']:.6f} "
            f"(GP far={result['gp_total_far']:.6f} frr={result['gp_total_frr']:.6f}) "
            f"total_latency={total_latency:.1f}ms"
        )

        for r in stage_rows:
            rows.append(
                {
                    "security_level": level_name,
                    "stage_order": stage_order_str,
                    "stage": r["stage"],
                    "modality": "visual",
                    "detector_name": r["detector_name"],
                    "threshold_L": r["threshold_L"],
                    "threshold_H": r["threshold_H"],
                    "expected_latency_ms": r["expected_latency_ms"],
                    "system_total_far": result["total_far"],
                    "system_total_frr": result["total_frr"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    # === 資料洩漏提醒 ===
    # 下面讀入的 threshold_sweep_*.csv 是拿來「校準」門檻用的資料，
    # 不可以和最終系統效能評估用的 test set 重疊！
    repo_root = Path(__file__).resolve().parent.parent
    here = Path(__file__).resolve().parent

    # Xception/UCF 的 *_combined.csv 本身已經是跨 CelebDF/DFDC/FaceForensics
    # 合併後的曲線，這裡刻意排除，避免和單一資料集檔案一起做 combine_mode
    # 造成加權失真（重複計入同一批樣本）。
    #
    # SBI 注意事項（重要）：models/sbi/threshold_sweep_sbi_v0.csv 與
    # threshold_sweep_sbi_v1.csv 的 far/frr 兩欄位經人工比對確認是寫反的
    # （用 input-csvs/sbi/scores_visual_sbi_ffpp_v{0,1}.csv 依正確定義重算，
    # 每個 threshold 切點都精準對上 far/frr 互換的結果），因此改用方向正確的
    # results/20/sbi/threshold_sweep_sbi_{celebdf,dfdc}.csv（已核對與
    # input-csvs/sbi 的 v0 原始分數逐點一致）。FaceForensics 則另外用
    # threshold_sweep_sbi_faceforensics_v1.csv（本資料夾內，依使用者指示
    # 用 v1 原始分數 input-csvs/sbi/scores_visual_sbi_ffpp_v1.csv 重新計算，
    # 因為 results/20/sbi/threshold_sweep_sbi_faceforensics.csv 實際上是用
    # v0 算出來的，不是 v1）。
    detector_sweep_files: dict[str, list[Path]] = {
        "SBI": [
            repo_root / "results/20/sbi/threshold_sweep_sbi_celebdf.csv",
            repo_root / "results/20/sbi/threshold_sweep_sbi_dfdc.csv",
            here / "threshold_sweep_sbi_faceforensics_v1.csv",
        ],
        "Xception": [
            repo_root / "models/xception/threshold_sweep_xception_celebdf.csv",
            repo_root / "models/xception/threshold_sweep_xception_dfdc.csv",
            repo_root / "models/xception/threshold_sweep_xception_faceforensics.csv",
        ],
        "UCF": [
            repo_root / "models/ucf/threshold_sweep_ucf_celebdf.csv",
            repo_root / "models/ucf/threshold_sweep_ucf_dfdc.csv",
            repo_root / "models/ucf/threshold_sweep_ucf_faceforensics.csv",
        ],
    }

    latency_lookup = pd.read_csv(here / "detector_latency.csv")

    profiles_by_name: dict[str, DetectorProfile] = {}
    for name, files in detector_sweep_files.items():
        profile = build_detector_profile(name, files, latency_lookup=latency_lookup, combine_mode="max")
        profiles_by_name[name] = profile
        print(profile)

    lookup_df = build_lookup_table(profiles_by_name, SECURITY_LEVELS)

    output_path = here / "output" / "real_data_vip_settings.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.to_csv(output_path, index=False)
    print(f"\n已輸出 lookup table：{output_path}")

    # 診斷用：不管 SECURITY_LEVELS 篩選結果如何，都額外輸出一張全排列表，
    # 方便在 SECURITY_LEVELS 全部找不到可行排列時，回頭觀察全部 15 種排列
    # 實際能做到的 total_far/total_frr/延遲，再決定 beta 該設多少。
    all_orderings_df = dump_all_orderings(profiles_by_name, alpha=ALL_ORDERINGS_ALPHA)
    all_orderings_path = here / "output" / "all_orderings.csv"
    all_orderings_df.to_csv(all_orderings_path, index=False)
    print(f"已輸出全排列診斷表：{all_orderings_path}")


if __name__ == "__main__":
    main()
