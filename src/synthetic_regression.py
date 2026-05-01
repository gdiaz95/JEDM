"""
Step 1 – Synthetic Regression Bootstrap (B = 1000) × ε sweep + Analysis.

For each (ε, dataset):
  1. Preprocess real data (same pipeline as ground_truth.py).
  2. Fit NPGC **once** with that ε on clean real DataFrame.
  3. For b = 1 … B:
       - Sample one synthetic DataFrame of size n.
       - Apply real-data-fitted encoders to the synthetic sample.
       - Fit OLS → record β̂_syn^(b), SE_syn^(b), CI_naive^(b).

Then computes per coefficient:
  - Bias_j         = mean_b(β̂_syn_j^(b)) − β_real_j
  - Coverage_naive = (1/B) Σ_b 𝟙[β_real_j ∈ CI_naive^(b)]

After all ε values are done, prints a summary table:

  ε        |  Bias   | Coverage naive
  ---------|---------|---------------
  0.1      | -0.031  |   61.0 %
  ...
  ∞ (no DP)| -0.004  |   93.0 %

Outputs:
  reports/synthetic_regression/{eps_label}/{dataset}.json  — raw arrays
  reports/analysis/{eps_label}/{dataset}.json              — bias + coverage
  reports/analysis/epsilon_sweep_table.json                — summary table
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

sys.path.insert(0, str(Path(__file__).parent))
from npgc_local import NPGC_local

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
META_DIR   = ROOT / ".metadata"
GT_DIR     = ROOT / "reports" / "ground_truth"
REPORT_DIR = ROOT / "reports" / "synthetic_regression"
ANAL_DIR   = ROOT / "reports" / "analysis"

B = 1000

# ε = None means no differential privacy (∞ budget)
EPSILONS: list[float | None] = [0.1, 0.5, 1.0, 2.0, 5.0, None]

DROP_COLS: dict[str, list[str]] = {
    "students_oulad.csv": ["date_unregistration"],
}
NAN_DROP_THRESHOLD = 0.30

DATASETS = [
    {"meta": "student_dropout_success.json", "delimiter": ";", "encoding": "utf-8-sig"},
    {"meta": "student_performance.json",     "delimiter": ";", "encoding": "utf-8-sig"},
    {"meta": "student_satisfaction.json",    "delimiter": ",", "encoding": "utf-8-sig"},
    {"meta": "students_oulad.json",          "delimiter": ",", "encoding": "utf-8"},
]


def eps_label(epsilon: float | None) -> str:
    return f"eps{epsilon}" if epsilon is not None else "eps_inf"


# ---------------------------------------------------------------------------
# Preprocessing — mirrors ground_truth.py exactly
# ---------------------------------------------------------------------------

def load_metadata(meta_file: str) -> dict:
    with open(META_DIR / meta_file) as f:
        return json.load(f)


def load_and_clean(meta: dict, delimiter: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / meta["filename"], sep=delimiter, encoding=encoding)
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    return df


def preprocess_target(df: pd.DataFrame, target_col: str, filename: str) -> pd.DataFrame:
    if filename == "student_satisfaction.csv":
        # Skip if already numeric (e.g. synthetic data sampled from a float column)
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            df = df.copy()
            df[target_col] = (
                df[target_col]
                .astype(str)
                .str.extract(r"/\s*([\d.]+)")[0]
                .astype(float)
            )
    return df


def drop_high_nan_cols(df: pd.DataFrame, threshold: float = NAN_DROP_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    nan_frac = df.isna().mean()
    drop = nan_frac[nan_frac > threshold].index.tolist()
    return df.drop(columns=drop), drop


class DatasetEncoders:
    """Encoders fitted on real data, reused for every synthetic sample."""

    def __init__(self):
        self.oe: OrdinalEncoder | None = None
        self.le: LabelEncoder | None = None
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.target_col: str = ""
        self.target_classes: list[str] | None = None
        self.feature_names: list[str] = []
        self.high_nan_cols: list[str] = []

    def fit(self, df: pd.DataFrame, target_col: str, filename: str):
        col_map = {c.strip(): c for c in df.columns}
        self.target_col = col_map.get(target_col.strip(), target_col)

        drop_explicit = DROP_COLS.get(filename, [])
        df = df.drop(columns=[c for c in drop_explicit if c in df.columns])
        df = preprocess_target(df, self.target_col, filename)

        y_raw = df[self.target_col]
        X_df  = df.drop(columns=[self.target_col])

        X_df, self.high_nan_cols = drop_high_nan_cols(X_df)
        X_df = X_df.dropna(axis=1, how="all")

        if not pd.api.types.is_numeric_dtype(y_raw):
            self.le = LabelEncoder()
            self.le.fit(y_raw.dropna().astype(str))
            self.target_classes = list(self.le.classes_)
        else:
            self.target_classes = None

        self.cat_cols = X_df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        self.num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

        if self.cat_cols:
            self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.oe.fit(X_df[self.cat_cols].astype(str))

        self.feature_names = self.num_cols + self.cat_cols

    def transform(self, df: pd.DataFrame, filename: str) -> tuple[np.ndarray, np.ndarray] | None:
        col_map = {c.strip(): c for c in df.columns}
        target_col = col_map.get(self.target_col.strip(), self.target_col)

        drop_explicit = DROP_COLS.get(filename, [])
        df = df.drop(columns=[c for c in drop_explicit if c in df.columns], errors="ignore")
        df = preprocess_target(df, target_col, filename)

        if target_col not in df.columns:
            return None

        y_raw = df[target_col]
        X_df  = df.drop(columns=[target_col], errors="ignore")
        X_df  = X_df.drop(columns=[c for c in self.high_nan_cols if c in X_df.columns], errors="ignore")

        if self.le is not None:
            y = self.le.transform(y_raw.fillna("nan").astype(str))
        else:
            y = y_raw.to_numpy(dtype=float)

        X_num = X_df[self.num_cols].to_numpy(dtype=float) if self.num_cols else np.empty((len(X_df), 0))
        X_cat = (
            self.oe.transform(X_df[self.cat_cols].fillna("nan").astype(str))
            if self.cat_cols and self.oe is not None
            else np.empty((len(X_df), 0))
        )
        X = np.hstack([X_num, X_cat])

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.astype(float)))
        X, y = X[mask], y[mask]

        if len(X) < len(self.feature_names) + 2:
            return None
        return X, y


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def fit_ols_silent(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    X_const = sm.add_constant(X, has_constant="add")
    col_names = ["const"] + feature_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.OLS(y.astype(float), pd.DataFrame(X_const, columns=col_names)).fit()


def regression_arrays(result, feature_names: list[str]) -> tuple[list, list, list, list]:
    names    = ["const"] + feature_names
    ci       = result.conf_int(alpha=0.05)
    beta     = [float(result.params[n]) for n in names]
    se       = [float(result.bse[n])    for n in names]
    ci_lower = [float(ci.loc[n, 0])    for n in names]
    ci_upper = [float(ci.loc[n, 1])    for n in names]
    return beta, se, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Analysis: bias + naive coverage
# ---------------------------------------------------------------------------

def compute_bias_and_coverage(syn: dict, gt: dict) -> dict:
    feature_names = syn["feature_names"]
    n_ok = syn["n_successful"]

    if n_ok == 0:
        return {}   # no successful reps — caller handles this

    beta     = np.array(syn["beta"],        dtype=float)  # [n_ok, n_coef]
    ci_lower = np.array(syn["ci_lower_95"], dtype=float)
    ci_upper = np.array(syn["ci_upper_95"], dtype=float)

    coef_results: dict[str, dict] = {}
    for j, name in enumerate(feature_names):
        if name not in gt["coefficients"]:
            continue
        beta_real = gt["coefficients"][name]["beta"]
        mean_syn  = float(np.mean(beta[:, j]))
        bias      = mean_syn - beta_real
        covered   = (ci_lower[:, j] <= beta_real) & (beta_real <= ci_upper[:, j])
        coverage  = float(np.mean(covered))
        coef_results[name] = {
            "beta_real":      beta_real,
            "mean_beta_syn":  mean_syn,
            "std_beta_syn":   float(np.std(beta[:, j])),
            "bias":           bias,
            "coverage_naive": coverage,
        }
    return coef_results


def save_analysis(name: str, syn: dict, gt: dict, label: str) -> tuple[float, float]:
    """Save per-coefficient analysis. Returns (mean_bias, mean_coverage_naive)."""
    out_dir = ANAL_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    coef_results = compute_bias_and_coverage(syn, gt)

    if not coef_results:
        print(f"  [WARN] 0 successful reps — bias/coverage set to NaN")
        return float("nan"), float("nan")

    non_const    = {k: v for k, v in coef_results.items() if k != "const"}

    mean_bias     = float(np.mean([v["bias"]             for v in non_const.values()]))
    mean_coverage = float(np.mean([v["coverage_naive"]   for v in non_const.values()]))
    mean_abs_bias = float(np.mean([abs(v["bias"])         for v in non_const.values()]))

    report = {
        "dataset":             gt["dataset"],
        "target_column":       gt["target_column"],
        "epsilon":             syn["epsilon"],
        "B":                   syn["n_successful"],
        "mean_bias":           mean_bias,
        "mean_coverage_naive": mean_coverage,
        "mean_abs_bias":       mean_abs_bias,
        "coefficients":        coef_results,
    }

    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Analysis → bias: {mean_bias:+.4f}  "
          f"coverage_naive: {mean_coverage:.3f}  "
          f"|bias|: {mean_abs_bias:.6f}")

    sorted_cov = sorted(non_const.items(), key=lambda kv: kv[1]["coverage_naive"])
    print("  Worst-covered coefficients:")
    for coef_name, vals in sorted_cov[:5]:
        print(f"    {coef_name:<45s}  "
              f"coverage={vals['coverage_naive']:.3f}  bias={vals['bias']:+.4f}")
    print(f"  Saved : {out_path.relative_to(ROOT)}")

    return mean_bias, mean_coverage


# ---------------------------------------------------------------------------
# Per-dataset bootstrap
# ---------------------------------------------------------------------------

def run_dataset(cfg: dict, epsilon: float | None, label: str) -> tuple[float, float]:
    """
    Run B-sample bootstrap for one dataset at a given epsilon.
    Returns (mean_bias, mean_coverage_naive) across non-intercept coefficients.
    """
    meta     = load_metadata(cfg["meta"])
    filename = meta["filename"]
    name     = Path(filename).stem
    target   = meta["target_column"]

    print(f"\n  Dataset : {filename}  |  ε={epsilon}  |  B={B}")

    # 1. Fit encoders on real data
    df_real = load_and_clean(meta, cfg["delimiter"], cfg["encoding"])
    enc = DatasetEncoders()
    enc.fit(df_real, target, filename)

    result_real = enc.transform(df_real, filename)
    if result_real is None:
        raise RuntimeError(f"Real data encoding failed for {filename}")
    X_real, _ = result_real
    n_rows = X_real.shape[0]
    print(f"  Real shape : {n_rows} obs × {X_real.shape[1]} features")

    # 2. Build clean DataFrame for NPGC
    df_for_npgc = load_and_clean(meta, cfg["delimiter"], cfg["encoding"])
    drop_explicit = DROP_COLS.get(filename, [])
    df_for_npgc = df_for_npgc.drop(columns=[c for c in drop_explicit if c in df_for_npgc.columns])
    df_for_npgc = preprocess_target(df_for_npgc, enc.target_col, filename)
    df_for_npgc = df_for_npgc.drop(columns=[c for c in enc.high_nan_cols if c in df_for_npgc.columns])
    df_for_npgc = df_for_npgc.dropna(subset=[enc.target_col])

    # 3. Fit NPGC once with the given epsilon
    t_fit = time.perf_counter()
    print(f"  Fitting NPGC (ε={epsilon}) … ", end="", flush=True)
    npgc = NPGC_local(epsilon=epsilon)
    npgc.fit(df_for_npgc)
    print(f"done  ({time.perf_counter() - t_fit:.1f}s)")

    # 4. Bootstrap
    all_beta, all_se, all_ci_lo, all_ci_hi = [], [], [], []
    failed: list[int] = []

    t_boot = time.perf_counter()
    for b in range(1, B + 1):
        df_syn = npgc.sample(num_rows=n_rows, seed=b)
        df_syn.columns = df_for_npgc.columns

        encoded = enc.transform(df_syn, filename)
        if encoded is None:
            failed.append(b)
        else:
            X_syn, y_syn = encoded
            try:
                res = fit_ols_silent(X_syn, y_syn, enc.feature_names)
                beta, se, ci_lo, ci_hi = regression_arrays(res, enc.feature_names)
                all_beta.append(beta)
                all_se.append(se)
                all_ci_lo.append(ci_lo)
                all_ci_hi.append(ci_hi)
            except Exception:
                failed.append(b)

        if b % 100 == 0:
            elapsed  = time.perf_counter() - t_boot
            rate     = b / elapsed                        # reps/s
            eta      = (B - b) / rate
            print(f"  [{b:>4}/{B}]  {rate:.1f} reps/s  ETA {eta:>5.0f}s"
                  f"  failures so far: {len(failed)}", flush=True)

    total_boot = time.perf_counter() - t_boot
    n_ok = len(all_beta)
    print(f"  Bootstrap done: {n_ok}/{B} ok  ({len(failed)} failed)  "
          f"total {total_boot:.1f}s  ({total_boot/B*1000:.0f}ms/rep)")

    syn_report = {
        "dataset":        filename,
        "target_column":  target,
        "target_classes": enc.target_classes,
        "epsilon":        epsilon,
        "B":              B,
        "n_successful":   n_ok,
        "n_rows":         n_rows,
        "n_features":     len(enc.feature_names),
        "feature_names":  ["const"] + enc.feature_names,
        "failed_reps":    failed,
        "beta":           all_beta,
        "se":             all_se,
        "ci_lower_95":    all_ci_lo,
        "ci_upper_95":    all_ci_hi,
    }

    syn_dir = REPORT_DIR / label
    syn_dir.mkdir(parents=True, exist_ok=True)
    syn_path = syn_dir / f"{name}.json"
    with open(syn_path, "w") as f:
        json.dump(syn_report, f, indent=2)
    print(f"  Bootstrap saved : {syn_path.relative_to(ROOT)}")

    # 5. Analysis
    gt_path = GT_DIR / f"{name}.json"
    if not gt_path.exists():
        print(f"  [WARN] ground_truth missing, skipping analysis.")
        return float("nan"), float("nan")

    with open(gt_path) as f:
        gt = json.load(f)

    return save_analysis(name, syn_report, gt, label)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_and_save_table(sweep: dict[str, dict[str, tuple[float, float]]]) -> None:
    """
    sweep: { eps_label → { dataset_name → (mean_bias, mean_coverage) } }
    Prints and saves the ε × (bias, coverage) summary table.
    """
    header = f"{'ε':<12} | {'Bias':>8} | {'Coverage naive':>14}"
    sep    = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    rows = []
    for label, datasets in sweep.items():
        biases    = [v[0] for v in datasets.values() if not np.isnan(v[0])]
        coverages = [v[1] for v in datasets.values() if not np.isnan(v[1])]
        mean_bias = float(np.mean(biases))     if biases     else float("nan")
        mean_cov  = float(np.mean(coverages))  if coverages  else float("nan")

        eps_str = "∞ (no DP)" if label == "eps_inf" else label.replace("eps", "ε=")
        cov_pct = f"{mean_cov * 100:.1f} %"
        print(f"{eps_str:<12} | {mean_bias:>+8.4f} | {cov_pct:>14}")

        rows.append({
            "epsilon_label":       label,
            "mean_bias":           mean_bias,
            "mean_coverage_naive": mean_cov,
            "per_dataset":         {ds: {"mean_bias": v[0], "mean_coverage_naive": v[1]}
                                    for ds, v in datasets.items()},
        })

    print(sep)

    table_path = ANAL_DIR / "epsilon_sweep_table.json"
    ANAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSummary table saved : {table_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Logging — tee stdout to a timestamped log file
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror every print() to both the terminal and a log file."""

    def __init__(self, log_path: Path):
        self._terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", buffering=1)   # line-buffered

    def write(self, msg: str):
        self._terminal.write(msg)
        self._file.write(msg)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ANAL_DIR.mkdir(parents=True, exist_ok=True)

    log_path = ROOT / "logs" / f"synthetic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    tee = _Tee(log_path)
    sys.stdout = tee

    try:
        print(f"Run started : {datetime.now().isoformat(timespec='seconds')}")
        print(f"Log file    : {log_path}")
        print(f"B={B}  |  ε values: {EPSILONS}")
        print(f"Datasets    : {[d['meta'] for d in DATASETS]}\n")

        n_total  = len(EPSILONS) * len(DATASETS)
        run_idx  = 0
        t_global = time.perf_counter()

        sweep: dict[str, dict[str, tuple[float, float]]] = {}

        for eps_i, epsilon in enumerate(EPSILONS, 1):
            label   = eps_label(epsilon)
            eps_str = str(epsilon) if epsilon is not None else "∞ (no DP)"
            print(f"\n{'#'*60}")
            print(f"# ε = {eps_str}  [{eps_i}/{len(EPSILONS)}]")
            print(f"{'#'*60}")

            sweep[label] = {}
            for ds_i, cfg in enumerate(DATASETS, 1):
                run_idx += 1
                meta = load_metadata(cfg["meta"])
                name = Path(meta["filename"]).stem

                elapsed_g = time.perf_counter() - t_global
                print(f"\n--- Run {run_idx}/{n_total}  |  "
                      f"ε={eps_str}  |  dataset {ds_i}/{len(DATASETS)}  |  "
                      f"global elapsed {elapsed_g:.0f}s ---")

                t_ds = time.perf_counter()
                mean_bias, mean_cov = run_dataset(cfg, epsilon=epsilon, label=label)
                ds_time = time.perf_counter() - t_ds

                sweep[label][name] = (mean_bias, mean_cov)
                print(f"  Dataset done in {ds_time:.0f}s")

        elapsed_total = time.perf_counter() - t_global
        print(f"\n{'='*60}")
        print(f"All {n_total} runs complete in {elapsed_total:.0f}s "
              f"({elapsed_total/60:.1f} min)")
        print(f"Run finished : {datetime.now().isoformat(timespec='seconds')}")

        print_and_save_table(sweep)
        print("\nAll done.")

    finally:
        sys.stdout = tee._terminal
        tee.close()
        print(f"Log saved : {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    run_all()
