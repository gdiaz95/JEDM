"""
Step 0 – Ground Truth Regression.

For each real dataset: fit OLS, record β_real, SE_real, CI_real (95 %).
Results are saved to reports/ground_truth/<dataset>.json.
These are the reference coefficients against which synthetic-data regressions
will be compared in subsequent steps.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
META_DIR = ROOT / ".metadata"
REPORT_DIR = ROOT / "reports" / "ground_truth"

# Columns to drop before fitting (leakage / mostly-NaN)
DROP_COLS: dict[str, list[str]] = {
    "students_oulad.csv": ["date_unregistration"],
}

# Threshold: drop any feature column where NaN fraction exceeds this value
NAN_DROP_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = [
    {"meta": "student_dropout_success.json", "delimiter": ";", "encoding": "utf-8-sig"},
    {"meta": "student_performance.json",     "delimiter": ";", "encoding": "utf-8-sig"},
    {"meta": "student_satisfaction.json",    "delimiter": ",", "encoding": "utf-8-sig"},
    {"meta": "students_oulad.json",          "delimiter": ",", "encoding": "utf-8"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata(meta_file: str) -> dict:
    with open(META_DIR / meta_file) as f:
        return json.load(f)


def load_dataset(meta: dict, delimiter: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / meta["filename"], sep=delimiter, encoding=encoding)
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    return df


def preprocess_target(df: pd.DataFrame, target_col: str, filename: str) -> pd.DataFrame:
    """Dataset-specific target transformations before encoding."""
    if filename == "student_satisfaction.csv":
        # Column contains strings like "3.50 / 70.00" — extract the percentage
        df = df.copy()
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.extract(r"/\s*([\d.]+)")[0]
            .astype(float)
        )
    return df


def encode_dataframe(df: pd.DataFrame, target_col: str, filename: str):
    """
    Encode a DataFrame for OLS.
    - Columns with NaN fraction > NAN_DROP_THRESHOLD are dropped.
    - Categorical features  → OrdinalEncoder
    - Categorical target    → LabelEncoder (integer codes)
    - Numeric columns       → passed through
    Returns X, y, feature_names, target_classes.
    """
    # Align target column name after strip
    col_map = {c.strip(): c for c in df.columns}
    target_col = col_map.get(target_col.strip(), target_col)

    # Drop explicitly blacklisted columns
    drop = DROP_COLS.get(filename, [])
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # Apply target-specific preprocessing
    df = preprocess_target(df, target_col, filename)

    y_raw = df[target_col].copy()
    X_df  = df.drop(columns=[target_col]).copy()

    # Drop columns entirely NaN or above NaN threshold
    nan_frac = X_df.isna().mean()
    high_nan = nan_frac[nan_frac > NAN_DROP_THRESHOLD].index.tolist()
    if high_nan:
        print(f"  Dropping high-NaN columns (>{NAN_DROP_THRESHOLD:.0%}): {high_nan}")
    X_df = X_df.drop(columns=high_nan)
    X_df = X_df.dropna(axis=1, how="all")

    # Encode target
    if not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        target_classes = list(le.classes_)
    else:
        y = y_raw.to_numpy(dtype=float)
        target_classes = None

    # Separate feature types — include PyArrow-backed strings ("str" dtype in pandas 3+)
    cat_cols = X_df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()

    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = oe.fit_transform(X_df[cat_cols].astype(str))
        X_cat_df = pd.DataFrame(X_cat, columns=cat_cols, index=X_df.index)
    else:
        X_cat_df = pd.DataFrame(index=X_df.index)

    X_full = pd.concat([X_df[num_cols], X_cat_df], axis=1)

    # Drop rows with remaining NaN in features or target
    mask = ~(X_full.isna().any(axis=1) | pd.isna(y))
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with NaN values")
    X_full = X_full[mask]
    y = y[mask]

    return X_full.to_numpy(dtype=float), y, X_full.columns.tolist(), target_classes


def fit_ols(X: np.ndarray, y: np.ndarray, feature_names: list):
    X_const = sm.add_constant(X, has_constant="add")
    col_names = ["const"] + feature_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.OLS(y, pd.DataFrame(X_const, columns=col_names)).fit()


def extract_ground_truth(result, feature_names: list) -> dict:
    ci = result.conf_int(alpha=0.05)
    names = ["const"] + feature_names
    coefficients = {
        name: {
            "beta":        float(result.params[name]),
            "se":          float(result.bse[name]),
            "t_stat":      float(result.tvalues[name]),
            "p_value":     float(result.pvalues[name]),
            "ci_lower_95": float(ci.loc[name, 0]),
            "ci_upper_95": float(ci.loc[name, 1]),
        }
        for name in names
    }
    return {
        "n_obs":         int(result.nobs),
        "n_features":    len(feature_names),
        "r_squared":     float(result.rsquared),
        "adj_r_squared": float(result.rsquared_adj),
        "f_statistic":   float(result.fvalue),
        "f_pvalue":      float(result.f_pvalue),
        "coefficients":  coefficients,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for cfg in DATASETS:
        meta     = load_metadata(cfg["meta"])
        filename = meta["filename"]
        name     = Path(filename).stem
        target   = meta["target_column"]

        print(f"\n{'='*60}")
        print(f"Dataset : {filename}")
        print(f"Target  : {target}")

        df = load_dataset(meta, cfg["delimiter"], cfg["encoding"])
        X, y, feature_names, target_classes = encode_dataframe(df, target, filename)

        print(f"Shape   : {X.shape[0]} obs × {X.shape[1]} features")
        if target_classes:
            print(f"Target encoding: {dict(enumerate(target_classes))}")

        result = fit_ols(X, y, feature_names)
        gt     = extract_ground_truth(result, feature_names)

        report = {
            "dataset":        filename,
            "target_column":  target,
            "target_classes": target_classes,
            **gt,
        }

        out_path = REPORT_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"R²      : {gt['r_squared']:.4f}  (adj. {gt['adj_r_squared']:.4f})")
        print(f"F-stat  : {gt['f_statistic']:.4f}  p={gt['f_pvalue']:.4e}")
        print(f"Saved   : {out_path.relative_to(ROOT)}")

        summary[name] = {
            "n_obs":         gt["n_obs"],
            "n_features":    gt["n_features"],
            "r_squared":     gt["r_squared"],
            "adj_r_squared": gt["adj_r_squared"],
            "f_statistic":   gt["f_statistic"],
            "f_pvalue":      gt["f_pvalue"],
        }

    summary_path = REPORT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path.relative_to(ROOT)}")
    return summary


if __name__ == "__main__":
    run_all()
