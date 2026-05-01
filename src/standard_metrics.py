"""
Standard metrics harness — one JSON per (synthesizer, dataset).

Groups: Fidelity (6), Utility / TSTR × 5 classifiers, Privacy (5 + MIA), Runtime.
Output: reports/standard_metrics/{synthesizer}/{dataset}.json
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
from mostlyai import qa

import sys
sys.path.insert(0, str(Path(__file__).parent))
from npgc_local import NPGC_local
from npgc import NPGC
from sklearn.preprocessing import LabelEncoder
from synthetic_regression import (
    load_and_clean, load_metadata,
    preprocess_target, DROP_COLS, NAN_DROP_THRESHOLD,
)

ROOT       = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports" / "standard_metrics"

SEED = 42
IMBALANCED = {"student_dropout_success.csv", "students_oulad.csv"}

SYNTHESIZERS = [
    {
        "name":   "npgc_allfix",
        "cls":    NPGC_local,
        "kwargs": {
            "epsilon":        1.0,
            "marginal_scale": True,
            "corr_formula":   True,
            "corr_noise":     True,
            "corr_delta":     True,
        },
    },
    {
        "name":   "npgc_standard",
        "cls":    NPGC,
        "kwargs": {},   # default epsilon=1.0
    },
]

DATASETS = [
    {"meta": "student_dropout_success.json", "delimiter": ";", "encoding": "utf-8-sig"},
    {"meta": "student_performance.json",     "delimiter": ";", "encoding": "utf-8-sig"},
    {"meta": "student_satisfaction.json",    "delimiter": ",", "encoding": "utf-8-sig"},
    {"meta": "students_oulad.json",          "delimiter": ",", "encoding": "utf-8"},
]

CLASSIFIERS = [
    ("logistic_regression", LogisticRegression(max_iter=1000, random_state=SEED)),
    ("random_forest",       RandomForestClassifier(random_state=SEED)),
    ("xgboost",             XGBClassifier(eval_metric="logloss", random_state=SEED, verbosity=0)),
    ("decision_tree",       DecisionTreeClassifier(random_state=SEED)),
    ("knn",                 KNeighborsClassifier()),
]


# ---------------------------------------------------------------------------
# MIA
# ---------------------------------------------------------------------------

def _encode(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat_cols = real_train.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    num_cols = real_train.select_dtypes(include=[np.number]).columns.tolist()

    X_num = real_train[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(real_train), 0))
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = oe.fit_transform(real_train[cat_cols].fillna("nan").astype(str))
        X_train = np.hstack([X_num, X_cat])

        def _apply(df: pd.DataFrame) -> np.ndarray:
            n = df[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(df), 0))
            c = oe.transform(df[cat_cols].fillna("nan").astype(str))
            return np.nan_to_num(np.hstack([n, c]))

        X_holdout = _apply(real_holdout)
        X_syn     = _apply(synthetic_df)
    else:
        X_train   = np.nan_to_num(X_num)
        X_holdout = np.nan_to_num(real_holdout[num_cols].to_numpy(dtype=float))
        X_syn     = np.nan_to_num(synthetic_df[num_cols].to_numpy(dtype=float))

    X_train = np.nan_to_num(X_train)
    return X_train, X_holdout, X_syn


def run_mia(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    structured_columns: list[str] | None = None,
) -> dict:
    cols = structured_columns or list(real_train.columns)
    cols = [c for c in cols if c in real_train.columns and c in synthetic_df.columns]

    X_train, X_holdout, X_syn = _encode(
        real_train[cols], real_holdout[cols], synthetic_df[cols]
    )
    X_real = np.vstack([X_train, X_holdout])
    labels = np.array([1] * len(X_train) + [0] * len(X_holdout))

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=1)
    nn.fit(X_syn)
    dists = nn.kneighbors(X_real)[0][:, 0]

    auroc = float(roc_auc_score(labels, -dists))
    return {
        "mia_auroc":        round(auroc, 4),
        "n_train":          int(len(real_train)),
        "n_holdout":        int(len(real_holdout)),
        "n_synthetic":      int(len(synthetic_df)),
        "mean_dist_train":  round(float(dists[labels == 1].mean()), 4),
        "mean_dist_holdout": round(float(dists[labels == 0].mean()), 4),
        "columns_used":     cols,
    }


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def run_fidelity(metrics_qa, synthetic_df: pd.DataFrame, real_train: pd.DataFrame) -> dict:
    sdv_meta = SingleTableMetadata()
    sdv_meta.detect_from_dataframe(real_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quality = evaluate_quality(real_train, synthetic_df, sdv_meta, verbose=False)

    props = dict(zip(quality.get_properties()["Property"], quality.get_properties()["Score"]))

    return {
        "overall_accuracy":    round(float(metrics_qa.accuracy.overall), 4),
        "univariate_accuracy": round(float(metrics_qa.accuracy.univariate), 4),
        "bivariate_accuracy":  round(float(metrics_qa.accuracy.bivariate), 4),
        "sdv_overall_score":   round(float(quality.get_score()), 4),
        "column_shapes":       round(float(props.get("Column Shapes", float("nan"))), 4),
        "column_pair_trends":  round(float(props.get("Column Pair Trends", float("nan"))), 4),
    }


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def run_privacy(
    metrics_qa,
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> dict:
    return {
        "discriminator_auc": round(float(metrics_qa.similarity.discriminator_auc_training_synthetic), 4),
        "dcr_training":      round(float(metrics_qa.distances.dcr_training), 4),
        "dcr_holdout":       round(float(metrics_qa.distances.dcr_holdout), 4),
        "dcr_share":         round(float(metrics_qa.distances.dcr_share), 4),
        "ims_training":      round(float(metrics_qa.distances.ims_training), 4),
        "mia":               run_mia(real_train, real_holdout, synthetic_df),
    }


# ---------------------------------------------------------------------------
# Utility — TSTR × 5 classifiers
# ---------------------------------------------------------------------------

def run_tstr_panel(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    target_col: str,
    filename: str,
) -> dict:
    """
    Self-contained TSTR panel.  No dependency on DatasetEncoders.

    Label strategy
    ──────────────
    Categorical target  → LabelEncoder fitted on real_train; unknown synthetic
                          labels mapped to most-common real class.
    Numeric target      → round to 6 d.p. (kills NPGC float noise), then snap
                          synthetic values to the nearest real-train value so
                          the synthetic label space ⊆ real label space.

    Each training set (real / synthetic) gets its own LabelEncoder so the
    labels passed to XGBoost are always exactly {0, …, k-1}.  The holdout is
    filtered per classifier to classes seen during training.
    """
    # ── 1. features / target split ────────────────────────────────────────
    feat_cols = [c for c in real_train.columns if c != target_col]
    cat_cols  = real_train[feat_cols].select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()
    num_cols  = [c for c in feat_cols if c not in cat_cols]

    # ── 2. feature encoding (fit on real_train only) ──────────────────────
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        oe.fit(real_train[cat_cols].fillna("nan").astype(str))

    def get_X(df: pd.DataFrame) -> np.ndarray:
        Xn = df[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(df), 0))
        if cat_cols:
            Xc = oe.transform(df[cat_cols].fillna("nan").astype(str))
            return np.nan_to_num(np.hstack([Xn, Xc]))
        return np.nan_to_num(Xn)

    X_rt = get_X(real_train)
    X_rh = get_X(real_holdout)
    X_sd = get_X(synthetic_df)

    # ── 3. label encoding ─────────────────────────────────────────────────
    y_rt_raw = real_train[target_col]
    y_rh_raw = real_holdout[target_col]
    y_sd_raw = synthetic_df[target_col]

    if not pd.api.types.is_numeric_dtype(y_rt_raw):
        le_cat = LabelEncoder()
        le_cat.fit(y_rt_raw.dropna().astype(str))

        def _cat_enc(series) -> np.ndarray:
            s = series.fillna("__nan__").astype(str).to_numpy()
            s[~np.isin(s, le_cat.classes_)] = le_cat.classes_[0]
            return le_cat.transform(s).astype(float)

        y_rt = _cat_enc(y_rt_raw)
        y_rh = _cat_enc(y_rh_raw)
        y_sd = _cat_enc(y_sd_raw)
    else:
        y_rt  = np.round(y_rt_raw.to_numpy(dtype=float), 6)
        y_rh  = np.round(y_rh_raw.to_numpy(dtype=float), 6)
        y_sd_f = np.round(y_sd_raw.to_numpy(dtype=float), 6)
        real_unique = np.unique(y_rt[~np.isnan(y_rt)])
        if len(real_unique) == 0:
            return {}
        snap_idx = np.argmin(np.abs(real_unique[:, None] - y_sd_f[None, :]), axis=0)
        y_sd = real_unique[snap_idx]

    # ── 4. drop NaN rows ──────────────────────────────────────────────────
    def _clean(X: np.ndarray, y: np.ndarray):
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.astype(float)))
        return X[mask], y[mask]

    X_rt, y_rt = _clean(X_rt, y_rt)
    X_rh, y_rh = _clean(X_rh, y_rh)
    X_sd, y_sd = _clean(X_sd, y_sd)

    if len(X_rt) == 0 or len(X_sd) == 0 or len(X_rh) == 0:
        return {}

    # ── 5. per-split LabelEncoder → guarantees {0,…,k-1} for XGBoost ─────
    le_r = LabelEncoder().fit(y_rt)
    le_s = LabelEncoder().fit(y_sd)

    y_rt_enc = le_r.transform(y_rt)
    y_sd_enc = le_s.transform(y_sd)

    mask_r = np.isin(y_rh, le_r.classes_)
    mask_s = np.isin(y_rh, le_s.classes_)
    if not mask_r.any() or not mask_s.any():
        return {}

    X_rh_r, y_rh_r = X_rh[mask_r], le_r.transform(y_rh[mask_r])
    X_rh_s, y_rh_s = X_rh[mask_s], le_s.transform(y_rh[mask_s])

    # ── 6. run classifiers ────────────────────────────────────────────────
    is_imbalanced = filename in IMBALANCED
    results: dict = {}
    syn_scores, gaps = [], []

    for clf_name, clf_template in CLASSIFIERS:
        clf_r = clone(clf_template)
        clf_s = clone(clf_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_r.fit(X_rt, y_rt_enc)
            clf_s.fit(X_sd, y_sd_enc)

        preds_r = clf_r.predict(X_rh_r)
        preds_s = clf_s.predict(X_rh_s)

        real_acc = float(accuracy_score(y_rh_r, preds_r))
        syn_acc  = float(accuracy_score(y_rh_s, preds_s))
        gap      = (real_acc - syn_acc) / real_acc * 100 if real_acc != 0 else float("inf")

        entry = {
            "real_accuracy":      round(real_acc, 4),
            "synthetic_accuracy": round(syn_acc,  4),
            "gap_pct":            round(gap, 4),
        }
        if is_imbalanced:
            entry["real_f1_macro"]      = round(float(f1_score(y_rh_r, preds_r, average="macro", zero_division=0)), 4)
            entry["synthetic_f1_macro"] = round(float(f1_score(y_rh_s, preds_s, average="macro", zero_division=0)), 4)

        results[clf_name] = entry
        syn_scores.append(syn_acc)
        gaps.append(gap)

    results["panel_summary"] = {
        "mean_synthetic_accuracy": round(float(np.mean(syn_scores)), 4),
        "std_synthetic_accuracy":  round(float(np.std(syn_scores)),  4),
        "mean_gap_pct":            round(float(np.mean(gaps)), 4),
        "std_gap_pct":             round(float(np.std(gaps)),  4),
    }
    return results


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(ds_cfg: dict, syn_cfg: dict) -> None:
    meta     = load_metadata(ds_cfg["meta"])
    filename = meta["filename"]
    name     = Path(filename).stem
    target   = meta["target_column"]

    out_path = REPORT_DIR / syn_cfg["name"] / f"{name}.json"
    if out_path.exists():
        print(f"  [SKIP] {syn_cfg['name']}/{name}")
        return

    print(f"\n  {filename}  |  {syn_cfg['name']}")

    # Load + clean
    df = load_and_clean(meta, ds_cfg["delimiter"], ds_cfg["encoding"])
    df = df.drop(columns=[c for c in DROP_COLS.get(filename, []) if c in df.columns])
    df = preprocess_target(df, target, filename)
    high_nan = df.isna().mean()
    df = df.drop(columns=high_nan[high_nan > NAN_DROP_THRESHOLD].index.tolist())
    df = df.dropna(subset=[target])

    # 80/20 split stratified on target
    try:
        train_df, holdout_df = train_test_split(
            df, test_size=0.2, random_state=SEED + 1, stratify=df[target]
        )
    except ValueError:
        train_df, holdout_df = train_test_split(
            df, test_size=0.2, random_state=SEED + 1
        )

    n_synthetic = min(1000, len(train_df))
    print(f"    train={len(train_df)}  holdout={len(holdout_df)}  synthetic={n_synthetic}")

    # Fit
    t0 = time.perf_counter()
    syn_model = syn_cfg["cls"](**syn_cfg["kwargs"])
    syn_model.fit(train_df)
    training_time = time.perf_counter() - t0
    print(f"    Fitted in {training_time:.1f}s")

    # Sample
    t1 = time.perf_counter()
    synthetic_df = syn_model.sample(num_rows=n_synthetic)
    synthetic_df.columns = train_df.columns
    sampling_time = time.perf_counter() - t1
    print(f"    Sampled in {sampling_time:.1f}s")

    # mostlyai.qa — shared call for fidelity + privacy
    print("    mostlyai.qa ...", end="", flush=True)
    _, metrics_qa = qa.report(
        syn_tgt_data=synthetic_df,
        trn_tgt_data=train_df,
        hol_tgt_data=holdout_df,
    )
    if os.path.exists("model-report.html"):
        os.remove("model-report.html")
    print(" done")

    print("    SDV fidelity ...", end="", flush=True)
    fidelity = run_fidelity(metrics_qa, synthetic_df, train_df)
    print(" done")

    privacy = run_privacy(metrics_qa, train_df, holdout_df, synthetic_df)

    print("    TSTR panel ...", end="", flush=True)
    utility = run_tstr_panel(train_df, holdout_df, synthetic_df, target, filename)
    print(" done")

    report = {
        "dataset":     filename,
        "synthesizer": syn_cfg["name"],
        "n_train":     len(train_df),
        "n_holdout":   len(holdout_df),
        "n_synthetic": n_synthetic,
        "fidelity":    fidelity,
        "utility":     utility,
        "privacy":     privacy,
        "runtime": {
            "training_time_s": round(training_time, 2),
            "sampling_time_s": round(sampling_time, 2),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Saved: {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for syn_cfg in SYNTHESIZERS:
        print(f"\n{'='*60}\nSynthesizer: {syn_cfg['name']}\n{'='*60}")
        for ds_cfg in DATASETS:
            run_dataset(ds_cfg, syn_cfg)
    print("\nDone.")


if __name__ == "__main__":
    run_all()
