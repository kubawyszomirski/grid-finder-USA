#!/usr/bin/env python3
"""
preprocess_tiles_tabular.py
Converts generate-tiles output into clean tabular datasets ready for ML.

Reads:
    {project}/{tiles_folder}/training_tiles.gpkg
    {project}/{tiles_folder}/prediction_tiles.gpkg   (optional)

Writes to {project}/{tiles_folder}/preprocessed/:
    X_train.parquet        — scaled training features
    y_train.parquet        — grid labels (0 / 1)
    meta_train.parquet     — tile_id / state for training rows
    X_val.parquet          — scaled validation features
    y_val.parquet
    meta_val.parquet
    X_predict.parquet      — scaled prediction features
    meta_predict.parquet   — tile_id / state for prediction rows
    scaler.pkl             — fitted RobustScaler (for inference)
    feature_config.json    — column lists, transform flags, split stats
    report.txt             — label balance, top correlations, NaN summary

Usage:
    python preprocess_tiles_tabular.py project-2 500_tiles
    python preprocess_tiles_tabular.py project-2 500_tiles --no-scale
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ==========================================
#            CONFIGURATION
# ==========================================

BASE_DIR     = Path(__file__).parent.parent
LABEL_COL    = "grid"
RANDOM_STATE = 42

# Columns that are never used as ML features
META_COLS = {"tile_id", "state", "geometry"}


# ==========================================
#              HELPERS
# ==========================================

def load_tiles(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [not found] {path}")
        return None
    gdf = gpd.read_file(path)
    df  = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    print(f"  {path.name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def _is_binary(series: pd.Series) -> bool:
    """True if a numeric column contains only 0 / 1 (ignoring NaN)."""
    unique = set(series.dropna().unique())
    return unique.issubset({0, 1, 0.0, 1.0})


def classify_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list, list, list]:
    """
    Split feature columns into three groups:
      binary  — 0/1 flags            → no transform, no scale benefit
      pct     — percentage (0–100)    → no log, but scale
      continuous — everything else   → log1p then scale
    """
    binary, pct, continuous = [], [], []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _is_binary(df[col]):
            binary.append(col)
        elif "pct" in col.lower() or "score" in col.lower():
            pct.append(col)
        else:
            continuous.append(col)
    return binary, pct, continuous


def nan_report(df: pd.DataFrame, feature_cols: list[str]) -> str:
    missing = df[feature_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return "  No missing values.\n"
    lines = []
    for col, n in missing.items():
        lines.append(f"  {col:<45s} {n:>6,}  ({n/len(df)*100:.1f}%)")
    return "\n".join(lines) + "\n"


def correlation_report(X: pd.DataFrame, y: pd.Series, top_n: int = 25) -> str:
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    lines = []
    for feat, val in corr.head(top_n).items():
        bar = "█" * int(val * 30)
        lines.append(f"  {feat:<45s}  {val:.4f}  {bar}")
    return "\n".join(lines)


def build_report(
    df_train_raw: pd.DataFrame,
    feature_cols: list[str],
    binary_cols: list[str],
    pct_cols: list[str],
    continuous_cols: list[str],
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_predict: pd.DataFrame,
) -> str:
    sep = "=" * 62
    lines = [
        sep,
        "  TILE PREPROCESSING REPORT",
        sep,
        f"  Training rows  : {len(X_all):,}",
        f"  Predict rows   : {len(X_predict):,}",
        "",
        f"  Total features : {len(feature_cols)}",
        f"    binary       : {len(binary_cols)}",
        f"    percentage   : {len(pct_cols)}",
        f"    continuous   : {len(continuous_cols)}  (log1p transformed)",
        "",
        f"  Label balance  : {y_all.value_counts().to_dict()}",
        f"  Positive rate  : {y_all.mean():.3f}",
        "",
        "-" * 62,
        "  NaN SUMMARY (original training set, before fill):",
        "-" * 62,
        nan_report(df_train_raw, feature_cols),
        "-" * 62,
        f"  TOP {min(25, len(feature_cols))} FEATURES BY CORRELATION WITH LABEL (post-transform):",
        "-" * 62,
        correlation_report(X_all, y_all),
    ]
    return "\n".join(lines)


# ==========================================
#           PREPROCESSING PIPELINE
# ==========================================

def preprocess(
    df_train: pd.DataFrame,
    df_predict: pd.DataFrame | None,
    apply_log: bool = True,
    apply_scale: bool = True,
) -> dict:
    """
    Full pipeline:
      1. Separate metadata, label, and features
      2. Report NaN (before fill)
      3. Fill NaN → 0
      4. Log1p-transform continuous features (optional)
      5. RobustScaler fit on all training data, applied to predict (optional)

    No train/val split — K-fold CV in the train script handles validation.
    """
    if LABEL_COL not in df_train.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found in training tiles.\n"
            "  → training_tiles.gpkg must be generated with --train mode."
        )

    # ---- 1. Split columns ----
    feature_cols = [
        c for c in df_train.columns
        if c not in META_COLS and c != LABEL_COL and pd.api.types.is_numeric_dtype(df_train[c])
    ]

    meta_all   = df_train[[c for c in ("tile_id", "state") if c in df_train.columns]].copy()
    y_all      = df_train[LABEL_COL].fillna(0).astype(int)
    df_nan_ref = df_train[feature_cols].copy()  # keep pre-fill for NaN report

    if df_predict is not None:
        meta_predict = df_predict[[c for c in ("tile_id", "state") if c in df_predict.columns]].copy()
        X_pred_raw   = df_predict.reindex(columns=feature_cols, fill_value=0).copy()
    else:
        meta_predict = pd.DataFrame()
        X_pred_raw   = pd.DataFrame(columns=feature_cols)

    # ---- 2. Classify columns ----
    binary_cols, pct_cols, continuous_cols = classify_columns(df_train, feature_cols)

    # ---- 3. Fill NaN ----
    X_all_raw  = df_train[feature_cols].fillna(0).copy()
    X_pred_raw = X_pred_raw.fillna(0)

    # ---- 4. Log1p on continuous features ----
    if apply_log and continuous_cols:
        X_all_raw[continuous_cols]  = np.log1p(X_all_raw[continuous_cols].clip(lower=0))
        X_pred_raw[continuous_cols] = np.log1p(X_pred_raw[continuous_cols].clip(lower=0))

    # ---- 5. Scale ----
    if apply_scale:
        scaler = RobustScaler()
        X_all_scaled  = pd.DataFrame(
            scaler.fit_transform(X_all_raw),
            columns=feature_cols, index=X_all_raw.index,
        )
        X_pred_scaled = pd.DataFrame(
            scaler.transform(X_pred_raw),
            columns=feature_cols, index=X_pred_raw.index,
        )
    else:
        scaler        = None
        X_all_scaled  = X_all_raw
        X_pred_scaled = X_pred_raw

    return {
        "X_all":          X_all_scaled,
        "y_all":          y_all,
        "meta_all":       meta_all,
        "X_predict":      X_pred_scaled,
        "meta_predict":   meta_predict,
        "scaler":         scaler,
        "feature_cols":   feature_cols,
        "binary_cols":    binary_cols,
        "pct_cols":       pct_cols,
        "continuous_cols": continuous_cols,
        "df_train_raw":   df_nan_ref,
    }


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess tile features into ML-ready tabular format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preprocess_tiles_tabular.py project-2 500_tiles\n"
            "  python preprocess_tiles_tabular.py project-2 500_tiles --val-frac 0.15\n"
            "  python preprocess_tiles_tabular.py project-2 500_tiles --no-scale\n"
        ),
    )
    parser.add_argument("project",   help="Project folder name, e.g. 'project-2'")
    parser.add_argument("tiles_dir", help="Tile folder inside the project, e.g. '500_tiles'")
    parser.add_argument(
        "--no-log",   action="store_true",
        help="Skip log1p transform on continuous features",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Skip RobustScaler (useful for tree-based models)",
    )
    args = parser.parse_args()

    tiles_path = BASE_DIR / args.project / args.tiles_dir
    if not tiles_path.exists():
        print(f"[ERROR] Tiles directory not found: {tiles_path}")
        return

    out_dir = tiles_path / "preprocessed"
    out_dir.mkdir(exist_ok=True)

    # ---- Load ----
    print(f"\nLoading tiles from {tiles_path} …")
    df_train   = load_tiles(tiles_path / "training_tiles.gpkg")
    df_predict = load_tiles(tiles_path / "prediction_tiles.gpkg")

    if df_train is None:
        print("[ERROR] training_tiles.gpkg not found — cannot continue.")
        return

    # ---- Preprocess ----
    print(
        f"\nPreprocessing  "
        f"(log={'off' if args.no_log else 'on'},"
        f" scale={'off' if args.no_scale else 'RobustScaler'})…"
    )
    result = preprocess(
        df_train, df_predict,
        apply_log=not args.no_log,
        apply_scale=not args.no_scale,
    )

    # ---- Save parquet ----
    print(f"\nWriting to {out_dir} …")
    result["X_all"].to_parquet(               out_dir / "X_all.parquet")
    result["y_all"].to_frame().to_parquet(    out_dir / "y_all.parquet")
    result["meta_all"].to_parquet(            out_dir / "meta_all.parquet")
    result["X_predict"].to_parquet(           out_dir / "X_predict.parquet")
    if not result["meta_predict"].empty:
        result["meta_predict"].to_parquet(    out_dir / "meta_predict.parquet")

    # ---- Save scaler ----
    if result["scaler"] is not None:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(result["scaler"], f)

    # ---- Feature config ----
    config = {
        "feature_cols":    result["feature_cols"],
        "binary_cols":     result["binary_cols"],
        "pct_cols":        result["pct_cols"],
        "continuous_cols": result["continuous_cols"],
        "label_col":       LABEL_COL,
        "apply_log":       not args.no_log,
        "apply_scale":     not args.no_scale,
        "random_state":    RANDOM_STATE,
        "n_all":           len(result["X_all"]),
        "n_predict":       len(result["X_predict"]),
        "positive_rate_all": float(result["y_all"].mean()),
    }
    with open(out_dir / "feature_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Report ----
    report = build_report(
        result["df_train_raw"],
        result["feature_cols"],
        result["binary_cols"],
        result["pct_cols"],
        result["continuous_cols"],
        result["X_all"], result["y_all"],
        result["X_predict"],
    )
    (out_dir / "report.txt").write_text(report)
    print("\n" + report)

    print(f"\n[DONE] → {out_dir}")
    print(f"  X_all:     {result['X_all'].shape}")
    print(f"  X_predict: {result['X_predict'].shape}")


if __name__ == "__main__":
    main()
