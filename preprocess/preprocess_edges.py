#!/usr/bin/env python3
"""
preprocess_edges.py
Converts generate_edges_paths output into clean ML-ready tabular format.

Two modes:
  Prediction-only  — no 'grid' label in the file; outputs X_predict + meta
  Training         — 'grid' label present; also outputs train / val split

Reads:
    {project}/edges/{state}_edges_clean.gpkg

Writes to {project}/edges/preprocessed/:
    X_predict.parquet       — scaled prediction features (all edges)
    meta_predict.parquet    — edge_id for all edges
    X_train.parquet         — scaled training features   (training mode only)
    y_train.parquet
    meta_train.parquet
    X_val.parquet
    y_val.parquet
    meta_val.parquet
    scaler.pkl              — fitted RobustScaler
    feature_config.json     — column lists, transform flags, split stats
    report.txt              — label balance, correlations, NaN summary

Usage:
    python preprocess_edges.py Massachusetts --project project-ma
    python preprocess_edges.py MA --project project-ma --val-frac 0.15
    python preprocess_edges.py MA --project project-ma --no-scale
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from generate.generate_utils import BASE_DIR, STATE_FULL_NAMES, _NAME_TO_ABBREV

warnings.filterwarnings("ignore")


# ==========================================
#            CONFIGURATION
# ==========================================

PROJECT      = "project-ma"
LABEL_COL    = "grid"
VAL_FRACTION = 0.20
RANDOM_STATE = 42

# Columns never used as ML features
META_COLS = {
    "edge_id", "geometry", "u", "v", "key", "osmid",
    "name", "ref", "maxspeed", "lanes", "oneway",
    "length", "edge_length_m", "shape_length",
}

# Road class column (from OSMnx / generate-edges-paths)
HIGHWAY_COL = "highway"

# Known highway classes — fixed list for consistent one-hot encoding
HIGHWAY_CLASSES = [
    "primary", "secondary", "tertiary", "residential",
    "service", "osm_grid", "unclassified", "trunk",
    "motorway", "living_street", "other",
]

# Usage-mode columns produced by pathfinding
USAGE_MODE_COLS = ["p1_usage_mode", "p2_usage_mode"]

# Known cluster dominant_class values — fixed for consistent one-hot encoding
USAGE_MODE_CLASSES = ["solar", "wind", "substation", "industrial", "unknown", "other"]

# Outlier clipping for distance / area features
CLIP_QUANTILES = (0.001, 0.999)


# ==========================================
#              HELPERS
# ==========================================

def load_edges(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [not found] {path}")
        return None
    gdf = gpd.read_file(path)
    df  = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    print(f"  {path.name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def _is_binary(series: pd.Series) -> bool:
    return set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0})


def one_hot_highway(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode highway column against HIGHWAY_CLASSES; unknown → 'other'."""
    if HIGHWAY_COL not in df.columns:
        for cls in HIGHWAY_CLASSES:
            df[f"highway_{cls}"] = 0
        return df
    raw = df[HIGHWAY_COL].astype(str).str.lower().str.strip()
    raw = raw.str.replace(r"[\[\]']", "", regex=True).str.split(",").str[0].str.strip()
    raw = raw.where(raw.isin(HIGHWAY_CLASSES), other="other")
    for cls in HIGHWAY_CLASSES:
        df[f"highway_{cls}"] = (raw == cls).astype(np.int8)
    return df.drop(columns=[HIGHWAY_COL], errors="ignore")


def one_hot_usage_mode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode p1_usage_mode / p2_usage_mode against USAGE_MODE_CLASSES."""
    for col in USAGE_MODE_COLS:
        if col not in df.columns:
            for cls in USAGE_MODE_CLASSES:
                df[f"{col}_{cls}"] = 0
            continue
        raw = df[col].fillna("none").astype(str).str.lower().str.strip()
        raw = raw.where(raw.isin(USAGE_MODE_CLASSES), other="other")
        for cls in USAGE_MODE_CLASSES:
            df[f"{col}_{cls}"] = (raw == cls).astype(np.int8)
        df = df.drop(columns=[col], errors="ignore")
    return df


def classify_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list, list, list]:
    """Split features into binary / pct / continuous."""
    binary, pct, continuous = [], [], []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _is_binary(df[col]):
            binary.append(col)
        elif "pct" in col.lower() or col.startswith("highway_") or col.startswith("p1_usage_") or col.startswith("p2_usage_"):
            pct.append(col)
        else:
            continuous.append(col)
    return binary, pct, continuous


def _clip_continuous(
    df: pd.DataFrame,
    cols: list[str],
    quantiles: tuple[float, float],
    bounds: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    lo_q, hi_q = quantiles
    if bounds is None:
        bounds = {}
        for c in cols:
            s = df[c]
            if s.notna().any():
                lo = float(s.quantile(lo_q))
                hi = float(s.quantile(hi_q))
                bounds[c] = (lo, hi)
                df[c] = s.clip(lo, hi)
    else:
        for c in cols:
            if c in bounds:
                lo, hi = bounds[c]
                df[c] = df[c].clip(lo, hi)
    return df, bounds


def nan_report(df: pd.DataFrame, feature_cols: list[str]) -> str:
    missing = df[feature_cols].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return "  No missing values.\n"
    lines = []
    for col, n in missing.items():
        lines.append(f"  {col:<50s} {n:>6,}  ({n / len(df) * 100:.1f}%)")
    return "\n".join(lines) + "\n"


def correlation_report(X: pd.DataFrame, y: pd.Series, top_n: int = 25) -> str:
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    lines = []
    for feat, val in corr.head(top_n).items():
        bar = "█" * int(val * 30)
        lines.append(f"  {feat:<50s}  {val:.4f}  {bar}")
    return "\n".join(lines)


def build_report(
    df_raw: pd.DataFrame,
    feature_cols: list[str],
    binary_cols: list[str],
    pct_cols: list[str],
    continuous_cols: list[str],
    result: dict,
    training_mode: bool,
) -> str:
    sep = "=" * 65
    lines = [sep, "  EDGE PREPROCESSING REPORT", sep]
    if training_mode:
        lines += [
            f"  Training rows  : {len(result['X_train']):,}",
            f"  Validation rows: {len(result['X_val']):,}",
        ]
    lines += [
        f"  Predict rows   : {len(result['X_predict']):,}",
        "",
        f"  Total features : {len(feature_cols)}",
        f"    binary       : {len(binary_cols)}",
        f"    percentage   : {len(pct_cols)}",
        f"    continuous   : {len(continuous_cols)}  (clip → log1p → scale)",
        "",
    ]
    if training_mode:
        lines += [
            f"  Label balance (train): {result['y_train'].value_counts().to_dict()}",
            f"  Label balance (val)  : {result['y_val'].value_counts().to_dict()}",
            f"  Positive rate  train : {result['y_train'].mean():.3f}",
            f"  Positive rate  val   : {result['y_val'].mean():.3f}",
            "",
        ]
    lines += [
        "-" * 65,
        "  NaN SUMMARY (before fill):",
        "-" * 65,
        nan_report(df_raw, feature_cols),
    ]
    if training_mode and result.get("X_train") is not None:
        lines += [
            "-" * 65,
            f"  TOP {min(25, len(feature_cols))} FEATURES BY |CORRELATION| WITH LABEL (post-transform):",
            "-" * 65,
            correlation_report(result["X_train"], result["y_train"]),
        ]
    return "\n".join(lines)


# ==========================================
#           PREPROCESSING PIPELINE
# ==========================================

def preprocess(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    apply_log: bool = True,
    apply_scale: bool = True,
) -> dict:
    """
    Full pipeline:
      1. One-hot encode highway and usage_mode columns
      2. Separate meta / label / features
      3. Classify columns (binary / pct / continuous)
      4. Record NaN distribution (before fill)
      5. Fill NaN → 0
      6. Clip continuous outliers
      7. Log1p continuous features (optional)
      8. RobustScaler fitted on full dataset (optional)
      9. Stratified train/val split (only if 'grid' label present)
    """
    training_mode = LABEL_COL in df.columns

    # ---- 1. Encode categoricals ----
    df = one_hot_highway(df.copy())
    df = one_hot_usage_mode(df)

    # ---- 2. Split out meta / label / features ----
    meta = df[["edge_id"]].copy() if "edge_id" in df.columns else pd.DataFrame(index=df.index)
    y    = df[LABEL_COL].fillna(0).astype(int) if training_mode else None

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
        and c != LABEL_COL
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_raw = df[feature_cols].copy()  # pre-fill snapshot for NaN report

    # ---- 3. Classify ----
    binary_cols, pct_cols, continuous_cols = classify_columns(df, feature_cols)

    # ---- 4. Numeric coerce + fill NaN ----
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # ---- 5. Clip continuous outliers ----
    X, clip_bounds = _clip_continuous(X, continuous_cols, CLIP_QUANTILES)

    # ---- 6. Log1p on continuous ----
    if apply_log and continuous_cols:
        X[continuous_cols] = np.log1p(X[continuous_cols].clip(lower=0))

    # ---- 7. Scale ----
    if apply_scale:
        scaler   = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=feature_cols, index=X.index,
        )
    else:
        scaler   = None
        X_scaled = X

    # ---- 8. Train/val split ----
    if training_mode:
        X_train, X_val, y_train, y_val, meta_tr, meta_val = train_test_split(
            X_scaled, y, meta,
            test_size=val_fraction,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    else:
        X_train = X_val = y_train = y_val = meta_tr = meta_val = None

    return {
        "X_predict":       X_scaled,
        "meta_predict":    meta,
        "X_train":         X_train,
        "y_train":         y_train,
        "meta_train":      meta_tr,
        "X_val":           X_val,
        "y_val":           y_val,
        "meta_val":        meta_val,
        "scaler":          scaler,
        "clip_bounds":     clip_bounds,
        "feature_cols":    feature_cols,
        "binary_cols":     binary_cols,
        "pct_cols":        pct_cols,
        "continuous_cols": continuous_cols,
        "df_raw":          df_raw,
        "training_mode":   training_mode,
    }


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess edge features into ML-ready tabular format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preprocess_edges.py Massachusetts --project project-ma\n"
            "  python preprocess_edges.py MA --project project-ma --val-frac 0.15\n"
            "  python preprocess_edges.py MA --project project-ma --no-scale\n"
        ),
    )
    parser.add_argument("state", help="Full state name or 2-letter abbreviation")
    parser.add_argument("--project", default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--val-frac", type=float, default=VAL_FRACTION,
                        help=f"Validation fraction (default: {VAL_FRACTION})")
    parser.add_argument("--no-log",   action="store_true",
                        help="Skip log1p transform on continuous features")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip RobustScaler")
    args = parser.parse_args()

    abbrev = _NAME_TO_ABBREV.get(args.state.strip())
    if abbrev is None:
        print(f"[ERROR] Unknown state: '{args.state}'")
        return

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    in_path = project_dir / "edges" / f"{abbrev}_edges_clean.gpkg"
    out_dir = project_dir / "edges" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Edge Preprocessing — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f" Project : {args.project}")
    print(f"{'='*60}")

    # ---- Load ----
    print(f"\nLoading {in_path.name} …")
    df = load_edges(in_path)
    if df is None:
        print("[ERROR] Edges file not found — run generate_edges_paths.py first.")
        return

    training_mode = LABEL_COL in df.columns
    print(f"  Mode: {'training (grid label found)' if training_mode else 'prediction-only'}")
    print(
        f"\nPreprocessing  "
        f"(log={'off' if args.no_log else 'on'},"
        f" scale={'off' if args.no_scale else 'RobustScaler'})…"
    )

    result = preprocess(
        df,
        val_fraction=args.val_frac,
        apply_log=not args.no_log,
        apply_scale=not args.no_scale,
    )

    # ---- Save parquets ----
    print(f"\nWriting to {out_dir} …")
    result["X_predict"].to_parquet(    out_dir / "X_predict.parquet")
    result["meta_predict"].to_parquet( out_dir / "meta_predict.parquet")

    if training_mode:
        result["X_train"].to_parquet(           out_dir / "X_train.parquet")
        result["y_train"].to_frame().to_parquet( out_dir / "y_train.parquet")
        result["meta_train"].to_parquet(         out_dir / "meta_train.parquet")
        result["X_val"].to_parquet(              out_dir / "X_val.parquet")
        result["y_val"].to_frame().to_parquet(   out_dir / "y_val.parquet")
        result["meta_val"].to_parquet(           out_dir / "meta_val.parquet")

    # ---- Scaler ----
    if result["scaler"] is not None:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(result["scaler"], f)

    # ---- Feature config ----
    config: dict = {
        "state":             abbrev,
        "feature_cols":      result["feature_cols"],
        "binary_cols":       result["binary_cols"],
        "pct_cols":          result["pct_cols"],
        "continuous_cols":   result["continuous_cols"],
        "label_col":         LABEL_COL,
        "training_mode":     training_mode,
        "apply_log":         not args.no_log,
        "apply_scale":       not args.no_scale,
        "val_fraction":      args.val_frac,
        "random_state":      RANDOM_STATE,
        "clip_quantiles":    list(CLIP_QUANTILES),
        "n_predict":         len(result["X_predict"]),
        "highway_classes":   HIGHWAY_CLASSES,
        "usage_mode_classes": USAGE_MODE_CLASSES,
    }
    if training_mode:
        config.update({
            "n_train":              len(result["X_train"]),
            "n_val":                len(result["X_val"]),
            "positive_rate_train":  float(result["y_train"].mean()),
            "positive_rate_val":    float(result["y_val"].mean()),
        })
    with open(out_dir / "feature_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Report ----
    report = build_report(
        result["df_raw"],
        result["feature_cols"],
        result["binary_cols"],
        result["pct_cols"],
        result["continuous_cols"],
        result,
        training_mode,
    )
    (out_dir / "report.txt").write_text(report)
    print("\n" + report)

    print(f"\n[DONE] → {out_dir}")
    print(f"  X_predict: {result['X_predict'].shape}")
    if training_mode:
        print(f"  X_train:   {result['X_train'].shape}")
        print(f"  X_val:     {result['X_val'].shape}")


if __name__ == "__main__":
    main()
