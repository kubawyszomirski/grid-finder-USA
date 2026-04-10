#!/usr/bin/env python3
"""
preprocess_edges.py
Converts generate_edges_paths output into clean ML-ready tabular format.

Reads (from {project}/edges/):
    {S}_edges_training.gpkg    — labelled edges (training mode)
    {S}_edges_prediction.gpkg  — unlabelled edges (prediction mode, processed separately)

If both files exist:
  - Training file is preprocessed and the scaler/clip_bounds are FIT on it.
  - Prediction file is loaded and the SAME fitted transforms are APPLIED (no refit).
  - X_predict / meta_predict come from the prediction file (not the training file).

Optional --roads-prob N:
  Adds a 'road_prob' feature to every edge — the y_proba of the road the edge
  belongs to (joined on road_id).
  Training data uses OOF road predictions (no leakage);
  prediction data uses full-model road predictions.

Writes to {project}/edges/preprocessed/:
    X_all.parquet           — scaled training features
    y_all.parquet           — grid label
    meta_all.parquet        — edge_id + centroid_lat + centroid_lon
    X_predict.parquet       — scaled prediction features (from prediction file if it exists)
    meta_predict.parquet    — edge_id for prediction rows
    scaler.pkl              — fitted RobustScaler
    feature_config.json     — column lists, transform flags
    report.txt              — label balance, correlations, NaN summary

Usage:
    python preprocess_edges.py Massachusetts --project project-ma
    python preprocess_edges.py MA --project project-ma --no-scale
    python preprocess_edges.py MA --project project-ma --roads-prob 0
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from generate.generate_utils import BASE_DIR, STATE_FULL_NAMES, _NAME_TO_ABBREV

warnings.filterwarnings("ignore")


# ==========================================
#            CONFIGURATION
# ==========================================

PROJECT      = "project-ma"
LABEL_COL    = "grid"
RANDOM_STATE = 42

# Columns never used as ML features
META_COLS = {
    "edge_id", "road_id", "geometry", "u", "v", "key", "osmid",
    "name", "ref", "maxspeed", "lanes", "oneway",
    "length", "edge_length_m", "shape_length",
}

HIGHWAY_COL = "highway"

HIGHWAY_CLASSES = [
    "primary", "secondary", "tertiary", "residential",
    "service", "osm_grid", "other",
]

HIGHWAY_EXCLUDE = {"artificial", "artificial_bridge"}

USAGE_MODE_COLS = ["p1_usage_mode", "p2_usage_mode"]

USAGE_MODE_CLASSES = ["solar", "wind", "substation", "industrial", "unknown", "other"]

CLIP_QUANTILES = (0.001, 0.999)


# ==========================================
#              HELPERS
# ==========================================

def load_edges(path: Path) -> gpd.GeoDataFrame | None:
    """Returns GeoDataFrame (with geometry) for centroid extraction."""
    if not path.exists():
        print(f"  [not found] {path}")
        return None
    gdf = gpd.read_file(path)
    if "highway" in gdf.columns:
        before = len(gdf)
        gdf = gdf[~gdf["highway"].isin(HIGHWAY_EXCLUDE)]
        dropped = before - len(gdf)
        if dropped:
            print(f"  Dropped {dropped:,} artificial edge segments")
    print(f"  {path.name}: {len(gdf):,} rows, {gdf.shape[1]} columns")
    return gdf


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
        elif (
            "pct" in col.lower()
            or "prob" in col.lower()
            or col.startswith("highway_")
            or col.startswith("p1_usage_")
            or col.startswith("p2_usage_")
        ):
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


def load_road_predictions(
    project_dir: Path,
    model_idx: int,
    training_mode: bool,
) -> pd.DataFrame | None:
    """
    Load road y_proba predictions as a DataFrame (road_id + y_proba).

    Training mode → OOF predictions (no leakage):
      {project}/models_roads/roads_model_{N}_oof_preds.parquet

    Prediction mode → full-model predictions:
      {project}/roads/predictions/roads_model_{N}_predictions.parquet
    """
    model_stem = f"roads_model_{model_idx}"

    if training_mode:
        parquet = project_dir / "models_roads" / f"{model_stem}_oof_preds.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)[["road_id", "y_proba"]]
            print(f"  Road predictions (OOF): {parquet.name}  {len(df):,} roads")
            return df
        print(f"  [warn] --roads-prob: OOF road predictions not found: {parquet}")
        print(f"    Run train_roads.py --project {project_dir.name} first")
        return None
    else:
        parquet = project_dir / "roads" / "predictions" / f"{model_stem}_predictions.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)[["road_id", "y_proba"]]
            print(f"  Road predictions (full model): {parquet.name}  {len(df):,} roads")
            return df
        print(f"  [warn] --roads-prob: Road predictions not found: {parquet}")
        print(f"    Run predict_roads.py --project {project_dir.name} --model {model_idx} first")
        return None


def compute_road_prob(df: pd.DataFrame, road_preds: pd.DataFrame) -> pd.Series:
    """
    Join road y_proba to edges on road_id.  Edges with no matching road get 0.
    Returns a Series aligned with df.index.
    """
    if "road_id" not in df.columns:
        print("  [warn] road_id column not in edges — road_prob will be 0 everywhere")
        return pd.Series(0.0, index=df.index, name="road_prob")

    # Deduplicate road_preds on road_id (average y_proba if duplicates exist)
    road_lookup = (
        road_preds.groupby("road_id", as_index=False)["y_proba"]
        .mean()
        .rename(columns={"y_proba": "road_prob"})
    )

    merged = df[["road_id"]].merge(road_lookup, on="road_id", how="left")
    result = merged["road_prob"].fillna(0.0)
    result.index = df.index
    result.name = "road_prob"

    n_covered = int((result > 0).sum())
    print(f"  road_prob: {n_covered:,}/{len(result):,} edges covered  "
          f"mean={result[result > 0].mean():.4f}  max={result.max():.4f}")
    return result


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
        lines += [f"  Training rows  : {len(result['X_all']):,}"]
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
            f"  Label balance  : {result['y_all'].value_counts().to_dict()}",
            f"  Positive rate  : {result['y_all'].mean():.3f}",
            "",
        ]
    lines += [
        "-" * 65,
        "  NaN SUMMARY (before fill):",
        "-" * 65,
        nan_report(df_raw, feature_cols),
    ]
    if training_mode:
        lines += [
            "-" * 65,
            f"  TOP {min(50, len(feature_cols))} FEATURES BY |CORRELATION| WITH LABEL (post-transform):",
            "-" * 65,
            correlation_report(result["X_all"], result["y_all"], top_n=50),
        ]
    return "\n".join(lines)


# ==========================================
#           PREPROCESSING PIPELINE
# ==========================================

def preprocess(
    gdf: gpd.GeoDataFrame,
    apply_log: bool = True,
    apply_scale: bool = True,
    road_preds: pd.DataFrame | None = None,
) -> dict:
    """
    Full pipeline:
      1. Extract centroid lat/lon from geometry (for spatial CV fold assignment)
      2. (optional) Compute road_prob — road y_proba joined on road_id
      3. One-hot encode highway and usage_mode columns
      4. Separate meta / label / features
      5. Classify columns (binary / pct / continuous)
      6. Record NaN distribution (before fill)
      7. Fill NaN → 0
      8. Clip continuous outliers (quantile bounds fitted on full dataset)
      9. Log1p continuous features (optional)
     10. RobustScaler fitted on full dataset (optional)

    No train/val split — spatial K-fold CV in train_edges.py handles validation.
    meta_all includes centroid_lat/lon so train_edges.py can assign spatial folds.
    """
    training_mode = LABEL_COL in gdf.columns

    # ---- 1. Extract centroids from geometry ----
    centroid_lat = centroid_lon = None
    try:
        geo = gdf.to_crs("EPSG:4326") if (gdf.crs is not None and gdf.crs.to_epsg() != 4326) else gdf
        cents = geo.geometry.centroid
        centroid_lat = cents.y.values
        centroid_lon = cents.x.values
    except Exception as e:
        print(f"  [warn] centroid extraction failed: {e}")

    # ---- 2. Compute road_prob (must happen before dropping geometry column) ----
    road_prob_series = None
    if road_preds is not None:
        # road_id is in the GDF as a regular column (not geometry)
        df_for_prob = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
        road_prob_series = compute_road_prob(df_for_prob, road_preds)

    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

    # ---- 3. One-hot encode categoricals ----
    df = one_hot_highway(df.copy())
    df = one_hot_usage_mode(df)

    # ---- Inject road_prob before feature column discovery ----
    if road_prob_series is not None:
        df["road_prob"] = road_prob_series.values

    # ---- 4. Build meta ----
    meta = pd.DataFrame(index=df.index)
    if "edge_id" in df.columns:
        meta["edge_id"] = df["edge_id"].values
    if centroid_lat is not None:
        meta["centroid_lat"] = centroid_lat
        meta["centroid_lon"] = centroid_lon

    y = df[LABEL_COL].fillna(0).astype(int) if training_mode else None

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS and c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_raw = df[feature_cols].copy()

    # ---- 5. Classify ----
    binary_cols, pct_cols, continuous_cols = classify_columns(df, feature_cols)

    # ---- 6. Numeric coerce + fill NaN ----
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---- 7. Clip continuous outliers ----
    X, clip_bounds = _clip_continuous(X, continuous_cols, CLIP_QUANTILES)

    # ---- 8. Log1p on continuous ----
    if apply_log and continuous_cols:
        X[continuous_cols] = np.log1p(X[continuous_cols].clip(lower=0))

    # ---- 9. Scale ----
    if apply_scale:
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=feature_cols, index=X.index,
        )
    else:
        scaler = None
        X_scaled = X

    meta_predict = meta[["edge_id"]].copy() if "edge_id" in meta.columns else pd.DataFrame(index=meta.index)

    return {
        "X_all":           X_scaled if training_mode else None,
        "y_all":           y,
        "meta_all":        meta if training_mode else None,
        "X_predict":       X_scaled,
        "meta_predict":    meta_predict,
        "scaler":          scaler,
        "clip_bounds":     clip_bounds,
        "feature_cols":    feature_cols,
        "binary_cols":     binary_cols,
        "pct_cols":        pct_cols,
        "continuous_cols": continuous_cols,
        "df_raw":          df_raw,
        "training_mode":   training_mode,
    }


def preprocess_predict(
    gdf: gpd.GeoDataFrame,
    fit_result: dict,
    apply_log: bool = True,
    apply_scale: bool = True,
    road_preds: pd.DataFrame | None = None,
) -> dict:
    """
    Apply transforms fitted on the training set to a separate prediction GeoDataFrame.

    Uses fit_result["scaler"].transform() (not fit_transform) and pre-computed
    fit_result["clip_bounds"], so no training information leaks from prediction data.

    Returns {"X_predict": pd.DataFrame, "meta_predict": pd.DataFrame}.
    """
    feature_cols    = fit_result["feature_cols"]
    continuous_cols = fit_result["continuous_cols"]
    clip_bounds     = fit_result["clip_bounds"]
    scaler          = fit_result["scaler"]

    # ---- Compute road_prob ----
    road_prob_series = None
    if road_preds is not None:
        df_for_prob = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
        road_prob_series = compute_road_prob(df_for_prob, road_preds)

    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    df = one_hot_highway(df.copy())
    df = one_hot_usage_mode(df)

    if road_prob_series is not None:
        df["road_prob"] = road_prob_series.values

    # ---- Build meta ----
    meta = pd.DataFrame(index=df.index)
    if "edge_id" in df.columns:
        meta["edge_id"] = df["edge_id"].values

    # ---- Reindex to training feature columns ----
    X = df.reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---- Clip using pre-fitted bounds (apply only, do not refit) ----
    X, _ = _clip_continuous(X, continuous_cols, CLIP_QUANTILES, bounds=clip_bounds)

    # ---- Log1p on continuous ----
    if apply_log and continuous_cols:
        X[continuous_cols] = np.log1p(X[continuous_cols].clip(lower=0))

    # ---- Scale using fitted scaler ----
    if apply_scale and scaler is not None:
        X_scaled = pd.DataFrame(
            scaler.transform(X), columns=feature_cols, index=X.index,
        )
    else:
        X_scaled = X

    meta_predict = meta[["edge_id"]].copy() if "edge_id" in meta.columns else pd.DataFrame(index=meta.index)

    return {
        "X_predict":    X_scaled,
        "meta_predict": meta_predict,
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
            "  python preprocess_edges.py MA --project project-ma --no-scale\n"
            "  python preprocess_edges.py MA --project project-ma --roads-prob 0\n"
        ),
    )
    parser.add_argument("state",      help="Full state name or 2-letter abbreviation")
    parser.add_argument("--project",  default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--no-log",   action="store_true",
                        help="Skip log1p transform on continuous features")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip RobustScaler")
    parser.add_argument("--roads-prob", type=int, metavar="N",
                        help="Roads model index (e.g. 0). Adds a 'road_prob' feature: "
                             "the road y_proba for each edge (joined on road_id). "
                             "Training mode uses OOF road predictions (no leakage); "
                             "prediction mode uses full-model road predictions.")
    args = parser.parse_args()

    abbrev = _NAME_TO_ABBREV.get(args.state.strip())
    if abbrev is None:
        print(f"[ERROR] Unknown state: '{args.state}'")
        return

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    edges_dir  = project_dir / "edges"
    train_path = edges_dir / f"{abbrev}_edges_training.gpkg"
    pred_path  = edges_dir / f"{abbrev}_edges_prediction.gpkg"
    out_dir    = project_dir / "edges" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    has_train = train_path.exists()
    has_pred  = pred_path.exists()

    if not has_train and not has_pred:
        print(f"[ERROR] No edges file found in {edges_dir} — run generate_edges_paths.py first.")
        return

    primary_path = train_path if has_train else pred_path
    has_separate_pred = has_train and has_pred

    print(f"\n{'='*60}")
    print(f" Edge Preprocessing — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f" Project : {args.project}")
    print(f"{'='*60}")

    # ---- Load primary (training) file ----
    print(f"\nLoading {primary_path.name} …")
    gdf = load_edges(primary_path)
    if gdf is None:
        return

    training_mode = LABEL_COL in gdf.columns
    print(f"  Mode: {'training (grid label found)' if training_mode else 'prediction-only'}")
    if has_separate_pred:
        print(f"  Prediction file: {pred_path.name}  (will be transformed with fitted params)")

    # ---- Load road predictions (optional) ----
    road_preds_train = None
    if args.roads_prob is not None:
        print(f"\nLoading road predictions (--roads-prob {args.roads_prob})…")
        road_preds_train = load_road_predictions(project_dir, args.roads_prob, training_mode)
        if road_preds_train is None:
            print("  [warn] Proceeding without road_prob feature.")

    print(
        f"\nPreprocessing training file  "
        f"(log={'off' if args.no_log else 'on'},"
        f" scale={'off' if args.no_scale else 'RobustScaler'}"
        f"{f', road_prob from roads_model_{args.roads_prob}' if road_preds_train is not None else ''})…"
    )

    result = preprocess(
        gdf,
        apply_log=not args.no_log,
        apply_scale=not args.no_scale,
        road_preds=road_preds_train,
    )

    # ---- Process separate prediction file (apply fitted transforms, no refit) ----
    if has_separate_pred:
        print(f"\nLoading prediction file {pred_path.name} …")
        gdf_pred = load_edges(pred_path)
        if gdf_pred is not None:
            road_preds_pred = None
            if args.roads_prob is not None:
                print(f"  Loading full-model road predictions for prediction file…")
                road_preds_pred = load_road_predictions(
                    project_dir, args.roads_prob, training_mode=False,
                )
            print(f"  Applying fitted transforms to {len(gdf_pred):,} prediction edges…")
            pred_result = preprocess_predict(
                gdf_pred, result,
                apply_log=not args.no_log,
                apply_scale=not args.no_scale,
                road_preds=road_preds_pred,
            )
            result["X_predict"]    = pred_result["X_predict"]
            result["meta_predict"] = pred_result["meta_predict"]
            print(f"  X_predict: {result['X_predict'].shape}  (from {pred_path.name})")
        else:
            print(f"  [warn] Could not load {pred_path.name} — X_predict will be training data")

    # ---- Save parquets ----
    print(f"\nWriting to {out_dir} …")
    result["X_predict"].to_parquet(    out_dir / "X_predict.parquet")
    result["meta_predict"].to_parquet( out_dir / "meta_predict.parquet")

    if training_mode:
        result["X_all"].to_parquet(            out_dir / "X_all.parquet")
        result["y_all"].to_frame().to_parquet( out_dir / "y_all.parquet")
        result["meta_all"].to_parquet(         out_dir / "meta_all.parquet")
        has_centroids = "centroid_lat" in result["meta_all"].columns
        print(f"  meta_all: edge_id + {'centroid_lat/lon' if has_centroids else '[no centroids — geometry missing?]'}")

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
        "random_state":      RANDOM_STATE,
        "clip_quantiles":    list(CLIP_QUANTILES),
        "n_predict":         len(result["X_predict"]),
        "highway_classes":   HIGHWAY_CLASSES,
        "usage_mode_classes": USAGE_MODE_CLASSES,
        "roads_prob": {
            "enabled":     road_preds_train is not None,
            "roads_model": args.roads_prob,
            "mode":        "oof" if training_mode else "predictions",
        } if args.roads_prob is not None else {"enabled": False},
    }
    if training_mode:
        config.update({
            "n_all":              len(result["X_all"]),
            "positive_rate_all":  float(result["y_all"].mean()),
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
    if training_mode:
        print(f"  X_all:     {result['X_all'].shape}")
    print(f"  X_predict: {result['X_predict'].shape}"
          f"{' (separate prediction file)' if has_separate_pred else ' (= training data)'}")


if __name__ == "__main__":
    main()
