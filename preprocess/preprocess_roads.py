#!/usr/bin/env python3
"""
preprocess_roads.py
Converts generate-roads output into clean ML-ready tabular format.

Reads (from {project}/roads/):
    {S}_roads_training.gpkg    — labelled roads (training mode)
    {S}_roads_prediction.gpkg  — unlabelled roads (prediction mode, processed separately)

If both files exist:
  - Training file is preprocessed and the scaler/clip_bounds are FIT on it.
  - Prediction file is loaded and the SAME fitted transforms are APPLIED (no refit).
  - X_predict / meta_predict come from the prediction file (not the training file).

Optional --tiles-prob <tiles_dir>:
  Adds a 'tiles_prob' feature to every road — the length-weighted average
  tile y_proba from all tiles the road segment passes through.
  Training data uses OOF tile predictions (no leakage);
  prediction data uses full-model tile predictions.

Writes to {project}/roads/preprocessed/:
    X_all.parquet           — scaled training features
    y_all.parquet           — grid label
    meta_all.parquet        — road_id + centroid_lat + centroid_lon
    X_predict.parquet       — scaled prediction features (from prediction file if it exists)
    meta_predict.parquet    — road_id for prediction rows
    scaler.pkl              — fitted RobustScaler
    feature_config.json     — column lists, transform flags
    report.txt              — label balance, correlations, NaN summary

Usage:
    python preprocess_roads.py Massachusetts --project project-ma
    python preprocess_roads.py MA --project project-ma --no-scale
    python preprocess_roads.py MA --project project-ma --tiles-prob 500_tiles
    python preprocess_roads.py MA --project project-ma --tiles-prob 500_tiles --tiles-model 1
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

# Columns that are never used as ML features
META_COLS = {"road_id", "name", "geometry", "u", "v", "key", "osmid",
             "from", "to", "ref", "maxspeed", "lanes", "oneway",
             "length", "edge_length_m", "shape_length"}

HIGHWAY_COL = "highway"

HIGHWAY_CLASSES = [
    "primary", "secondary", "tertiary", "residential",
    "service", "osm_grid", "other",
]

HIGHWAY_EXCLUDE = {"artificial", "artificial_bridge"}

CLIP_QUANTILES = (0.001, 0.999)

# For roads with a unique street name (or unnamed), name-aggregated features
# are NaN.  Rather than filling with 0, use the road's own segment value.
NAME_COL_FALLBACKS = {
    "name_total_length_m": "length_m",
    "name_curvature_per_m": "curvature_per_m",
}


# ==========================================
#              HELPERS
# ==========================================

def load_roads(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        print(f"  [not found] {path}")
        return None
    gdf = gpd.read_file(path)
    if "highway" in gdf.columns:
        before = len(gdf)
        gdf = gdf[~gdf["highway"].isin(HIGHWAY_EXCLUDE)]
        dropped = before - len(gdf)
        if dropped:
            print(f"  Dropped {dropped:,} artificial road segments")
    print(f"  {path.name}: {len(gdf):,} rows, {gdf.shape[1]} columns")
    return gdf


def _is_binary(series: pd.Series) -> bool:
    return set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0})


def one_hot_highway(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the highway column against the fixed HIGHWAY_CLASSES list.
    Values not in the list are collapsed to 'other'.
    Returns df with highway column replaced by highway_<class> binary columns.
    """
    if HIGHWAY_COL not in df.columns:
        for cls in HIGHWAY_CLASSES:
            df[f"highway_{cls}"] = 0
        return df

    raw = df[HIGHWAY_COL].astype(str).str.lower().str.strip()
    raw = raw.str.replace(r"[\[\]']", "", regex=True).str.split(",").str[0].str.strip()
    raw = raw.where(raw.isin(HIGHWAY_CLASSES), other="other")

    for cls in HIGHWAY_CLASSES:
        df[f"highway_{cls}"] = (raw == cls).astype(np.int8)

    df = df.drop(columns=[HIGHWAY_COL], errors="ignore")
    return df


def classify_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list, list, list]:
    binary, pct, continuous = [], [], []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _is_binary(df[col]):
            binary.append(col)
        elif "pct" in col.lower() or "prob" in col.lower() or col.startswith("highway_"):
            # probability and percentage columns: scale but no log1p
            pct.append(col)
        else:
            continuous.append(col)
    return binary, pct, continuous


def _clip_continuous(df: pd.DataFrame, cols: list[str],
                     quantiles: tuple[float, float],
                     bounds: dict | None = None) -> tuple[pd.DataFrame, dict]:
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


def load_tile_predictions(
    project_dir: Path,
    tiles_dir: str,
    model_idx: int,
    training_mode: bool,
) -> gpd.GeoDataFrame | None:
    """
    Load tile y_proba predictions as a GeoDataFrame (geometry + y_proba).

    Training mode → OOF predictions (no leakage):
      1. {project}/{tiles_dir}/predictions/tiles_model_{N}_oof_preds.gpkg  (preferred)
      2. {project}/models_tiles/tiles_model_{N}_oof_preds.parquet
         joined with {project}/{tiles_dir}/training_tiles.gpkg             (fallback)

    Prediction mode → full-model predictions:
      {project}/{tiles_dir}/predictions/tiles_model_{N}_predictions.gpkg
    """
    model_stem = f"tiles_model_{model_idx}"
    preds_dir  = project_dir / tiles_dir / "predictions"

    if training_mode:
        gpkg = preds_dir / f"{model_stem}_oof_preds.gpkg"
        if gpkg.exists():
            gdf = gpd.read_file(gpkg)
            print(f"  Tile predictions (OOF gpkg): {gpkg.name}  {len(gdf):,} tiles")
            return gdf[["geometry", "y_proba"]].copy()

        # Fallback: parquet + training tile geometry
        parquet  = project_dir / "models_tiles" / f"{model_stem}_oof_preds.parquet"
        geom_src = project_dir / tiles_dir / "training_tiles.gpkg"
        if parquet.exists() and geom_src.exists():
            oof_df    = pd.read_parquet(parquet)[["tile_id", "y_proba"]]
            tiles_gdf = gpd.read_file(geom_src)[["tile_id", "geometry"]]
            merged    = tiles_gdf.merge(oof_df, on="tile_id", how="inner")
            print(f"  Tile predictions (OOF parquet + geometry): {len(merged):,} tiles")
            return merged[["geometry", "y_proba"]].copy()

        print(f"  [warn] --tiles-prob: OOF tile predictions not found.")
        print(f"    Run predict_tiles_tabular.py --project {project_dir.name} "
              f"--tiles-dir {tiles_dir} --model {model_idx} --oof")
        return None
    else:
        gpkg = preds_dir / f"{model_stem}_predictions.gpkg"
        if gpkg.exists():
            gdf = gpd.read_file(gpkg)
            print(f"  Tile predictions (full model): {gpkg.name}  {len(gdf):,} tiles")
            return gdf[["geometry", "y_proba"]].copy()
        print(f"  [warn] --tiles-prob: Tile predictions not found: {gpkg}")
        print(f"    Run predict_tiles_tabular.py --project {project_dir.name} "
              f"--tiles-dir {tiles_dir} --model {model_idx}")
        return None


def compute_tiles_prob(
    gdf_roads: gpd.GeoDataFrame,
    gdf_tiles: gpd.GeoDataFrame,
) -> pd.Series:
    """
    For each road segment, compute the length-weighted average tile y_proba
    across all tiles the road passes through.

    Roads that don't intersect any tile receive tiles_prob = 0.
    Computation happens in EPSG:5070 (metric) for accurate length weighting.
    Returns a Series aligned with gdf_roads.index.
    """
    print(f"  Computing tile-road overlaps "
          f"({len(gdf_roads):,} roads × {len(gdf_tiles):,} tiles)…")

    # Project to metric CRS for accurate length measurement
    roads_m = gdf_roads[["geometry"]].copy().to_crs("EPSG:5070")
    tiles_m = gdf_tiles[["geometry", "y_proba"]].copy().to_crs("EPSG:5070")

    # Preserve original index as a column before overlay resets it
    roads_m = roads_m.assign(_road_idx=roads_m.index).reset_index(drop=True)

    # Clip each road to the tiles it passes through; keep_geom_type=True
    # retains only line/multiline results (drops point touches at tile corners)
    clipped = gpd.overlay(roads_m, tiles_m, how="intersection", keep_geom_type=True)

    if clipped.empty:
        print("  tiles_prob: no road-tile overlaps found — feature will be 0 everywhere")
        return pd.Series(0.0, index=gdf_roads.index, name="tiles_prob")

    clipped["seg_length"] = clipped.geometry.length
    clipped = clipped[clipped["seg_length"] > 0]

    if clipped.empty:
        return pd.Series(0.0, index=gdf_roads.index, name="tiles_prob")

    # Length-weighted average y_proba per road
    tiles_prob = (
        clipped.groupby("_road_idx")
        .apply(lambda g: float(np.average(g["y_proba"], weights=g["seg_length"])))
    )

    # Align to original road index; roads outside all tiles → 0
    result = tiles_prob.reindex(gdf_roads.index).fillna(0.0)
    result.name = "tiles_prob"

    n_covered = int((result > 0).sum())
    print(f"  tiles_prob: {n_covered:,}/{len(result):,} roads covered  "
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
    lines = [sep, "  ROAD PREPROCESSING REPORT", sep]
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
            f"  TOP {min(25, len(feature_cols))} FEATURES BY |CORRELATION| WITH LABEL (post-transform):",
            "-" * 65,
            correlation_report(result["X_all"], result["y_all"]),
        ]
    return "\n".join(lines)


# ==========================================
#           PREPROCESSING PIPELINE
# ==========================================

def preprocess(
    gdf: gpd.GeoDataFrame,
    apply_log: bool = True,
    apply_scale: bool = True,
    tiles_gdf: gpd.GeoDataFrame | None = None,
) -> dict:
    """
    Full pipeline:
      1. Extract centroid lat/lon from geometry (for spatial CV fold assignment)
      2. (optional) Compute tiles_prob — length-weighted tile y_proba per road
      3. One-hot encode highway class
      4. Separate meta / label / features
      5. Classify columns (binary / pct / continuous)
         tiles_prob → pct group: scaled but not log-transformed
      6. Record NaN distribution (before fill)
      7. Fill NaN → 0
      8. Clip continuous outliers (quantile bounds fitted on full dataset)
      9. Log1p continuous features (optional)
     10. RobustScaler fitted on full dataset (optional)

    No train/val split — spatial K-fold CV in train_roads.py handles validation.
    meta_all includes centroid_lat/lon so train_roads.py can assign spatial folds.
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

    # ---- 2. Compute tiles_prob (needs geometry — must happen before dropping it) ----
    tiles_prob_series: pd.Series | None = None
    if tiles_gdf is not None:
        tiles_prob_series = compute_tiles_prob(gdf, tiles_gdf)

    # Convert to plain DataFrame
    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

    # ---- 3. One-hot highway ----
    df = one_hot_highway(df.copy())

    # ---- Inject tiles_prob before feature column discovery ----
    if tiles_prob_series is not None:
        df["tiles_prob"] = tiles_prob_series.values

    # ---- 3. Build meta ----
    meta = pd.DataFrame(index=df.index)
    if "road_id" in df.columns:
        meta["road_id"] = df["road_id"].values
    if centroid_lat is not None:
        meta["centroid_lat"] = centroid_lat
        meta["centroid_lon"] = centroid_lon

    y = df[LABEL_COL].fillna(0).astype(int) if training_mode else None

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS and c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_raw = df[feature_cols].copy()

    # ---- 4. Classify ----
    binary_cols, pct_cols, continuous_cols = classify_columns(df, feature_cols)

    # ---- 5. Numeric coerce ----
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # ---- 5a. Fallback fill for name_* columns ----
    # Roads with a unique (or no) street name have NaN for name-aggregated
    # features.  Fill with the road's own segment value rather than 0.
    for col, fallback in NAME_COL_FALLBACKS.items():
        if col in X.columns and fallback in X.columns:
            mask = X[col].isna()
            if mask.any():
                X.loc[mask, col] = X.loc[mask, fallback]

    # ---- 5b. Fill remaining NaN → 0 ----
    X = X.fillna(0)

    # ---- 6. Clip continuous outliers ----
    X, clip_bounds = _clip_continuous(X, continuous_cols, CLIP_QUANTILES)

    # ---- 7. Log1p on continuous ----
    if apply_log and continuous_cols:
        X[continuous_cols] = np.log1p(X[continuous_cols].clip(lower=0))

    # ---- 8. Scale ----
    if apply_scale:
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=feature_cols, index=X.index,
        )
    else:
        scaler = None
        X_scaled = X

    meta_predict = meta[["road_id"]].copy() if "road_id" in meta.columns else pd.DataFrame(index=meta.index)

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
    tiles_gdf: gpd.GeoDataFrame | None = None,
) -> dict:
    """
    Apply transforms fitted on the training set to a separate prediction GeoDataFrame.

    Uses fit_result["scaler"].transform() (not fit_transform) and the pre-computed
    fit_result["clip_bounds"], so no information from the prediction set leaks into
    the fitted parameters.

    Returns {"X_predict": pd.DataFrame, "meta_predict": pd.DataFrame}.
    """
    feature_cols    = fit_result["feature_cols"]
    continuous_cols = fit_result["continuous_cols"]
    clip_bounds     = fit_result["clip_bounds"]
    scaler          = fit_result["scaler"]

    # ---- Compute tiles_prob (needs geometry) ----
    tiles_prob_series = None
    if tiles_gdf is not None:
        tiles_prob_series = compute_tiles_prob(gdf, tiles_gdf)

    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    df = one_hot_highway(df.copy())

    if tiles_prob_series is not None:
        df["tiles_prob"] = tiles_prob_series.values

    # ---- Build meta ----
    meta = pd.DataFrame(index=df.index)
    if "road_id" in df.columns:
        meta["road_id"] = df["road_id"].values

    # ---- Reindex to training feature columns ----
    X = df.reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # ---- Fallback fill for name_* columns (same logic as training) ----
    for col, fallback in NAME_COL_FALLBACKS.items():
        if col in X.columns and fallback in X.columns:
            mask = X[col].isna()
            if mask.any():
                X.loc[mask, col] = X.loc[mask, fallback]

    X = X.fillna(0)

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

    meta_predict = meta[["road_id"]].copy() if "road_id" in meta.columns else pd.DataFrame(index=meta.index)

    return {
        "X_predict":    X_scaled,
        "meta_predict": meta_predict,
    }


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess road features into ML-ready tabular format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preprocess_roads.py Massachusetts --project project-ma\n"
            "  python preprocess_roads.py MA --project project-ma --no-scale\n"
        ),
    )
    parser.add_argument("state",      help="Full state name or 2-letter abbreviation")
    parser.add_argument("--project",  default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--no-log",   action="store_true",
                        help="Skip log1p transform on continuous features")
    parser.add_argument("--no-scale", action="store_true",
                        help="Skip RobustScaler")
    parser.add_argument("--tiles-prob", metavar="TILES_DIR",
                        help="Tile subfolder name (e.g. '500_tiles'). Adds a "
                             "'tiles_prob' feature: length-weighted tile y_proba "
                             "per road. Training mode uses OOF tile predictions "
                             "(no leakage); prediction mode uses full-model preds.")
    parser.add_argument("--tiles-model", type=int, default=0, metavar="N",
                        help="Tiles model index to use for tiles_prob (default: 0)")
    args = parser.parse_args()

    abbrev = _NAME_TO_ABBREV.get(args.state.strip())
    if abbrev is None:
        print(f"[ERROR] Unknown state: '{args.state}'")
        return

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    roads_dir  = project_dir / "roads"
    train_path = roads_dir / f"{abbrev}_roads_training.gpkg"
    pred_path  = roads_dir / f"{abbrev}_roads_prediction.gpkg"
    out_dir    = project_dir / "roads" / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    has_train = train_path.exists()
    has_pred  = pred_path.exists()

    if not has_train and not has_pred:
        print(f"[ERROR] No roads file found in {roads_dir} — run generate_roads.py first.")
        return

    # Primary file: training if available, otherwise prediction-only
    primary_path = train_path if has_train else pred_path
    has_separate_pred = has_train and has_pred

    print(f"\n{'='*60}")
    print(f" Road Preprocessing — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f" Project : {args.project}")
    print(f"{'='*60}")

    # ---- Load primary (training) file ----
    print(f"\nLoading {primary_path.name} …")
    gdf = load_roads(primary_path)
    if gdf is None:
        return

    training_mode = LABEL_COL in gdf.columns
    print(f"  Mode: {'training (grid label found)' if training_mode else 'prediction-only'}")
    if has_separate_pred:
        print(f"  Prediction file: {pred_path.name}  (will be transformed with fitted params)")

    # ---- Load tile predictions for training data ----
    tiles_gdf = None
    if args.tiles_prob:
        print(f"\nLoading tile predictions (--tiles-prob {args.tiles_prob}, model {args.tiles_model})…")
        tiles_gdf = load_tile_predictions(
            project_dir, args.tiles_prob, args.tiles_model, training_mode,
        )
        if tiles_gdf is None:
            print("  [warn] Proceeding without tiles_prob feature.")

    print(
        f"\nPreprocessing training file  "
        f"(log={'off' if args.no_log else 'on'},"
        f" scale={'off' if args.no_scale else 'RobustScaler'}"
        f"{f', tiles_prob from {args.tiles_prob}' if tiles_gdf is not None else ''})…"
    )

    result = preprocess(
        gdf,
        apply_log=not args.no_log,
        apply_scale=not args.no_scale,
        tiles_gdf=tiles_gdf,
    )

    # ---- Process separate prediction file (apply fitted transforms, no refit) ----
    if has_separate_pred:
        print(f"\nLoading prediction file {pred_path.name} …")
        gdf_pred = load_roads(pred_path)
        if gdf_pred is not None:
            # Tile predictions for prediction file: always full-model (no OOF)
            tiles_gdf_pred = None
            if args.tiles_prob:
                print(f"  Loading full-model tile predictions for prediction file…")
                tiles_gdf_pred = load_tile_predictions(
                    project_dir, args.tiles_prob, args.tiles_model, training_mode=False,
                )
            print(f"  Applying fitted transforms to {len(gdf_pred):,} prediction roads…")
            pred_result = preprocess_predict(
                gdf_pred, result,
                apply_log=not args.no_log,
                apply_scale=not args.no_scale,
                tiles_gdf=tiles_gdf_pred,
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
        result["X_all"].to_parquet(              out_dir / "X_all.parquet")
        result["y_all"].to_frame().to_parquet(   out_dir / "y_all.parquet")
        result["meta_all"].to_parquet(           out_dir / "meta_all.parquet")
        has_centroids = "centroid_lat" in result["meta_all"].columns
        print(f"  meta_all: road_id + {'centroid_lat/lon' if has_centroids else '[no centroids — geometry missing?]'}")

    # ---- Scaler ----
    if result["scaler"] is not None:
        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(result["scaler"], f)

    # ---- Feature config ----
    config: dict = {
        "state":           abbrev,
        "feature_cols":    result["feature_cols"],
        "binary_cols":     result["binary_cols"],
        "pct_cols":        result["pct_cols"],
        "continuous_cols": result["continuous_cols"],
        "label_col":       LABEL_COL,
        "training_mode":   training_mode,
        "apply_log":       not args.no_log,
        "apply_scale":     not args.no_scale,
        "random_state":    RANDOM_STATE,
        "clip_quantiles":  list(CLIP_QUANTILES),
        "n_predict":       len(result["X_predict"]),
        "highway_classes": HIGHWAY_CLASSES,
        "tiles_prob":      {
            "enabled":     tiles_gdf is not None,
            "tiles_dir":   args.tiles_prob,
            "tiles_model": args.tiles_model,
            "mode":        "oof" if training_mode else "predictions",
        } if args.tiles_prob else {"enabled": False},
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
