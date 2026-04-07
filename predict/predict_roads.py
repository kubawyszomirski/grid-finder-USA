#!/usr/bin/env python3
"""
predict_roads.py
Run inference with a trained roads model on the full prediction set.

Reads:
    {project}/models_roads/roads_model_{N}.ubj          — trained model
    {project}/models_roads/roads_model_{N}_meta.json    — feature list
    {project}/roads/preprocessed/X_predict.parquet      — scaled features
    {project}/roads/preprocessed/meta_predict.parquet   — road_id index
    raw_data/{state}/TRANSPORT/{state}_roads_features.gpkg — original geometries

Writes to {project}/roads/predictions/:
    roads_model_{N}_predictions.parquet   — road_id + y_proba (no geometry)
    roads_model_{N}_predictions.gpkg      — road_id + y_proba + geometry (for QGIS)

Usage:
    python predict_roads.py --project project-ma --model 0
    python predict_roads.py --project project-ma --model 0 --no-geo
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from generate.generate_utils import BASE_DIR, FINAL_CRS

warnings.filterwarnings("ignore")


# ==========================================
#            CONFIGURATION
# ==========================================

PROJECT     = "project-ma"
MODEL_INDEX = 0


# ==========================================
#               HELPERS
# ==========================================

def _load_model(model_path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def _align_features(X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Reindex X to exactly the columns the model was trained on.
    Missing columns are filled with 0; extra columns are dropped.
    """
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        print(f"  [warn] {len(missing)} feature(s) missing from X_predict — filling with 0: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    return X.reindex(columns=feature_cols, fill_value=0)


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run roads model inference on the full prediction set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict_roads.py --project project-ma --model 0\n"
            "  python predict_roads.py --project project-ma --model 0 --no-geo\n"
        ),
    )
    parser.add_argument("--project", default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--model", type=int, default=MODEL_INDEX,
                        help="Model index to load (default: %(default)s)")
    parser.add_argument("--no-geo", action="store_true",
                        help="Skip joining geometry; output parquet only")
    args = parser.parse_args()

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    model_stem  = f"roads_model_{args.model}"
    models_dir  = project_dir / "models_roads"
    prep_dir    = project_dir / "roads" / "preprocessed"
    out_dir     = project_dir / "roads" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Predict Roads — {args.project}  ←  {model_stem}")
    print(f"{'='*60}")

    # ---- Load model ----
    model_path = models_dir / f"{model_stem}.ubj"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    print(f"\nLoading model from {model_path} …")
    model = _load_model(model_path)

    # ---- Load model meta (feature list + state) ----
    meta_path = models_dir / f"{model_stem}_meta.json"
    if not meta_path.exists():
        print(f"[ERROR] Model meta not found: {meta_path}")
        return
    model_meta = json.loads(meta_path.read_text())
    feature_cols: list[str] = model_meta["feature_cols"]
    state: str | None = model_meta.get("state")
    print(f"  Features expected: {len(feature_cols)}")
    print(f"  Trained on state : {state or 'unknown'}")
    print(f"  Val ROC-AUC      : {model_meta.get('val_roc_auc', 'n/a')}")
    print(f"  Val PR-AUC       : {model_meta.get('val_pr_auc', 'n/a')}")

    # ---- Load preprocessed predict set ----
    X_path    = prep_dir / "X_predict.parquet"
    meta_p    = prep_dir / "meta_predict.parquet"
    if not X_path.exists():
        print(f"[ERROR] X_predict.parquet not found — run preprocess-roads.py first.")
        return
    print(f"\nLoading {X_path.name} …")
    X_predict = pd.read_parquet(X_path)
    meta_predict = pd.read_parquet(meta_p) if meta_p.exists() else pd.DataFrame(index=X_predict.index)
    print(f"  {len(X_predict):,} road segments, {X_predict.shape[1]} features")

    # ---- Align features ----
    X_aligned = _align_features(X_predict, feature_cols)

    # ---- Inference ----
    print("\nRunning inference …")
    y_proba = model.predict_proba(X_aligned)[:, 1].astype(np.float32)
    print(f"  Predictions: min={y_proba.min():.4f}  mean={y_proba.mean():.4f}  max={y_proba.max():.4f}")

    # ---- Build result DataFrame ----
    result = pd.DataFrame({"y_proba": y_proba}, index=X_predict.index)
    if "road_id" in meta_predict.columns:
        result.insert(0, "road_id", meta_predict["road_id"].values)

    # ---- Save parquet (no geometry) ----
    parquet_out = out_dir / f"{model_stem}_predictions.parquet"
    result.to_parquet(parquet_out, index=False)
    print(f"\n  Saved parquet → {parquet_out}")

    # ---- Join geometry and save GeoPackage ----
    if args.no_geo:
        print("  [--no-geo] Skipping GeoPackage output.")
    elif state is None:
        print("  [warn] State unknown — cannot locate roads geometry. Skipping GeoPackage.")
    else:
        roads_geo_path = BASE_DIR / "raw_data" / state / "TRANSPORT" / f"{state}_roads_features.gpkg"
        if not roads_geo_path.exists():
            print(f"  [warn] Roads geometry not found at {roads_geo_path} — skipping GeoPackage.")
        else:
            print(f"\nJoining geometry from {roads_geo_path.name} …")
            roads_gdf = gpd.read_file(roads_geo_path)
            if "road_id" in result.columns and "road_id" in roads_gdf.columns:
                merged = roads_gdf[["road_id", "geometry"]].merge(result, on="road_id", how="inner")
            else:
                print("  [warn] No road_id to join on — attaching geometry by position.")
                merged = roads_gdf[["geometry"]].copy()
                for col in result.columns:
                    merged[col] = result[col].values[: len(merged)]

            merged = gpd.GeoDataFrame(merged, geometry="geometry")
            if merged.crs is None or merged.crs.to_epsg() != 4326:
                merged = merged.to_crs(FINAL_CRS)

            gpkg_out = out_dir / f"{model_stem}_predictions.gpkg"
            merged.to_file(gpkg_out, driver="GPKG")
            print(f"  Saved GeoPackage ({len(merged):,} roads) → {gpkg_out}")

    print(f"\n{'='*60}")
    print(f" Done — {len(result):,} predictions written.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
