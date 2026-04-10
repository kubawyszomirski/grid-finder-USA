#!/usr/bin/env python3
"""
predict_roads.py
Run inference with a trained roads model on the full prediction set.

Two modes:
  default  — predict on the full road set using the full model
  --oof    — join OOF training predictions (generated during training) with
             road segment geometry and write a GeoPackage

Reads:
    {project}/models_roads/roads_model_{N}.ubj          — trained model
    {project}/models_roads/roads_model_{N}_meta.json    — feature list + state
    {project}/models_roads/roads_model_{N}_oof_preds.parquet — OOF preds (--oof)
    {project}/roads/preprocessed/X_predict.parquet      — scaled features
    {project}/roads/preprocessed/meta_predict.parquet   — road_id index
    raw_data/{state}/TRANSPORT/{state}_roads_features.gpkg — original geometries

Writes to {project}/roads/predictions/:
    roads_model_{N}_predictions.parquet   — road_id + y_proba
    roads_model_{N}_predictions.gpkg      — road_id + y_proba + geometry
    roads_model_{N}_oof_preds.gpkg        — OOF preds + geometry (--oof)

Usage:
    python predict_roads.py --project project-ma --model 0
    python predict_roads.py --project project-ma --model 0 --oof
    python predict_roads.py --project project-ma --model 0 --no-geo
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        print(f"  [warn] {len(missing)} feature(s) missing from X_predict — filling with 0: "
              f"{missing[:5]}{'…' if len(missing) > 5 else ''}")
    return X.reindex(columns=feature_cols, fill_value=0)


def _join_geometry(
    result: pd.DataFrame, state: str, id_col: str, project_dir: Path,
) -> gpd.GeoDataFrame | None:
    """
    Load road geometry from {project}/roads/ and merge with result on id_col.

    Loads BOTH training and prediction GeoPackages (whichever exist) so that
    OOF results (training road_ids) and predict results (prediction road_ids)
    both resolve to the correct geometry.
    """
    roads_dir = project_dir / "roads"
    geo_parts = []
    for fname in [f"{state}_roads_training.gpkg", f"{state}_roads_prediction.gpkg"]:
        p = roads_dir / fname
        if p.exists():
            gdf = gpd.read_file(p)
            if id_col in gdf.columns:
                geo_parts.append(gdf[[id_col, "geometry"]])
                print(f"\nLoaded geometry: {fname}  ({len(gdf):,} roads)")

    if not geo_parts:
        print(f"  [warn] No roads geometry files found in {roads_dir} — skipping GeoPackage.")
        return None

    roads_gdf = gpd.GeoDataFrame(
        pd.concat(geo_parts, ignore_index=True), geometry="geometry"
    )

    merged = roads_gdf.merge(result, on=id_col, how="inner")
    merged = gpd.GeoDataFrame(merged, geometry="geometry")
    if merged.crs is None or merged.crs.to_epsg() != 4326:
        merged = merged.to_crs(FINAL_CRS)
    return merged


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
            "  python predict_roads.py --project project-ma --model 0 --oof\n"
            "  python predict_roads.py --project project-ma --model 0 --no-geo\n"
        ),
    )
    parser.add_argument("--project", default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--model", type=int, default=MODEL_INDEX,
                        help="Model index to load (default: %(default)s)")
    parser.add_argument("--oof",    action="store_true",
                        help="Output OOF training predictions instead of predicting on predict set")
    parser.add_argument("--no-geo", action="store_true",
                        help="Skip joining geometry; output parquet only")
    args = parser.parse_args()

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    model_stem = f"roads_model_{args.model}"
    models_dir = project_dir / "models_roads"
    prep_dir   = project_dir / "roads" / "preprocessed"
    out_dir    = project_dir / "roads" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Predict Roads — {args.project}  ←  {model_stem}")
    print(f"{'='*60}")

    # ---- Load model meta (shared by both modes) ----
    meta_path = models_dir / f"{model_stem}_meta.json"
    if not meta_path.exists():
        print(f"[ERROR] Model meta not found: {meta_path}")
        return
    model_meta   = json.loads(meta_path.read_text())
    feature_cols = model_meta["feature_cols"]
    state: str | None = model_meta.get("state")

    print(f"  Features expected : {len(feature_cols)}")
    print(f"  Trained on state  : {state or 'unknown'}")
    oof_roc = model_meta.get("oof_roc_auc", model_meta.get("val_roc_auc", "n/a"))
    oof_pr  = model_meta.get("oof_pr_auc",  model_meta.get("val_pr_auc",  "n/a"))
    print(f"  OOF ROC-AUC       : {oof_roc}")
    print(f"  OOF PR-AUC        : {oof_pr}")

    # ==================================================================
    # OOF MODE — join saved OOF predictions with road segment geometry
    # ==================================================================
    if args.oof:
        oof_parquet = models_dir / f"{model_stem}_oof_preds.parquet"
        if not oof_parquet.exists():
            print(f"[ERROR] OOF predictions not found: {oof_parquet}")
            return
        oof_df = pd.read_parquet(oof_parquet)
        print(f"\n  {len(oof_df):,} OOF predictions loaded")

        # Always save parquet to predictions folder
        parquet_out = out_dir / f"{model_stem}_oof_preds.parquet"
        oof_df.to_parquet(parquet_out, index=False)
        print(f"  Saved OOF parquet → {parquet_out}")

        if not args.no_geo:
            if state is None:
                print("  [warn] State unknown — cannot locate road geometry. Skipping GeoPackage.")
            else:
                merged = _join_geometry(oof_df, state, "road_id", project_dir)
                if merged is not None:
                    gpkg_out = out_dir / f"{model_stem}_oof_preds.gpkg"
                    merged.to_file(gpkg_out, driver="GPKG")
                    print(f"  Saved OOF GeoPackage ({len(merged):,} roads) → {gpkg_out}")

        print(f"\n{'='*60}")
        print(f" Done — {len(oof_df):,} OOF predictions written.")
        print(f"{'='*60}")
        return

    # ==================================================================
    # DEFAULT MODE — predict on full road set using the full model
    # ==================================================================

    # ---- Load model ----
    model_path = models_dir / f"{model_stem}.ubj"
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    print(f"\nLoading model from {model_path} …")
    model = _load_model(model_path)

    # ---- Load preprocessed predict set ----
    X_path = prep_dir / "X_predict.parquet"
    meta_p = prep_dir / "meta_predict.parquet"
    if not X_path.exists():
        print(f"[ERROR] X_predict.parquet not found — run preprocess_roads.py first.")
        return
    print(f"\nLoading {X_path.name} …")
    X_predict    = pd.read_parquet(X_path)
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

    # ---- Save parquet ----
    parquet_out = out_dir / f"{model_stem}_predictions.parquet"
    result.to_parquet(parquet_out, index=False)
    print(f"\n  Saved parquet → {parquet_out}")

    # ---- Join geometry and save GeoPackage ----
    if args.no_geo:
        print("  [--no-geo] Skipping GeoPackage output.")
    elif state is None:
        print("  [warn] State unknown — cannot locate road geometry. Skipping GeoPackage.")
    else:
        merged = _join_geometry(result, state, "road_id", project_dir)
        if merged is not None:
            gpkg_out = out_dir / f"{model_stem}_predictions.gpkg"
            merged.to_file(gpkg_out, driver="GPKG")
            print(f"  Saved GeoPackage ({len(merged):,} roads) → {gpkg_out}")

    print(f"\n{'='*60}")
    print(f" Done — {len(result):,} predictions written.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
