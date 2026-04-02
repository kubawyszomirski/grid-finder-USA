#!/usr/bin/env python3
"""
preprocess-tiles-raster.py
Converts tile features into spatial tensor arrays for U-Net / CNN training.

Reads:
    {project}/{tiles_folder}/training_tiles.gpkg   → patches + spatial block split
    {project}/{tiles_folder}/prediction_tiles.gpkg → full raster volume for inference

Writes to {project}/{tiles_folder}/raster/:
    X_train.npy              — (N, C, H, W) float32 training patches
    y_train.npy              — (N, 1, H, W) float32 label patches
    X_val.npy                — (N, C, H, W) float32 validation patches
    y_val.npy                — (N, 1, H, W) float32
    X_predict.npy            — (rows, cols, C) float32 full prediction raster
    predict_meta.json        — spatial extent + CRS for georeferencing predictions
    channel_names.json       — [channel_index → feature_name]
    raster_config.json       — all parameters + normalization stats
    debug_train_tiles.geojson
    debug_val_tiles.geojson

Raster parameters (all optional, sensible defaults):
    --patch-size   N     tiles per patch side                [default: 32]
    --block-size   N     tiles per spatial-block side        [default: 64]
    --stride       N     patch extraction step (augmentation)[default: 16]
    --val-frac     F     fraction of blocks for validation   [default: 0.15]
    --min-coverage F     min valid-tile fraction per patch   [default: 0.40]
    --tile-size-m  N     tile edge length in metres          [default: parsed from folder name, else 500]
    --seed         N     random seed                         [default: 42]
    --no-debug           skip saving debug split GeoJSONs

Usage:
    python preprocess-tiles-raster.py project-2 500_tiles
    python preprocess-tiles-raster.py project-2 500_tiles --patch-size 16 --stride 8
    python preprocess-tiles-raster.py project-2 500_tiles --block-size 32 --val-frac 0.20 --no-debug
"""
from __future__ import annotations

import argparse
import json
import random
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================
#            CONFIGURATION
# ==========================================

BASE_DIR   = Path(".")
METRIC_CRS = "EPSG:5070"
FINAL_CRS  = "EPSG:4326"
LABEL_COL  = "grid"

META_COLS = {"tile_id", "state", "geometry", "grid"}


# ==========================================
#              HELPERS
# ==========================================

def load_tiles(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        print(f"  [not found] {path}")
        return None
    gdf = gpd.read_file(path).to_crs(METRIC_CRS)
    print(f"  {path.name}: {len(gdf):,} tiles loaded")
    return gdf


def parse_tile_size(tiles_dir: str, override: int | None) -> int:
    """Parse tile size from folder name (e.g. '500_tiles' → 500) or use override."""
    if override is not None:
        return override
    m = re.match(r"^(\d+)_tiles$", tiles_dir)
    if m:
        return int(m.group(1))
    print(f"  [Warning] could not parse tile size from '{tiles_dir}', defaulting to 500 m")
    return 500


def detect_feature_cols(gdf: gpd.GeoDataFrame) -> list[str]:
    """All numeric columns that are not metadata / label."""
    return [
        c for c in gdf.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(gdf[c])
    ]


def _is_binary(series: pd.Series) -> bool:
    return set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0})


def build_norm_stats(gdf: gpd.GeoDataFrame, feature_cols: list[str]) -> dict:
    """
    Compute per-column normalization parameters from training data.
    Three strategies (all output [0, 1]):
      binary   — already 0/1, no transform
      pct      — divide by 100
      log_scale — log1p, then divide by 99th percentile of the log values
    """
    stats: dict[str, dict] = {}
    for col in feature_cols:
        s = gdf[col].fillna(0)
        if _is_binary(s):
            stats[col] = {"type": "binary"}
        elif "pct" in col.lower() or "score" in col.lower():
            stats[col] = {"type": "pct"}
        else:
            log_vals = np.log1p(np.clip(s.values, 0, None))
            p99 = float(np.percentile(log_vals, 99))
            stats[col] = {"type": "log_scale", "p99": p99 if p99 > 0 else 1.0}
    return stats


def apply_norm(values: np.ndarray, stat: dict) -> np.ndarray:
    kind = stat["type"]
    if kind == "binary":
        return np.clip(values, 0, 1).astype(np.float32)
    if kind == "pct":
        return np.clip(values / 100.0, 0, 1).astype(np.float32)
    # log_scale
    log_v = np.log1p(np.clip(values, 0, None))
    return np.clip(log_v / stat["p99"], 0, 1).astype(np.float32)


def assign_grid_indices(
    gdf: gpd.GeoDataFrame, tile_size_m: int
) -> tuple[gpd.GeoDataFrame, int, int]:
    """
    Assign integer (row_idx, col_idx) to each tile based on spatial position.
    Returns the augmented GDF plus (n_rows, n_cols).
    """
    centroids = gdf.geometry.centroid
    minx = centroids.x.min()
    maxy = centroids.y.max()
    gdf = gdf.copy()
    gdf["col_idx"] = np.floor((centroids.x - minx) / tile_size_m).astype(int)
    gdf["row_idx"] = np.floor((maxy - centroids.y) / tile_size_m).astype(int)
    n_rows = int(gdf["row_idx"].max()) + 1
    n_cols = int(gdf["col_idx"].max()) + 1
    return gdf, n_rows, n_cols


def build_split_mask(
    gdf: gpd.GeoDataFrame,
    n_rows: int,
    n_cols: int,
    block_size: int,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, set]:
    """
    Spatial block-based train/val split.
    Returns (split_mask, val_blocks_set) where mask: 0=train, 1=val.
    """
    gdf = gdf.copy()
    gdf["block_r"] = gdf["row_idx"] // block_size
    gdf["block_c"] = gdf["col_idx"] // block_size

    unique_blocks = gdf[["block_r", "block_c"]].drop_duplicates().values.tolist()
    random.seed(seed)
    random.shuffle(unique_blocks)
    n_val = max(1, int(len(unique_blocks) * val_frac))
    val_blocks_set = {tuple(b) for b in unique_blocks[:n_val]}

    print(f"  Spatial blocks: {len(unique_blocks)} total, {n_val} validation")

    max_br = n_rows // block_size + 1
    max_bc = n_cols // block_size + 1
    block_lookup = np.zeros((max_br, max_bc), dtype=np.uint8)
    for br, bc in val_blocks_set:
        block_lookup[br, bc] = 1

    full_map  = block_lookup.repeat(block_size, axis=0).repeat(block_size, axis=1)
    split_mask = full_map[:n_rows, :n_cols]
    return split_mask, val_blocks_set


def fill_volume(
    gdf: gpd.GeoDataFrame,
    feature_cols: list[str],
    norm_stats: dict,
    n_rows: int,
    n_cols: int,
    has_label: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize tile features into a (n_rows, n_cols, n_channels) volume.
    Returns (X_vol, y_vol, valid_mask).
    """
    n_ch   = len(feature_cols)
    X_vol  = np.zeros((n_rows, n_cols, n_ch), dtype=np.float32)
    y_vol  = np.zeros((n_rows, n_cols),        dtype=np.float32)
    valid  = np.zeros((n_rows, n_cols),        dtype=np.uint8)

    rows = gdf["row_idx"].values
    cols = gdf["col_idx"].values

    for i, col in enumerate(feature_cols):
        vals = gdf[col].fillna(0).values.astype(np.float32)
        X_vol[rows, cols, i] = apply_norm(vals, norm_stats[col])

    if has_label and LABEL_COL in gdf.columns:
        y_vol[rows, cols] = gdf[LABEL_COL].fillna(0).values.astype(np.float32)

    valid[rows, cols] = 1
    return X_vol, y_vol, valid


def extract_patches(
    X_vol: np.ndarray,
    y_vol: np.ndarray,
    valid: np.ndarray,
    split_mask: np.ndarray,
    patch_size: int,
    stride: int,
    min_coverage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Sliding-window patch extraction with strict split purity check.
    Returns X_train, y_train, X_val, y_val, n_dropped.
    """
    n_rows, n_cols, _ = X_vol.shape
    X_tr, y_tr = [], []
    X_vl, y_vl = [], []
    dropped = 0

    for r in range(0, n_rows - patch_size + 1, stride):
        for c in range(0, n_cols - patch_size + 1, stride):
            # Coverage check
            if np.mean(valid[r:r + patch_size, c:c + patch_size]) < min_coverage:
                continue

            # Purity check — no cross-border patches
            sp = split_mask[r:r + patch_size, c:c + patch_size]
            is_train = np.all(sp == 0)
            is_val   = np.all(sp == 1)
            if not is_train and not is_val:
                dropped += 1
                continue

            px = X_vol[r:r + patch_size, c:c + patch_size, :]   # (H, W, C)
            py = y_vol[r:r + patch_size, c:c + patch_size]       # (H, W)
            if is_val:
                X_vl.append(px); y_vl.append(py)
            else:
                X_tr.append(px); y_tr.append(py)

    def _stack(lst):
        arr = np.array(lst, dtype=np.float32)
        return arr.transpose(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)

    def _stack_y(lst):
        return np.array(lst, dtype=np.float32)[:, np.newaxis, :, :]  # (N, 1, H, W)

    return _stack(X_tr), _stack_y(y_tr), _stack(X_vl), _stack_y(y_vl), dropped


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize tile features into spatial tensors for CNN/U-Net training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preprocess-tiles-raster.py project-2 500_tiles\n"
            "  python preprocess-tiles-raster.py project-2 500_tiles --patch-size 16 --stride 8\n"
            "  python preprocess-tiles-raster.py project-2 500_tiles --block-size 32 --val-frac 0.20\n"
        ),
    )
    parser.add_argument("project",   help="Project folder name, e.g. 'project-2'")
    parser.add_argument("tiles_dir", help="Tile folder inside the project, e.g. '500_tiles'")

    parser.add_argument("--patch-size",   type=int,   default=32,   metavar="N",
                        help="Tiles per patch side (default: 32)")
    parser.add_argument("--block-size",   type=int,   default=64,   metavar="N",
                        help="Tiles per spatial-block side for train/val split (default: 64)")
    parser.add_argument("--stride",       type=int,   default=16,   metavar="N",
                        help="Patch extraction stride — smaller = more augmentation (default: 16)")
    parser.add_argument("--val-frac",     type=float, default=0.15, metavar="F",
                        help="Fraction of spatial blocks held out for validation (default: 0.15)")
    parser.add_argument("--min-coverage", type=float, default=0.40, metavar="F",
                        help="Min fraction of valid tiles required per patch (default: 0.40)")
    parser.add_argument("--tile-size-m",  type=int,   default=None, metavar="N",
                        help="Tile edge length in metres (default: parsed from folder name)")
    parser.add_argument("--seed",         type=int,   default=42,   metavar="N",
                        help="Random seed (default: 42)")
    parser.add_argument("--no-debug",     action="store_true",
                        help="Skip saving debug split GeoJSONs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    tiles_path = BASE_DIR / args.project / args.tiles_dir
    if not tiles_path.exists():
        print(f"[ERROR] Tiles directory not found: {tiles_path}")
        return

    out_dir = tiles_path / "raster"
    out_dir.mkdir(exist_ok=True)

    tile_size_m = parse_tile_size(args.tiles_dir, args.tile_size_m)
    print(f"\nTile size: {tile_size_m} m  |  Patch: {args.patch_size}×{args.patch_size} tiles"
          f"  ({args.patch_size * tile_size_m / 1000:.1f} km side)")
    print(f"Block: {args.block_size}×{args.block_size} tiles"
          f"  ({args.block_size * tile_size_m / 1000:.1f} km side)")
    print(f"Stride: {args.stride}  |  Val frac: {args.val_frac}  |  Min coverage: {args.min_coverage}")

    # ---- Load ----
    print("\nLoading tiles…")
    gdf_train   = load_tiles(tiles_path / "training_tiles.gpkg")
    gdf_predict = load_tiles(tiles_path / "prediction_tiles.gpkg")

    if gdf_train is None:
        print("[ERROR] training_tiles.gpkg not found — cannot continue.")
        return

    # ---- Feature columns ----
    feature_cols = detect_feature_cols(gdf_train)
    print(f"\nFeature channels: {len(feature_cols)}")

    # ---- Normalization stats (fit on training data only) ----
    print("Computing normalization stats from training data…")
    norm_stats = build_norm_stats(gdf_train, feature_cols)
    n_log    = sum(1 for s in norm_stats.values() if s["type"] == "log_scale")
    n_pct    = sum(1 for s in norm_stats.values() if s["type"] == "pct")
    n_bin    = sum(1 for s in norm_stats.values() if s["type"] == "binary")
    print(f"  binary: {n_bin}  |  pct: {n_pct}  |  log_scale: {n_log}")

    # ---- Assign spatial grid indices ----
    print("\nAssigning grid indices…")
    gdf_train, n_rows, n_cols = assign_grid_indices(gdf_train, tile_size_m)
    print(f"  Training raster: {n_rows} × {n_cols} tiles")

    # ---- Spatial block split ----
    print("Building spatial train/val split…")
    split_mask, val_blocks_set = build_split_mask(
        gdf_train, n_rows, n_cols, args.block_size, args.val_frac, args.seed
    )

    # ---- Fill training volume ----
    print("Filling training data volume…")
    X_vol, y_vol, valid_mask = fill_volume(
        gdf_train, feature_cols, norm_stats, n_rows, n_cols, has_label=True
    )
    total_tiles = int(valid_mask.sum())
    pos_tiles   = int(y_vol[valid_mask.astype(bool)].sum())
    print(f"  Valid tiles: {total_tiles:,}  |  Positive (grid=1): {pos_tiles:,}"
          f"  ({pos_tiles / max(total_tiles, 1) * 100:.1f}%)")

    # ---- Patch extraction ----
    print(f"\nExtracting patches (patch={args.patch_size}, stride={args.stride})…")
    X_train, y_train, X_val, y_val, n_dropped = extract_patches(
        X_vol, y_vol, valid_mask, split_mask,
        args.patch_size, args.stride, args.min_coverage,
    )
    print(f"  Dropped {n_dropped:,} cross-border patches")

    if X_train.size == 0 or X_val.size == 0:
        print("[ERROR] One or both splits are empty — try smaller --block-size or --patch-size.")
        return

    print(f"\n  Train: {X_train.shape}  labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape}    labels: {y_val.shape}")

    # ---- Save training tensors ----
    print("\nSaving training tensors…")
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_val.npy",   y_val)

    # ---- Prediction volume ----
    if gdf_predict is not None:
        print("\nBuilding prediction volume…")
        gdf_predict, pr_rows, pr_cols = assign_grid_indices(gdf_predict, tile_size_m)
        print(f"  Prediction raster: {pr_rows} × {pr_cols} tiles")
        X_pred_vol, _, _ = fill_volume(
            gdf_predict, feature_cols, norm_stats, pr_rows, pr_cols, has_label=False
        )
        np.save(out_dir / "X_predict.npy", X_pred_vol)
        print(f"  X_predict: {X_pred_vol.shape}")

        bounds = gdf_predict.total_bounds.tolist()
        predict_meta = {
            "crs":        METRIC_CRS,
            "bounds":     {"minx": bounds[0], "miny": bounds[1],
                           "maxx": bounds[2], "maxy": bounds[3]},
            "n_rows":     pr_rows,
            "n_cols":     pr_cols,
            "tile_size_m": tile_size_m,
            "feature_cols": feature_cols,
        }
        with open(out_dir / "predict_meta.json", "w") as f:
            json.dump(predict_meta, f, indent=2)

    # ---- Channel names ----
    with open(out_dir / "channel_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ---- Raster config ----
    config = {
        "patch_size":    args.patch_size,
        "block_size":    args.block_size,
        "stride":        args.stride,
        "val_frac":      args.val_frac,
        "min_coverage":  args.min_coverage,
        "tile_size_m":   tile_size_m,
        "seed":          args.seed,
        "metric_crs":    METRIC_CRS,
        "n_channels":    len(feature_cols),
        "feature_cols":  feature_cols,
        "norm_stats":    norm_stats,
        "n_train_patches": int(X_train.shape[0]),
        "n_val_patches":   int(X_val.shape[0]),
        "n_dropped_patches": n_dropped,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_val":   float(y_val.mean()),
        "training_raster_shape": [n_rows, n_cols],
    }
    with open(out_dir / "raster_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Debug split GeoJSONs ----
    if not args.no_debug:
        print("\nSaving debug split GeoJSONs…")
        gdf_train["is_val"] = gdf_train.apply(
            lambda row: (int(row["row_idx"]) // args.block_size,
                         int(row["col_idx"]) // args.block_size) in val_blocks_set,
            axis=1,
        )
        gdf_out = gdf_train[[c for c in ("tile_id", "state", "grid", "is_val", "geometry")
                              if c in gdf_train.columns]].to_crs(FINAL_CRS)
        gdf_out[~gdf_out["is_val"]].to_file(out_dir / "debug_train_tiles.geojson", driver="GeoJSON")
        gdf_out[ gdf_out["is_val"]].to_file(out_dir / "debug_val_tiles.geojson",   driver="GeoJSON")

    print(f"\n[DONE] → {out_dir}")
    print(f"  Channels : {len(feature_cols)}")
    print(f"  Train    : {X_train.shape[0]:,} patches")
    print(f"  Val      : {X_val.shape[0]:,}  patches")
    if gdf_predict is not None:
        print(f"  Predict  : {X_pred_vol.shape[:2]} volume")


if __name__ == "__main__":
    main()
