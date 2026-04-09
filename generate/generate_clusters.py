#!/usr/bin/env python3
"""
generate-clusters.py
Full cluster pipeline for a single state within a project:

  Phase 1 — Generate labeled training points (smart sampling near grid
             lines + roads + background, clipped to training range)
  Phase 2 — Assign ML features to training points
  Phase 3 — Generate prediction candidates (high-res near roads/buildings,
             low-res elsewhere, clipped to project prediction_extent)
  Phase 4 — Assign ML features to prediction candidates
  Phase 5 — Train XGBoost (spatial CV) and predict grid_prob
  Phase 6 — HDBSCAN cluster polygons   (high prob → grid areas)
  Phase 7 — HDBSCAN anti-cluster polygons (low prob → no-grid areas)
  Phase 8 — minor cluster polygons (moderate prob + artificial)

Reads:
  grid_data/3_phase/<S>_3phase.parquet              grid lines
  grid_ranges/3_phase/<S>_3phase_range.parquet      training area
  <project>/prediction_extent.gpkg                  prediction area
  raw_data/<S>/TRANSPORT/<S>_roads_raw.gpkg
  raw_data/<S>/BUILDINGS/<S>_fema_buildings.geojson
  raw_data/<S>/EXCLUSIONS/<S>_exclusions.gpkg
  raw_data/<S>/...                                  all infra + raster layers

Writes (to <project>/clusters/):
  <S>_training_features.parquet
  <S>_prediction_features.parquet
  <S>_points_predicted.parquet
  <S>_cluster_polygons.geojson
  <S>_anti_cluster_polygons.geojson
  <S>_cluster_polygons_minor.geojson

Usage:
  python generate-clusters.py Massachusetts
  python generate-clusters.py MA --force
  python generate-clusters.py MA --project project-ma --force
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import HDBSCAN
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb

from .generate_utils import (
    METRIC_CRS, FINAL_CRS, BASE_DIR, STATE_FULL_NAMES, _NAME_TO_ABBREV,
    CDL_3PHASE_CODES, NLCD_DEVELOPED, NLCD_NATURE, NLCD_CROPS,
    _find, load_geo,
)

warnings.filterwarnings("ignore")

_STATE_BORDERS_S3    = "s3://grid-research-raw-data/USA/state borders/cb_2023_us_state_20m.gpkg"
_STATE_BORDERS_LOCAL = BASE_DIR / "raw_data" / "_shared" / "cb_2023_us_state_20m.gpkg"


def _ensure_state_borders() -> Path:
    if not _STATE_BORDERS_LOCAL.exists():
        _STATE_BORDERS_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        bucket, key = _STATE_BORDERS_S3[5:].split("/", 1)
        boto3.client("s3").download_file(bucket, key, str(_STATE_BORDERS_LOCAL))
    return _STATE_BORDERS_LOCAL


def get_state_border_geom(state: str):
    states = gpd.read_file(_ensure_state_borders())
    name = STATE_FULL_NAMES[state]
    border = states[states["NAME"] == name].to_crs(METRIC_CRS)
    return border.geometry.unary_union


# ==========================================
#               CONFIGURATION
# ==========================================

# ---- Training sampling (per 1,000 sq km of training area) ----
N_GRID_SAMPLES_PER_1K_KM2 = 800
N_ROAD_SAMPLES_PER_1K_KM2 = 800
N_BG_SAMPLES_PER_1K_KM2   = 500
TRAINING_POINT_MULTIPLIER  = 1.5    # scale all sampling counts without changing ratios

# ---- Dynamic buffer (scales with building density) ----
DENSITY_RADIUS_M       = 1_000
BUFFER_URBAN_M         =   100
BUFFER_RURAL_M         =   450
DENSITY_THRESHOLD_HIGH =   600   # buildings within 1 km → fully urban

# ---- Prediction candidate grid ----
RES_HIGH_M = 150    # near roads/buildings
RES_LOW_M  = 1_000  # everywhere else
BUF_ROAD_M = 300
BUF_BLDG_M = 200

# ---- XGBoost ----
XGB_PARAMS = {
    "learning_rate":    0.02,
    "max_depth":        8,
    "subsample":        0.65,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.5,
    "reg_lambda":       2.0,
    "objective":        "binary:logistic",
    "n_jobs":           -1,
    "random_state":     42,
    "min_child_weight": 5,
    "eval_metric":      "auc",
}

# ---- Clustering ----
PROB_THRESHOLD        = 0.89   # grid clusters: high probability
MINOR_PROB_THRESHOLD  = 0.65   # minor clusters: moderate probability
MAX_PROB_ANTI         = 0.08   # anti-clusters: low probability
MIN_CLUSTER_SIZE      = 2
MIN_CLUSTER_SIZE_ANTI = 5
MIN_SAMPLES_CLUSTER   = 5
MIN_SAMPLES_ANTI      = 10
HULL_RATIO            = 0.25
CLUSTER_FALLBACK_BUF  = 150    # expand degenerate cluster hulls (small_buf)
CLUSTER_HULL_FALLBACK = 600    # buffer for fully-failed concave hull (fallback_buf)
ANTI_FALLBACK_BUF     = 500
ANTI_SHRINK_M         = 150    # shrink anti-cluster hull inward
ARTIFICIAL_BUF_M      = 100    # buffer around point/polygon features for artificial clusters

# ---- Model training ----
VALIDATION_SPLIT = 0.2   # fraction of training points held out for early stopping

PROJECT = "project-ma"
PREDICTION_STATES = ["MA"]
TRAINING_STATES   = ["MA"]
FORCE = False
EXCLUDE_GRID_FROM_PREDICTION = False  # subtract 3-phase grid range from prediction extent

# Infrastructure tasks: key → list of actions in ["dist", "count", "area"]
INFRA_TASKS: dict[str, list[str]] = {
    "solar":                      ["dist", "area"],
    "wind":                       ["dist"],
    "gas_and_hydro":              ["dist"],
    "bess_charging_stations":     ["dist", "count"],
    "oil_gas_chemical":           ["dist", "count"],
    "mining":                     ["dist", "count"],
    "utilities":                  ["dist", "count"],
    "industry":                   ["dist", "count"],
    "frs_primary":                ["dist", "count"],
    "frs_secondary":              ["dist", "count"],
    "towers_major":               ["dist"],
    "towers_minor":               ["dist"],
    "agriculture":                ["dist", "count"],
    "works":                      ["dist", "count"],
    "public_infra":               ["dist", "count"],
    "big_buildings_and_clusters": ["dist", "count"],
    "railway":                    ["dist"],
    "substations":                ["dist"],
    "dams":                       ["dist"],
    "hospitality":                ["dist"],
    "transmission_lines":         ["dist"],
}

# Keys whose counts fold into the single composite column
COMPOSITE_LAYERS = {
    "bess_charging_stations", "oil_gas_chemical", "mining", "utilities",
    "industry", "frs_primary", "frs_secondary", "agriculture", "public_infra",
}

EXCLUDE_COLS = {
    "geometry", "source", "index_right", "index_left",
    "id", "grid",
    "dist_to_grid", "dynamic_threshold", "bldg_count_1km",
}


# ==========================================
#            PATH RESOLUTION
# ==========================================

def get_state_paths(s: str) -> dict:
    raw = BASE_DIR / "raw_data" / s
    return {
        # Core area + grid lines
        "training_range":     BASE_DIR / "grid_ranges" / "3_phase" / f"{s}_3phase_range.parquet",
        "grid_range_parquet": BASE_DIR / "grid_ranges" / "3_phase" / f"{s}_3phase_range.parquet",
        "grid_lines":         BASE_DIR / "grid_data"   / "3_phase" / f"{s}_3phase.parquet",
        # Specified by user
        "roads":          raw / "TRANSPORT" / f"{s}_roads_raw.gpkg",
        "buildings":      raw / "BUILDINGS" / f"{s}_fema_buildings.geojson",
        "exclusions":     raw / "EXCLUSIONS" / f"{s}_exclusions.gpkg",
        # Infrastructure vectors
        "solar":                      _find(raw / "GENERATORS",  f"{s}_solar_merged"),
        "wind":                       _find(raw / "GENERATORS",  f"{s}_wind_merged"),
        "gas_and_hydro":              _find(raw / "GENERATORS",  f"{s}_eia_gas_hydro"),
        "bess_charging_stations":     _find(raw / "GENERATORS",  f"{s}_bess_charging_stations"),
        "oil_gas_chemical":           _find(raw / "INDUSTRY",    f"{s}_oil_gas_chemical"),
        "industry":                   _find(raw / "INDUSTRY",    f"{s}_industry"),
        "works":                      _find(raw / "INDUSTRY",    f"{s}_works"),
        "mining":                     _find(raw / "MINING",      f"{s}_mining_final"),
        "utilities":                  _find(raw / "UTILITIES",   f"{s}_utilities_merged"),
        "frs_primary":                _find(raw / "FRS",         f"{s}_frs_primary_merged"),
        "frs_secondary":              _find(raw / "FRS",         f"{s}_frs_secondary_merged"),
        "towers_major":               _find(raw / "TELECOM",     f"{s}_towers_major"),
        "towers_minor":               _find(raw / "TELECOM",     f"{s}_towers_minor"),
        "agriculture":                _find(raw / "AGRICULTURE", f"{s}_ag_farms_merged"),
        "public_infra":               _find(raw / "PUBLIC",      f"{s}_public_infra"),
        "hospitality":                _find(raw / "PUBLIC",      f"{s}_hospitality"),
        "big_buildings_and_clusters": _find(raw / "BUILDINGS",   f"{s}_fema_building_clusters"),
        "railway":                    _find(raw / "TRANSPORT",   f"{s}_rail_network"),
        "substations":                _find(raw / "SUBSTATIONS", f"{s}_substations_final"),
        "dams":                       _find(raw / "TRANSPORT",   f"{s}_dams_major"),
        "transmission_lines":         _find(raw / "GRID",        f"{s}_transmission_lines"),
        "osm_distribution_lines":     _find(raw / "GRID",        f"{s}_osm_distribution_lines"),
        # Land vectors
        "wetlands":        _find(raw / "LAND", f"{s}_wetlands"),
        "vrm":             _find(raw / "LAND", f"{s}_vrm"),
        "dso_boundaries":  _find(raw / "LAND", f"{s}_dso_boundaries"),
        # Rasters
        "cdl":   raw / "LAND" / f"{s}_cdl.tif",
        "nlcd":  raw / "LAND" / f"{s}_nlcd.tif",
        "dem":   raw / "LAND" / f"{s}_dem.tif",
        "lanid": raw / "LAND" / f"{s}_lanid.tif",
    }


# ==========================================
#                HELPERS
# ==========================================


def _dynamic_buffer(pts: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Building-density-weighted buffer: rural (550 m) → urban (120 m)."""
    centroids  = buildings.geometry.centroid
    tree       = cKDTree(np.c_[centroids.x, centroids.y])
    counts     = tree.query_ball_point(
        np.c_[pts.geometry.x, pts.geometry.y],
        r=DENSITY_RADIUS_M, return_length=True,
    )
    return np.interp(counts, [0, DENSITY_THRESHOLD_HIGH], [BUFFER_RURAL_M, BUFFER_URBAN_M])



def _grid_points(zone_geom, res: int) -> gpd.GeoDataFrame:
    """Regular grid of points within zone_geom at spacing `res` metres."""
    if zone_geom is None or zone_geom.is_empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)
    minx, miny, maxx, maxy = zone_geom.bounds
    xs, ys = np.arange(minx, maxx, res), np.arange(miny, maxy, res)
    pts    = [Point(x, y) for x in xs for y in ys]
    gdf    = gpd.GeoDataFrame(geometry=pts, crs=METRIC_CRS)
    return gdf[gdf.within(zone_geom)].reset_index(drop=True)


# ==========================================
#  PHASE 1: TRAINING POINT GENERATION
# ==========================================

def _sample_near_lines(lines: gpd.GeoDataFrame, n: int, max_dist: float) -> gpd.GeoDataFrame:
    """Uniform random cloud within max_dist of line network (weighted by length)."""
    lines = lines.copy()
    lines["_len"] = lines.geometry.length
    lines["_w"]   = lines["_len"] / lines["_len"].sum()
    chosen = lines.sample(n=n, weights="_w", replace=True)
    pts = []
    for geom in chosen.geometry:
        pt    = geom.interpolate(np.random.uniform(0, geom.length))
        dist  = np.random.uniform(0, max_dist)
        theta = np.random.uniform(0, 2 * np.pi)
        pts.append(Point(pt.x + dist * np.cos(theta), pt.y + dist * np.sin(theta)))
    return gpd.GeoDataFrame(geometry=pts, crs=lines.crs)


def _sample_random_in_polygon(poly_union, n: int) -> gpd.GeoDataFrame:
    """Rejection-sample n uniform random points inside poly_union."""
    minx, miny, maxx, maxy = poly_union.bounds
    pts = []
    while len(pts) < n:
        batch = (n - len(pts)) * 3
        xs = np.random.uniform(minx, maxx, batch)
        ys = np.random.uniform(miny, maxy, batch)
        for x, y in zip(xs, ys):
            if poly_union.contains(Point(x, y)):
                pts.append(Point(x, y))
                if len(pts) >= n:
                    break
    return gpd.GeoDataFrame(geometry=pts, crs=METRIC_CRS)


def generate_training_points(paths: dict) -> gpd.GeoDataFrame:
    """
    Sample balanced (1:1) labeled training points within the training range.

    Returns GeoDataFrame with columns:
      geometry, source, dynamic_buffer_size, bldg_count_1km,
      dist_to_grid, dynamic_threshold, grid (0/1)
    """
    print("\n[Phase 1] Generating training points...")

    grid_lines = load_geo(paths["grid_lines"], METRIC_CRS)
    if grid_lines is None or grid_lines.empty:
        raise FileNotFoundError(f"Grid lines not found: {paths['grid_lines']}")

    training_range = load_geo(paths["training_range"], METRIC_CRS)
    if training_range is None or training_range.empty:
        raise FileNotFoundError(f"Training range not found: {paths['training_range']}")

    roads      = load_geo(paths["roads"],     METRIC_CRS)
    buildings  = load_geo(paths["buildings"], METRIC_CRS)
    exclusions = load_geo(paths["exclusions"],METRIC_CRS)

    range_union = training_range.geometry.unary_union
    if exclusions is not None and not exclusions.empty:
        excl_geom = exclusions.geometry.make_valid()
        range_union = range_union.difference(excl_geom.unary_union)
    valid_gdf = gpd.GeoDataFrame(geometry=[range_union], crs=METRIC_CRS)

    area_km2 = range_union.area / 1e6
    scale    = area_km2 / 1_000
    n_grid   = max(100, int(N_GRID_SAMPLES_PER_1K_KM2 * scale * TRAINING_POINT_MULTIPLIER))
    n_road   = max(100, int(N_ROAD_SAMPLES_PER_1K_KM2 * scale * TRAINING_POINT_MULTIPLIER))
    n_bg     = max(100, int(N_BG_SAMPLES_PER_1K_KM2   * scale * TRAINING_POINT_MULTIPLIER))
    print(f"   training area: {area_km2:,.0f} km²  →  grid={n_grid:,}  road={n_road:,}  bg={n_bg:,}")

    # Sample near grid lines, near roads, background
    s_grid        = _sample_near_lines(grid_lines, n_grid, BUFFER_RURAL_M)
    s_grid["source"] = "near_grid"

    if roads is not None and not roads.empty:
        s_roads = _sample_near_lines(roads, n_road, BUFFER_RURAL_M)
        s_roads["source"] = "near_road"
    else:
        s_roads = gpd.GeoDataFrame(columns=["geometry", "source"], crs=METRIC_CRS)

    s_bg = _sample_random_in_polygon(range_union, n_bg)
    s_bg["source"] = "background"

    all_pts = gpd.GeoDataFrame(
        pd.concat([s_grid, s_roads, s_bg], ignore_index=True),
        geometry="geometry", crs=METRIC_CRS,
    )

    # Clip to valid area
    all_pts = gpd.sjoin(all_pts, valid_gdf, how="inner", predicate="within")
    all_pts = all_pts[["geometry", "source"]].reset_index(drop=True)
    print(f"   {len(all_pts):,} points after clipping to valid area")

    # Dynamic buffer from building density
    if buildings is not None and not buildings.empty:
        buf   = _dynamic_buffer(all_pts, buildings)
        tree  = cKDTree(np.c_[buildings.geometry.centroid.x, buildings.geometry.centroid.y])
        bldg_counts = tree.query_ball_point(
            np.c_[all_pts.geometry.x, all_pts.geometry.y],
            r=DENSITY_RADIUS_M, return_length=True,
        )
    else:
        buf         = np.full(len(all_pts), float(BUFFER_RURAL_M))
        bldg_counts = np.zeros(len(all_pts), dtype=int)

    all_pts["dynamic_buffer_size"] = buf
    all_pts["bldg_count_1km"]      = bldg_counts

    # Distance to nearest grid line
    print("   computing dist_to_grid...")
    g_sindex   = grid_lines.sindex
    search_buf = BUFFER_RURAL_M * 1.5
    dists = []
    for pt in all_pts.geometry:
        cands = list(g_sindex.query(pt.buffer(search_buf), predicate="intersects"))
        dists.append(float(grid_lines.iloc[cands].distance(pt).min()) if cands else 9_999.0)
    all_pts["dist_to_grid"] = dists

    # Dynamic threshold + label
    all_pts["dynamic_threshold"] = np.interp(
        all_pts["bldg_count_1km"],
        [0, DENSITY_THRESHOLD_HIGH],
        [BUFFER_RURAL_M, BUFFER_URBAN_M],
    )
    all_pts["grid"] = (all_pts["dist_to_grid"] <= all_pts["dynamic_threshold"]).astype(int)

    # Balance 1:1
    pos   = all_pts[all_pts["grid"] == 1]
    neg   = all_pts[all_pts["grid"] == 0]
    n_each = min(len(pos), len(neg))
    all_pts = pd.concat([
        pos.sample(n=n_each, random_state=42),
        neg.sample(n=n_each, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"   balanced: {len(all_pts):,} points ({n_each} pos + {n_each} neg)")

    return gpd.GeoDataFrame(all_pts, geometry="geometry", crs=METRIC_CRS)


# ==========================================
#  PHASE 2 / 4: FEATURE ASSIGNMENT (shared)
# ==========================================

def _nearest_kdtree(pts: gpd.GeoDataFrame, target: gpd.GeoDataFrame,
                    fallback: float = 99_999.0) -> np.ndarray:
    """KDTree distance from each point to nearest target centroid."""
    if target is None or target.empty:
        return np.full(len(pts), fallback)
    cens  = target.geometry.centroid
    tree  = cKDTree(np.c_[cens.x, cens.y])
    dists, _ = tree.query(np.c_[pts.geometry.x, pts.geometry.y], k=1)
    return dists


def _count_in_dynamic_buf(pts: gpd.GeoDataFrame, target: gpd.GeoDataFrame) -> np.ndarray:
    if target is None or target.empty:
        return np.zeros(len(pts), dtype=int)
    buf_gdf           = pts[["dynamic_buffer_size", "geometry"]].copy()
    buf_gdf["geometry"] = pts.geometry.buffer(pts["dynamic_buffer_size"])
    joined  = gpd.sjoin(buf_gdf[["geometry"]], target[["geometry"]], how="left", predicate="intersects")
    counts  = joined.groupby(joined.index)["index_right"].count()
    return counts.reindex(pts.index, fill_value=0).values


def _area_in_dynamic_buf(pts: gpd.GeoDataFrame, target: gpd.GeoDataFrame) -> np.ndarray:
    if target is None or target.empty:
        return np.zeros(len(pts))
    sindex = target.sindex
    result = []
    for pt, radius in zip(pts.geometry, pts["dynamic_buffer_size"]):
        buf   = pt.buffer(radius)
        cands = list(sindex.query(buf, predicate="intersects"))
        result.append(
            float(target.iloc[cands].geometry.intersection(buf).area.sum()) if cands else 0.0
        )
    return np.array(result)


def _fraction_in_dynamic_buf(pts: gpd.GeoDataFrame, layer: gpd.GeoDataFrame | None) -> np.ndarray:
    """Fraction (0–1) of each point's dynamic-radius buffer covered by layer."""
    if layer is None or layer.empty:
        return np.zeros(len(pts))
    sindex = layer.sindex
    result = []
    for pt, radius in zip(pts.geometry, pts["dynamic_buffer_size"]):
        buf   = pt.buffer(radius)
        cands = list(sindex.query(buf, predicate="intersects"))
        result.append(
            float(layer.iloc[cands].geometry.intersection(buf).area.sum() / buf.area) if cands else 0.0
        )
    return np.array(result)


def _road_length_by_type(pts: gpd.GeoDataFrame, roads: gpd.GeoDataFrame | None) -> pd.DataFrame:
    if roads is None or roads.empty:
        return pd.DataFrame(index=pts.index)
    buf_gdf = pts[["dynamic_buffer_size", "geometry"]].copy()
    buf_gdf["buffer_geom"] = pts.geometry.buffer(pts["dynamic_buffer_size"])
    buf_gdf = buf_gdf.set_geometry("buffer_geom")
    type_col = "new_class" if "new_class" in roads.columns else "highway"
    join_right = gpd.GeoDataFrame(geometry=buf_gdf["buffer_geom"], crs=buf_gdf.crs)
    joined  = gpd.sjoin(roads[[type_col, "geometry"]], join_right, how="inner", predicate="intersects")
    joined  = joined.join(buf_gdf["buffer_geom"], on="index_right", rsuffix="_buf")
    joined["seg_len"] = joined.geometry.intersection(joined["buffer_geom"]).length
    pivoted = joined.pivot_table(
        index="index_right", columns=type_col, values="seg_len", aggfunc="sum", fill_value=0,
    )
    pivoted.columns = [f"road_len_{c}" for c in pivoted.columns]
    return pivoted


def _sample_nlcd(pts: gpd.GeoDataFrame, nlcd_path: Path) -> dict[str, list]:
    keys  = ["nlcd_dev_pct", "nlcd_nature_pct", "nlcd_crops_pct",
             "nlcd_nature_pct_x10", "nlcd_crops_pct_x10"]
    empty = {k: [0.0] * len(pts) for k in keys}
    if not nlcd_path.exists():
        return empty

    with rasterio.open(nlcd_path) as src:
        data = src.read(1)
        H, W = data.shape
        nd   = src.nodata
        pts_proj = pts.to_crs(src.crs)
        rows_a, cols_a = rowcol(
            src.transform,
            [p.x for p in pts_proj.geometry],
            [p.y for p in pts_proj.geometry],
        )
        px       = src.res[0]
        rads_1x  = np.maximum((pts["dynamic_buffer_size"].values / px).astype(int), 1)
        rads_10x = np.maximum((pts["dynamic_buffer_size"].values * 10 / px).astype(int), 1)

        def sample_win(r, c, rad):
            if r < 0 or c < 0 or r >= H or c >= W:
                return 0.0, 0.0, 0.0
            r0, r1 = max(0, r - rad), min(H, r + rad + 1)
            c0, c1 = max(0, c - rad), min(W, c + rad + 1)
            win = data[r0:r1, c0:c1]
            oy, ox = np.ogrid[r0 - r:r1 - r, c0 - c:c1 - c]
            mask = (ox * ox + oy * oy) <= rad * rad
            vals = win[mask]
            if nd is not None:
                vals = vals[vals != nd]
            if vals.size == 0:
                return 0.0, 0.0, 0.0
            tot = vals.size
            return (
                float(np.isin(vals, NLCD_DEVELOPED).sum() / tot * 100),
                float(np.isin(vals, NLCD_NATURE).sum()    / tot * 100),
                float(np.isin(vals, NLCD_CROPS).sum()     / tot * 100),
            )

        res = {k: [] for k in keys}
        for r, c, r1, r10 in zip(rows_a, cols_a, rads_1x, rads_10x):
            d1, n1, c1   = sample_win(r, c, r1)
            _,  n10, c10 = sample_win(r, c, r10)
            res["nlcd_dev_pct"].append(d1)
            res["nlcd_nature_pct"].append(n1)
            res["nlcd_crops_pct"].append(c1)
            res["nlcd_nature_pct_x10"].append(n10)
            res["nlcd_crops_pct_x10"].append(c10)
    return res


def _sample_terrain_tpi(pts: gpd.GeoDataFrame, dem_path: Path) -> dict[str, list]:
    empty = {"terrain_elev_diff": [0.0] * len(pts), "terrain_elev_diff_x10": [0.0] * len(pts)}
    if not dem_path.exists():
        return empty

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        dem[dem < -1000] = np.nan
        dv  = np.where(np.isnan(dem), 0.0, dem)
        dm  = np.where(np.isnan(dem), 0,   1)
        iis = np.pad(dv.cumsum(0).cumsum(1), ((1, 0), (1, 0)), "constant")
        iic = np.pad(dm.cumsum(0).cumsum(1), ((1, 0), (1, 0)), "constant")

        pts_proj = pts.to_crs(src.crs)
        rows_a, cols_a = rowcol(
            src.transform,
            [p.x for p in pts_proj.geometry],
            [p.y for p in pts_proj.geometry],
        )
        rows_a  = np.array(rows_a);  cols_a = np.array(cols_a)
        px      = src.res[0]
        rads_1x  = np.maximum((pts["dynamic_buffer_size"].values / px).astype(int), 1)
        rads_10x = np.maximum((pts["dynamic_buffer_size"].values * 10 / px).astype(int), 1)

        def box_mean(rc, cc, rr):
            r0 = np.clip(rc - rr, 0, dem.shape[0])
            r1 = np.clip(rc + rr + 1, 0, dem.shape[0])
            c0 = np.clip(cc - rr, 0, dem.shape[1])
            c1 = np.clip(cc + rr + 1, 0, dem.shape[1])
            s  = iis[r1, c1] - iis[r0, c1] - iis[r1, c0] + iis[r0, c0]
            n  = iic[r1, c1] - iic[r0, c1] - iic[r1, c0] + iic[r0, c0]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(n > 0, s / n, np.nan)

        mean_1x  = box_mean(rows_a, cols_a, rads_1x)
        mean_10x = box_mean(rows_a, cols_a, rads_10x)

        valid   = (rows_a >= 0) & (rows_a < dem.shape[0]) & (cols_a >= 0) & (cols_a < dem.shape[1])
        pt_elev = np.full(len(pts), np.nan)
        pt_elev[valid] = dem[rows_a[valid], cols_a[valid]]

        return {
            "terrain_elev_diff":     np.nan_to_num(pt_elev - mean_1x,  nan=0.0).tolist(),
            "terrain_elev_diff_x10": np.nan_to_num(pt_elev - mean_10x, nan=0.0).tolist(),
        }


def _sample_cdl_3phase(pts: gpd.GeoDataFrame, cdl_path: Path) -> list[float]:
    if not cdl_path.exists():
        return [0.0] * len(pts)
    with rasterio.open(cdl_path) as src:
        data = src.read(1)
        H, W = data.shape
        pts_proj = pts.to_crs(src.crs)
        rows_a, cols_a = rowcol(
            src.transform,
            [p.x for p in pts_proj.geometry],
            [p.y for p in pts_proj.geometry],
        )
        px = src.res[0]
        fracs = []
        for r, c, radius in zip(rows_a, cols_a, pts["dynamic_buffer_size"].values):
            rad = max(int(radius / px), 1)
            if r < 0 or c < 0 or r >= H or c >= W:
                fracs.append(0.0); continue
            r0, r1 = max(0, r - rad), min(H, r + rad + 1)
            c0, c1 = max(0, c - rad), min(W, c + rad + 1)
            win = data[r0:r1, c0:c1]
            oy, ox = np.ogrid[r0 - r:r1 - r, c0 - c:c1 - c]
            mask = (ox * ox + oy * oy) <= rad * rad
            vals = win[mask]
            fracs.append(float(np.isin(vals, CDL_3PHASE_CODES).sum() / vals.size) if vals.size else 0.0)
    return fracs


def _sample_lanid_dist(pts: gpd.GeoDataFrame, lanid_path: Path) -> list[float]:
    if not lanid_path.exists():
        return [99_999.0] * len(pts)
    with rasterio.open(lanid_path) as src:
        data  = src.read(1)
        mask  = (data <= 0).astype(np.uint8)
        dist_m = distance_transform_edt(mask) * src.res[0]
        pts_proj = pts.to_crs(src.crs)
        rows_a, cols_a = rowcol(
            src.transform,
            [p.x for p in pts_proj.geometry],
            [p.y for p in pts_proj.geometry],
        )
        H, W  = dist_m.shape
        dists = []
        for r, c in zip(rows_a, cols_a):
            dists.append(float(dist_m[r, c]) if 0 <= r < H and 0 <= c < W else 99_999.0)
    return dists


def _dist_to_polygon_boundaries(pts: gpd.GeoDataFrame, polys: gpd.GeoDataFrame,
                                 fallback: float = 99_999.0) -> np.ndarray:
    """Distance from each point to the nearest polygon boundary edge (works for interior points too)."""
    if polys is None or polys.empty:
        return np.full(len(pts), fallback)
    bounds = polys.geometry.boundary.explode(index_parts=False)
    bounds = bounds[~bounds.is_empty].reset_index(drop=True)
    if bounds.empty:
        return np.full(len(pts), fallback)
    # DSO has ~50 polygons, so iterating over pts and broadcasting distance is fast enough
    return np.array([float(bounds.distance(pt).min()) for pt in pts.geometry])


def _dist_to_lines(pts: gpd.GeoDataFrame, lines: gpd.GeoDataFrame,
                   fallback: float = 99_999.0) -> np.ndarray:
    """Distance from each point to the nearest line geometry (uses sindex for performance)."""
    if lines is None or lines.empty:
        return np.full(len(pts), fallback)
    sindex = lines.sindex
    result = []
    for pt in pts.geometry:
        # progressive buffer: try 5 km first, fall back to full dataset
        cands = list(sindex.query(pt.buffer(5_000), predicate="intersects"))
        if not cands:
            cands = list(range(len(lines)))
        result.append(float(lines.iloc[cands].distance(pt).min()))
    return np.array(result)


def assign_features(pts: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """
    Compute all ML features for a set of points.
    Requires 'dynamic_buffer_size' column.
    """
    pts = pts.copy()
    print(f"  assigning features to {len(pts):,} points...")

    # ---- Infrastructure ----
    print("\n  [infrastructure distances / counts / areas]")
    pts["count_infra_composite"] = 0
    for key, actions in INFRA_TASKS.items():
        data = load_geo(paths.get(key), METRIC_CRS)
        if "dist" in actions:
            print(f"   dist → {key}")
            pts[f"dist_to_{key}"] = _nearest_kdtree(pts, data)
        if "count" in actions:
            counts = _count_in_dynamic_buf(pts, data)
            if key in COMPOSITE_LAYERS:
                pts["count_infra_composite"] += counts
            else:
                pts[f"count_{key}"] = counts
        if "area" in actions:
            print(f"   area → {key}")
            pts[f"area_{key}"] = _area_in_dynamic_buf(pts, data)

    # ---- Building stats ----
    print("\n  [building stats]")
    bldgs = load_geo(paths.get("buildings"), METRIC_CRS)
    if bldgs is not None and not bldgs.empty:
        if "Polygon" in bldgs.geometry.iloc[0].geom_type:
            bldgs["_area"] = bldgs.geometry.area
        elif "area_m2" in bldgs.columns:
            bldgs["_area"] = pd.to_numeric(bldgs["area_m2"], errors="coerce").fillna(0.0)
        else:
            bldgs["_area"] = 0.0

        sindex = bldgs.sindex
        medians, stds, densities = [], [], []
        for pt, radius in zip(pts.geometry, pts["dynamic_buffer_size"]):
            buf   = pt.buffer(radius)
            cands = list(sindex.query(buf, predicate="intersects"))
            if not cands:
                medians.append(0.0); stds.append(0.0); densities.append(0.0); continue
            matches = bldgs.iloc[cands]
            matches = matches[matches.intersects(buf)]
            if matches.empty:
                medians.append(0.0); stds.append(0.0); densities.append(0.0)
            else:
                medians.append(float(matches["_area"].median()))
                stds.append(float(matches["_area"].std()) if len(matches) > 1 else 0.0)
                densities.append(float(matches["_area"].sum() / (buf.area / 1e6)))
        pts["bldg_median_m2"]    = medians
        pts["bldg_std_m2"]       = stds
        pts["bldg_area_per_km2"] = densities
    else:
        pts["bldg_median_m2"] = pts["bldg_std_m2"] = pts["bldg_area_per_km2"] = 0.0


    # ---- NLCD (dual radius) ----
    print("\n  [NLCD dual radius]")
    for k, v in _sample_nlcd(pts, paths["nlcd"]).items():
        pts[k] = v

    # ---- Terrain TPI (dual radius) ----
    print("\n  [terrain TPI dual radius]")
    for k, v in _sample_terrain_tpi(pts, paths["dem"]).items():
        pts[k] = v

    # ---- CDL 3-phase fraction ----
    print("\n  [CDL 3-phase]")
    pts["cdl_3phase_frac"] = _sample_cdl_3phase(pts, paths["cdl"])

    # ---- LANID distance ----
    print("\n  [LANID distance]")
    pts["dist_to_lanid"] = _sample_lanid_dist(pts, paths["lanid"])

    # ---- Wetlands / VRM fraction ----
    print("\n  [wetlands / VRM]")
    pts["wetlands_pct"] = _fraction_in_dynamic_buf(pts, load_geo(paths.get("wetlands"), METRIC_CRS))
    pts["vrm_pct"]      = _fraction_in_dynamic_buf(pts, load_geo(paths.get("vrm"),      METRIC_CRS))

    # ---- DSO boundary distance ----
    print("\n  [DSO boundary distance]")
    pts["dist_to_dso_boundary"] = _dist_to_polygon_boundaries(
        pts, load_geo(paths.get("dso_boundaries"), METRIC_CRS)
    )

    # ---- OSM distribution lines distance ----
    print("\n  [OSM distribution lines distance]")
    pts["dist_to_osm_distribution_lines"] = _dist_to_lines(
        pts, load_geo(paths.get("osm_distribution_lines"), METRIC_CRS)
    )

    return pts


# ==========================================
#  PHASE 3: PREDICTION CANDIDATE GENERATION
# ==========================================

def generate_prediction_candidates(paths: dict) -> gpd.GeoDataFrame:
    """
    High-res grid near roads/buildings + low-res grid elsewhere,
    clipped to prediction_extent (project-specific) or training_range as fallback,
    minus exclusions.
    """
    print("\n[Phase 3] Generating prediction candidates...")

    # Use project prediction_extent if provided, otherwise fall back to training_range
    pred_extent = load_geo(paths.get("prediction_extent"), METRIC_CRS)
    if pred_extent is not None and not pred_extent.empty:
        area_gdf = pred_extent
    else:
        area_gdf = load_geo(paths["training_range"], METRIC_CRS)
        if area_gdf is None or area_gdf.empty:
            raise FileNotFoundError(f"Training range not found: {paths['training_range']}")

    roads      = load_geo(paths["roads"],      METRIC_CRS)
    buildings  = load_geo(paths["buildings"],  METRIC_CRS)
    exclusions = load_geo(paths["exclusions"], METRIC_CRS)

    valid_area = area_gdf.geometry.unary_union
    if EXCLUDE_GRID_FROM_PREDICTION:
        grid_range = load_geo(paths.get("grid_range_parquet"), METRIC_CRS)
        if grid_range is not None and not grid_range.empty:
            valid_area = valid_area.difference(grid_range.geometry.unary_union)
    if exclusions is not None and not exclusions.empty:
        valid_area = valid_area.difference(exclusions.geometry.make_valid().unary_union)

    road_buf = roads.geometry.buffer(BUF_ROAD_M).unary_union \
        if roads is not None and not roads.empty else None
    bldg_buf = buildings.geometry.buffer(BUF_BLDG_M).unary_union \
        if buildings is not None and not buildings.empty else None

    if road_buf is not None and bldg_buf is not None:
        high_zone = road_buf.union(bldg_buf)
    elif road_buf is not None:
        high_zone = road_buf
    elif bldg_buf is not None:
        high_zone = bldg_buf
    else:
        high_zone = valid_area

    high_zone = high_zone.intersection(valid_area)
    low_zone  = valid_area.difference(high_zone)

    high_pts = _grid_points(high_zone, RES_HIGH_M)
    high_pts["type"] = "high_res"
    low_pts  = _grid_points(low_zone,  RES_LOW_M)
    low_pts["type"] = "low_res"
    print(f"   high-res: {len(high_pts):,}   low-res: {len(low_pts):,}")

    cands = gpd.GeoDataFrame(
        pd.concat([high_pts, low_pts], ignore_index=True),
        geometry="geometry", crs=METRIC_CRS,
    )

    cands["dynamic_buffer_size"] = (
        _dynamic_buffer(cands, buildings)
        if buildings is not None and not buildings.empty
        else float(BUFFER_RURAL_M)
    )
    print(f"   total: {len(cands):,} prediction candidates")
    return cands


# ==========================================
#  PHASE 5: TRAIN XGBOOST + PREDICT
# ==========================================

def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric cleaning + interaction features shared by train + predict."""
    X = df.drop(columns=[c for c in df.columns if c in EXCLUDE_COLS], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    dist_cols = [c for c in X.columns if "dist_" in c]
    X[dist_cols] = X[dist_cols].fillna(99_999.0)
    X = X.fillna(0.0)

    buf = df["dynamic_buffer_size"].replace(0, 500) if "dynamic_buffer_size" in df.columns \
        else pd.Series(500.0, index=df.index)

    if "count_infra_composite" in X.columns:
        X["infra_density"] = X["count_infra_composite"] / buf.values

    road_cols = [c for c in X.columns if c.startswith("road_len_")]
    if road_cols:
        X["road_len_total"] = X[road_cols].sum(axis=1)
        X["road_density"]   = X["road_len_total"] / buf.values

    for col in ["dist_to_substations", "dist_to_utilities", "dist_to_industry"]:
        if col in X.columns:
            X[f"log_{col}"] = np.log1p(X[col])

    return X


def _best_threshold(y_true, y_probs) -> tuple[float, float]:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.02):
        f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def train_and_predict(
    train_pts: gpd.GeoDataFrame,
    pred_pts:  gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Train one XGBoost model on all training points (20% random validation split
    for early stopping), then apply it to all prediction candidates.
    """
    print("\n[Phase 5] Training XGBoost + predicting...")

    from sklearn.model_selection import train_test_split

    X = _feature_engineering(train_pts)
    y = train_pts["grid"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos   = y_train.sum(); neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        scale_pos_weight=scale,
        n_estimators=5000,
        early_stopping_rounds=50,
        **XGB_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    val_probs       = model.predict_proba(X_valid)[:, 1]
    best_t, best_f1 = _best_threshold(y_valid, val_probs)
    val_preds       = (val_probs >= best_t).astype(int)
    acc = accuracy_score(y_valid, val_preds)
    auc = roc_auc_score(y_valid, val_probs)
    print(f"  Val — Acc={acc:.3f}  AUC={auc:.3f}  F1={best_f1:.3f}  thresh={best_t:.2f}")
    print(f"  Trees used: {model.best_iteration + 1}")

    X_pred = _feature_engineering(pred_pts)
    for c in set(X_train.columns) - set(X_pred.columns):
        X_pred[c] = 0.0
    X_pred = X_pred[X_train.columns]

    pred_pts = pred_pts.copy()
    pred_pts["grid_prob"] = model.predict_proba(X_pred)[:, 1]
    return pred_pts.reset_index(drop=True)


# ==========================================
#  PHASE 6 / 7: CLUSTER + ANTI-CLUSTER POLYGONS
# ==========================================

def _classify_points(pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pts = pts.copy()
    infra   = pts["count_infra_composite"] if "count_infra_composite" in pts.columns \
              else pd.Series(0, index=pts.index)
    d_sub   = pts["dist_to_substations"]   if "dist_to_substations"   in pts.columns \
              else pd.Series(99_999.0, index=pts.index)
    buf     = pts["dynamic_buffer_size"]   if "dynamic_buffer_size"   in pts.columns \
              else pd.Series(500.0, index=pts.index)
    a_solar = pts["area_solar"]            if "area_solar"            in pts.columns \
              else pd.Series(0.0, index=pts.index)
    pts["point_class"] = np.select(
        [infra >= 1, d_sub < buf, a_solar > 0],
        ["industrial", "substation", "solar"],
        default="residential",
    )
    return pts


def _hull(multipoint: MultiPoint, small_buf: float, fallback_buf: float):
    try:
        h = multipoint.concave_hull(ratio=HULL_RATIO, allow_holes=False)
        if h.geom_type in ("LineString", "MultiLineString", "Point", "MultiPoint"):
            h = h.buffer(small_buf)
        return h
    except AttributeError:
        return multipoint.buffer(fallback_buf).buffer(-fallback_buf * 0.8)


def generate_cluster_polygons(pred_pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """HDBSCAN on high-probability points → concave hull cluster polygons."""
    print(f"\n[Phase 6] Cluster polygons (prob >= {PROB_THRESHOLD})...")
    valid = pred_pts[pred_pts["grid_prob"] >= PROB_THRESHOLD].copy()
    print(f"  {len(valid):,} points above threshold")
    if valid.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)

    valid = _classify_points(valid)
    coords = np.c_[valid.geometry.x, valid.geometry.y]
    valid["cluster_id"] = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES_CLUSTER,
        metric="euclidean",
    ).fit_predict(coords)

    clustered = valid[valid["cluster_id"] != -1]
    print(f"  {clustered['cluster_id'].nunique()} clusters  "
          f"({(valid['cluster_id'] == -1).sum()} noise points discarded)")

    rows = []
    for cid, grp in clustered.groupby("cluster_id"):
        cc = grp["point_class"].value_counts()
        rows.append({
            "cluster_id":    cid,
            "dominant_class": grp["point_class"].mode()[0],
            "point_count":   len(grp),
            "mean_prob":     round(float(grp["grid_prob"].mean()), 3),
            "n_industrial":  int(cc.get("industrial", 0)),
            "n_substation":  int(cc.get("substation",  0)),
            "n_solar":       int(cc.get("solar",       0)),
            "n_residential": int(cc.get("residential", 0)),
            "geometry":      _hull(MultiPoint(grp.geometry.tolist()),
                                   CLUSTER_FALLBACK_BUF, CLUSTER_HULL_FALLBACK),
        })
    return gpd.GeoDataFrame(rows, crs=METRIC_CRS)


def generate_anti_cluster_polygons(pred_pts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """HDBSCAN on low-probability points → shrunk concave hull anti-cluster polygons."""
    print(f"\n[Phase 7] Anti-cluster polygons (prob <= {MAX_PROB_ANTI})...")
    valid = pred_pts[pred_pts["grid_prob"] <= MAX_PROB_ANTI].copy()
    print(f"  {len(valid):,} points below threshold")
    if valid.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)

    valid = _classify_points(valid)
    coords = np.c_[valid.geometry.x, valid.geometry.y]
    valid["cluster_id"] = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE_ANTI, min_samples=MIN_SAMPLES_ANTI, metric="euclidean"
    ).fit_predict(coords)

    clustered = valid[valid["cluster_id"] != -1]
    print(f"  {clustered['cluster_id'].nunique()} anti-clusters  "
          f"({(valid['cluster_id'] == -1).sum()} noise points discarded)")

    rows = []
    for cid, grp in clustered.groupby("cluster_id"):
        cc   = grp["point_class"].value_counts()
        base = _hull(MultiPoint(grp.geometry.tolist()), ANTI_FALLBACK_BUF, 1_000)
        shrunk = base.buffer(-ANTI_SHRINK_M)
        if shrunk.is_empty:
            shrunk = base
        rows.append({
            "anti_cluster_id": cid,
            "dominant_class":  grp["point_class"].mode()[0],
            "point_count":     len(grp),
            "mean_prob":       round(float(grp["grid_prob"].mean()), 4),
            "n_industrial":    int(cc.get("industrial", 0)),
            "n_substation":    int(cc.get("substation",  0)),
            "n_solar":         int(cc.get("solar",       0)),
            "n_residential_wilderness": int(cc.get("residential", 0)),
            "geometry":        shrunk,
        })
    return gpd.GeoDataFrame(rows, crs=METRIC_CRS)


# ==========================================
#  PHASE 8: MINOR CLUSTER POLYGONS
# ==========================================

def generate_minor_cluster_polygons(
    pred_pts:      gpd.GeoDataFrame,
    major_clusters: gpd.GeoDataFrame,
    paths:         dict,
) -> gpd.GeoDataFrame:
    """
    1. HDBSCAN on points with prob >= MINOR_PROB_THRESHOLD.
    2. Drop any minor polygon that overlaps a major cluster polygon.
    3. Add one artificial polygon per solar plant not covered by any
       major or minor HDBSCAN cluster.
    4. Add one artificial polygon per frs_primary feature not covered
       by any major or minor HDBSCAN cluster.

    Returns a single GeoDataFrame with a 'source' column:
      'model'    — HDBSCAN-derived minor clusters
      'solar'    — artificial solar-plant cluster
      'frs'      — artificial frs_primary cluster
    """
    print(f"\n[Phase 8] Minor cluster polygons (prob >= {MINOR_PROB_THRESHOLD})...")

    # ---- Step 1: HDBSCAN minor clusters ----
    valid = pred_pts[pred_pts["grid_prob"] >= MINOR_PROB_THRESHOLD].copy()
    print(f"  {len(valid):,} points above minor threshold")

    minor_model: gpd.GeoDataFrame
    if valid.empty:
        minor_model = gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)
    else:
        valid = _classify_points(valid)
        coords = np.c_[valid.geometry.x, valid.geometry.y]
        valid["cluster_id"] = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES_CLUSTER,
            metric="euclidean",
        ).fit_predict(coords)

        clustered = valid[valid["cluster_id"] != -1]
        print(f"  {clustered['cluster_id'].nunique()} HDBSCAN minor clusters  "
              f"({(valid['cluster_id'] == -1).sum()} noise discarded)")

        rows = []
        for cid, grp in clustered.groupby("cluster_id"):
            cc = grp["point_class"].value_counts()
            rows.append({
                "cluster_id":     int(cid),
                "dominant_class": grp["point_class"].mode()[0],
                "point_count":    len(grp),
                "mean_prob":      round(float(grp["grid_prob"].mean()), 3),
                "n_industrial":   int(cc.get("industrial", 0)),
                "n_substation":   int(cc.get("substation",  0)),
                "n_solar":        int(cc.get("solar",       0)),
                "n_residential":  int(cc.get("residential", 0)),
                "source":         "model",
                "geometry":       _hull(MultiPoint(grp.geometry.tolist()),
                                        CLUSTER_FALLBACK_BUF, CLUSTER_HULL_FALLBACK),
            })
        minor_model = gpd.GeoDataFrame(rows, crs=METRIC_CRS) if rows \
            else gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)

    # ---- Step 2: Remove minor clusters that overlap major clusters ----
    if not major_clusters.empty and not minor_model.empty:
        major_union = major_clusters.geometry.unary_union
        keep        = ~minor_model.geometry.intersects(major_union)
        minor_model = minor_model[keep].reset_index(drop=True)
        print(f"  {len(minor_model)} minor model clusters after removing major overlaps")

    # Union of all HDBSCAN clusters (major + minor) — used to check artificial additions
    parts = []
    if not major_clusters.empty:
        parts.append(major_clusters.geometry.unary_union)
    if not minor_model.empty:
        parts.append(minor_model.geometry.unary_union)
    hdbscan_union = parts[0].union(parts[1]) if len(parts) == 2 \
        else parts[0] if parts else None

    all_rows: list[dict] = []
    next_id = 10_000   # artificial cluster IDs start here

    pred_extent = paths.get("prediction_extent")
    pred_extent_union = pred_extent.geometry.unary_union \
        if pred_extent is not None and not pred_extent.empty else None

    # ---- Step 3: Artificial clusters — solar plants ----
    solar = load_geo(paths.get("solar"), METRIC_CRS)
    if solar is not None and not solar.empty:
        added = 0
        for _, row in solar.iterrows():
            geom = row.geometry
            poly = geom if geom.geom_type in ("Polygon", "MultiPolygon") \
                else geom.buffer(ARTIFICIAL_BUF_M)
            poly = poly.buffer(ARTIFICIAL_BUF_M)   # small margin around the plant
            if pred_extent_union is not None and not poly.intersects(pred_extent_union):
                continue
            if hdbscan_union is not None and poly.intersects(hdbscan_union):
                continue
            all_rows.append({
                "cluster_id":     next_id,
                "dominant_class": "solar",
                "point_count":    0,
                "mean_prob":      None,
                "n_industrial":   0,
                "n_substation":   0,
                "n_solar":        1,
                "n_residential":  0,
                "source":         "solar",
                "geometry":       poly,
            })
            next_id += 1; added += 1
        print(f"  added {added} artificial solar clusters")

    # ---- Step 4: Artificial clusters — frs_primary ----
    frs = load_geo(paths.get("frs_primary"), METRIC_CRS)
    if frs is not None and not frs.empty:
        added = 0
        for _, row in frs.iterrows():
            geom = row.geometry
            poly = geom.centroid.buffer(ARTIFICIAL_BUF_M) \
                if geom.geom_type != "Point" else geom.buffer(ARTIFICIAL_BUF_M)
            if pred_extent_union is not None and not poly.intersects(pred_extent_union):
                continue
            if hdbscan_union is not None and poly.intersects(hdbscan_union):
                continue
            all_rows.append({
                "cluster_id":     next_id,
                "dominant_class": "industrial",
                "point_count":    0,
                "mean_prob":      None,
                "n_industrial":   1,
                "n_substation":   0,
                "n_solar":        0,
                "n_residential":  0,
                "source":         "frs",
                "geometry":       poly,
            })
            next_id += 1; added += 1
        print(f"  added {added} artificial frs_primary clusters")

    # ---- Combine model + artificial ----
    artificial_gdf = gpd.GeoDataFrame(all_rows, crs=METRIC_CRS) if all_rows \
        else gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)

    if minor_model.empty and artificial_gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=METRIC_CRS)

    parts_to_concat = [df for df in [minor_model, artificial_gdf] if not df.empty]
    result = gpd.GeoDataFrame(
        pd.concat(parts_to_concat, ignore_index=True),
        geometry="geometry", crs=METRIC_CRS,
    )
    print(f"  total minor clusters: {len(result)}")
    return result


# ==========================================
#                  MAIN
# ==========================================

def _process_state(abbrev: str, project_dir: Path, force: bool = False) -> None:
    paths = get_state_paths(abbrev)

    state_border = get_state_border_geom(abbrev)

    training_extent_path = project_dir / "training_extent.gpkg"
    if training_extent_path.exists():
        te = gpd.read_file(training_extent_path).to_crs(METRIC_CRS)
        te_clipped = te.geometry.unary_union.intersection(state_border)
        grid_range = load_geo(
            BASE_DIR / "grid_ranges" / "3_phase" / f"{abbrev}_3phase_range.parquet", METRIC_CRS
        )
        if grid_range is not None and not grid_range.empty:
            te_clipped = te_clipped.intersection(grid_range.geometry.unary_union)
        te_gdf = gpd.GeoDataFrame(geometry=[te_clipped], crs=METRIC_CRS)
        paths["training_range"] = te_gdf
    else:
        print(f"  [warn] No training_extent.gpkg in {project_dir} — using grid training_range")

    prediction_extent_path = project_dir / "prediction_extent.gpkg"
    if prediction_extent_path.exists():
        pe = gpd.read_file(prediction_extent_path).to_crs(METRIC_CRS)
        pe_clipped = pe.geometry.unary_union.intersection(state_border)
        pe_gdf = gpd.GeoDataFrame(geometry=[pe_clipped], crs=METRIC_CRS)
        paths["prediction_extent"] = pe_gdf
    else:
        print(f"  [warn] No prediction_extent.gpkg in {project_dir} — using training_range")

    out = project_dir / "clusters"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Cluster Pipeline — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f" Project : {project_dir.name}")
    print(f" Output  : {out}")
    print(f"{'='*60}")

    p_train_feats    = out / f"{abbrev}_training_features.gpkg"
    p_pred_feats     = out / f"{abbrev}_prediction_features.gpkg"
    p_predicted      = out / f"{abbrev}_points_predicted.gpkg"
    p_clusters       = out / f"{abbrev}_cluster_polygons.gpkg"
    p_clusters_minor = out / f"{abbrev}_cluster_polygons_minor.gpkg"
    p_anti           = out / f"{abbrev}_anti_cluster_polygons.gpkg"

    # ---- Phases 1 + 2: training points + features ----
    if not force and p_train_feats.exists():
        print(f"\n[cached] {p_train_feats.name} — loading (--force to rerun)")
        train_pts = gpd.read_file(p_train_feats).to_crs(METRIC_CRS)
    else:
        train_pts = generate_training_points(paths)
        print("\n[Phase 2] Assigning features to training points...")
        train_pts = assign_features(train_pts, paths)
        train_pts.to_crs(FINAL_CRS).to_file(p_train_feats, driver="GPKG")
        print(f"  saved → {p_train_feats.name}")

    # ---- Phases 3 + 4: prediction candidates + features ----
    if not force and p_pred_feats.exists():
        print(f"\n[cached] {p_pred_feats.name} — loading (--force to rerun)")
        pred_pts = gpd.read_file(p_pred_feats).to_crs(METRIC_CRS)
    else:
        pred_pts = generate_prediction_candidates(paths)
        print("\n[Phase 4] Assigning features to prediction candidates...")
        pred_pts = assign_features(pred_pts, paths)
        pred_pts.to_crs(FINAL_CRS).to_file(p_pred_feats, driver="GPKG")
        print(f"  saved → {p_pred_feats.name}")

    # ---- Phase 5: train + predict ----
    pred_pts = train_and_predict(train_pts, pred_pts)
    pred_pts.to_crs(FINAL_CRS).to_file(p_predicted, driver="GPKG")
    print(f"\n  saved → {p_predicted.name}  ({len(pred_pts):,} points with grid_prob)")

    # ---- Phase 6: cluster polygons ----
    clusters = generate_cluster_polygons(pred_pts)
    clusters.to_crs(FINAL_CRS).to_file(p_clusters, driver="GPKG")
    print(f"  saved → {p_clusters.name}  ({len(clusters)} clusters)")

    # ---- Phase 7: anti-cluster polygons ----
    anti = generate_anti_cluster_polygons(pred_pts)
    anti.to_crs(FINAL_CRS).to_file(p_anti, driver="GPKG")
    print(f"  saved → {p_anti.name}  ({len(anti)} anti-clusters)")

    # ---- Phase 8: minor cluster polygons ----
    minor = generate_minor_cluster_polygons(pred_pts, clusters, paths)
    minor.to_crs(FINAL_CRS).to_file(p_clusters_minor, driver="GPKG")
    print(f"  saved → {p_clusters_minor.name}  ({len(minor)} minor clusters)")

    print(f"\n{'='*60}")
    print(f" DONE — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f"{'='*60}\n")


def main() -> None:
    project_dir = BASE_DIR / PROJECT

    print("\n" + "=" * 60)
    print("PREDICTION CLUSTERS")
    print("=" * 60)
    for state in PREDICTION_STATES:
        abbrev = _NAME_TO_ABBREV.get(state, state)
        _process_state(abbrev, project_dir, force=FORCE)

    print("\n" + "=" * 60)
    print("TRAINING CLUSTERS")
    print("=" * 60)
    for state in TRAINING_STATES:
        abbrev = _NAME_TO_ABBREV.get(state, state)
        _process_state(abbrev, project_dir, force=FORCE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster pipeline — training → features → model → polygons.")
    parser.add_argument("--project", default=PROJECT,           help="Project folder name (default: %(default)s)")
    parser.add_argument("--predict", nargs="+", default=PREDICTION_STATES, metavar="STATE", help="State(s) for prediction clusters")
    parser.add_argument("--train",   nargs="+", default=TRAINING_STATES,   metavar="STATE", help="State(s) for training clusters")
    parser.add_argument("--force", "-f", action="store_true",   help="Rerun all steps even if intermediate outputs already exist")
    parser.add_argument("--exclude-grid-from-prediction", action="store_true", help="Exclude 3-phase grid range from prediction extent")
    args = parser.parse_args()

    PROJECT                      = args.project
    PREDICTION_STATES            = args.predict
    TRAINING_STATES              = args.train
    FORCE                        = args.force
    EXCLUDE_GRID_FROM_PREDICTION = args.exclude_grid_from_prediction

    main()
