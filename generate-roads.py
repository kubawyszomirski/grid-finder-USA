#!/usr/bin/env python3
"""
generate-roads.py
Generates ML features for every road segment in roads_clean.

Reads   raw_data/<STATE>/TRANSPORT/<STATE>_roads_clean.gpkg
Writes  raw_data/<STATE>/TRANSPORT/<STATE>_roads_features.gpkg

Usage:
    python generate-roads.py Massachusetts
    python generate-roads.py Massachusetts --train   # also adds grid feature
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from generate_utils import (
    METRIC_CRS, FINAL_CRS, BASE_DIR, STATE_FULL_NAMES, _NAME_TO_ABBREV,
    CDL_3PHASE_CODES, NLCD_DEVELOPED, NLCD_NATURE, NLCD_CROPS,
    _find, load_geo, _raster_pct,
)

warnings.filterwarnings("ignore")

# ==========================================
#               CONFIGURATION
# ==========================================

BUFFER_DIST = 250  # metres — buffer radius for land / building features


# ==========================================
#            PATH RESOLUTION
# ==========================================

def get_state_paths(abbrev: str) -> dict:
    raw = BASE_DIR / "raw_data" / abbrev
    s   = abbrev
    t   = raw / "TRANSPORT"
    return {
        # ---- Road datasets produced by roads.py ----
        "roads_clean":              t / f"{s}_roads_clean.gpkg",
        "roads_raw":                t / f"{s}_roads_raw.gpkg",   # full network (routing reference)
        # ---- Grid ----
        "grid_lines":               BASE_DIR / "grid_data" / "3_phase" / f"{s}_3phase.parquet",
        # ---- Generators ----
        "solar":                    _find(raw / "GENERATORS", f"{s}_solar_merged"),
        "wind":                     _find(raw / "GENERATORS", f"{s}_wind_merged"),
        "gas_and_hydro":            _find(raw / "GENERATORS", f"{s}_eia_gas_hydro"),
        "bess_charging_stations":   _find(raw / "GENERATORS", f"{s}_bess_charging_stations"),
        # ---- Industry ----
        "oil_gas_chemical":         _find(raw / "INDUSTRY",    f"{s}_oil_gas_chemical"),
        "industry":                 _find(raw / "INDUSTRY",    f"{s}_industry"),
        "works":                    _find(raw / "INDUSTRY",    f"{s}_works"),
        # ---- Other infrastructure ----
        "mining":                   _find(raw / "MINING",      f"{s}_mining_final"),
        "utilities":                _find(raw / "UTILITIES",   f"{s}_utilities_merged"),
        "frs_primary":              _find(raw / "FRS",         f"{s}_frs_primary_merged"),
        "frs_secondary":            _find(raw / "FRS",         f"{s}_frs_secondary_merged"),
        "towers_major":             _find(raw / "TELECOM",     f"{s}_towers_major"),
        "towers_minor":             _find(raw / "TELECOM",     f"{s}_towers_minor"),
        "agriculture":              _find(raw / "AGRICULTURE", f"{s}_ag_farms_merged"),
        "public_infra":             _find(raw / "PUBLIC",      f"{s}_public_infra"),
        "hospitality":              _find(raw / "PUBLIC",      f"{s}_hospitality"),
        "big_buildings_and_clusters": _find(raw / "BUILDINGS", f"{s}_fema_building_clusters"),
        "railway":                  _find(t,                   f"{s}_rail_network"),
        "substations":              _find(raw / "SUBSTATIONS", f"{s}_substations_final"),
        "dams":                     _find(t,                   f"{s}_dams_major"),
        "transmission_lines":       _find(raw / "GRID",        f"{s}_transmission_lines"),
        # ---- Linear overlaps ----
        "tunnels":                  _find(t, f"{s}_tunnels_major"),
        "bridges":                  _find(t, f"{s}_bridges_lines"),
        "scenic_byways":            _find(raw / "LAND",        f"{s}_scenic_byways"),
        # ---- Buildings & land ----
        "fema_buildings":           _find(raw / "BUILDINGS",   f"{s}_fema_buildings"),
        "wetlands":                 _find(raw / "LAND",        f"{s}_wetlands"),
        "vrm":                      _find(raw / "LAND",        f"{s}_vrm"),
        "parcels":                  _find(raw / "LAND",        f"{s}_parcels"),
        # ---- Rasters ----
        "cdl":   raw / "LAND" / f"{s}_cdl.tif",
        "nlcd":  raw / "LAND" / f"{s}_nlcd.tif",
        "dem":   raw / "LAND" / f"{s}_dem.tif",
        "lanid": raw / "LAND" / f"{s}_lanid.tif",
    }


# ==========================================
#              HELPERS
# ==========================================

def road_curvature(line) -> float:
    """Total turning angle (radians) per metre of road length."""
    coords = np.array(line.coords)
    if len(coords) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(coords) - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        total += np.arccos(cos_a)
    return total / max(line.length, 1e-6)


def _snap_infra_to_points(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace polygon geometries with their centroids for distance queries."""
    data = data[["geometry"]].copy().reset_index(drop=True)
    non_point = ~data.geometry.geom_type.isin(["Point", "MultiPoint"])
    if non_point.any():
        data.loc[non_point, "geometry"] = data.loc[non_point, "geometry"].centroid
    return data


# ==========================================
#          FEATURE: GRID (training only)
# ==========================================

def compute_grid(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """Binary: ≥50 % of the road runs alongside a grid line (buffer 20 m)."""
    print("   grid")
    roads = roads.copy()
    grid_data = load_geo(paths.get("grid_lines"), METRIC_CRS)
    if grid_data is None or grid_data.empty:
        roads["grid"] = 0
        return roads
    grid_buf = grid_data.geometry.buffer(20).unary_union
    road_len  = roads.geometry.length.replace(0, np.nan)
    overlap   = roads.geometry.intersection(grid_buf).length
    roads["grid"] = ((overlap / road_len).fillna(0) >= 0.50).astype(int)
    return roads


# ==========================================
#       FEATURE: INFRASTRUCTURE DISTANCES
# ==========================================

# (column_suffix, path_key)
DISTANCE_INFRA: list[tuple[str, str]] = [
    ("solar",                    "solar"),
    ("wind",                     "wind"),
    ("gas_and_hydro",            "gas_and_hydro"),
    ("bess_charging_stations",   "bess_charging_stations"),
    ("oil_gas_chemical",         "oil_gas_chemical"),
    ("mining",                   "mining"),
    ("utilities",                "utilities"),
    ("industry",                 "industry"),
    ("frs_primary",              "frs_primary"),
    ("frs_secondary",            "frs_secondary"),
    ("towers_major",             "towers_major"),
    ("towers_minor",             "towers_minor"),
    ("agriculture",              "agriculture"),
    ("works",                    "works"),
    ("public_infra",             "public_infra"),
    ("big_buildings_and_clusters", "big_buildings_and_clusters"),
    ("railway",                  "railway"),
    ("substations",              "substations"),
    ("dams",                     "dams"),
    ("hospitality",              "hospitality"),
    ("transmission_lines",       "transmission_lines"),
]


def compute_distances(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """
    Euclidean distance from each road's centroid to the nearest infrastructure
    feature.  Polygonal infra (solar etc.) is snapped to centroid before querying.
    """
    roads = roads.copy()
    centroids = gpd.GeoDataFrame(
        {"road_id": roads["road_id"].values},
        geometry=roads.geometry.centroid,
        crs=METRIC_CRS,
    )

    for name, key in DISTANCE_INFRA:
        col = f"dist_{name}"
        print(f"   dist → {name}")
        data = load_geo(paths.get(key), METRIC_CRS)
        if data is None or data.empty:
            roads[col] = np.nan
            continue

        infra = _snap_infra_to_points(data)
        # Drop stale join-index columns that would conflict with sjoin_nearest
        infra = infra.drop(columns=[c for c in ("index_left", "index_right") if c in infra.columns])

        joined = gpd.sjoin_nearest(centroids, infra, how="left", distance_col=col)
        joined = joined[~joined.index.duplicated(keep="first")]
        roads[col] = joined[col].values

    return roads


# ==========================================
#        FEATURE: BUILDING BUFFER STATS
# ==========================================

def compute_building_stats(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """Total area, median size, and std of FEMA building footprints within 250 m."""
    print("   building stats (250 m)")
    roads = roads.copy()
    for col in ["bldg_total_area_250m", "bldg_median_m2_250m", "bldg_std_m2_250m"]:
        roads[col] = 0.0

    bldgs = load_geo(paths.get("fema_buildings"), METRIC_CRS)
    if bldgs is None or bldgs.empty:
        return roads

    bldgs = bldgs.copy()
    bldgs["_area"] = bldgs.geometry.area
    sindex  = bldgs.sindex
    buffers = roads.geometry.buffer(BUFFER_DIST)

    total_areas, medians, stds = [], [], []
    for buf in buffers:
        cands = list(sindex.query(buf, predicate="intersects"))
        if not cands:
            total_areas.append(0.0); medians.append(0.0); stds.append(0.0)
            continue
        local = bldgs.iloc[cands]
        areas = local["_area"]
        total_areas.append(float(areas.sum()))
        medians.append(float(areas.median()))
        stds.append(float(areas.std()) if len(areas) > 1 else 0.0)

    roads["bldg_total_area_250m"] = total_areas
    roads["bldg_median_m2_250m"]  = medians
    roads["bldg_std_m2_250m"]     = stds
    return roads


# ==========================================
#      FEATURE: RASTER BUFFER (CDL / NLCD)
# ==========================================

def compute_raster_buffer(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    roads = roads.copy()

    # Buffer in METRIC_CRS, then reproject to each raster's CRS
    road_bufs_metric = gpd.GeoDataFrame(
        geometry=roads.geometry.buffer(BUFFER_DIST), crs=METRIC_CRS
    )

    # ---- CDL (3-phase crop fraction) ----
    roads["cdl_3phase_pct_250m"] = 0.0
    cdl_path = paths.get("cdl")
    if cdl_path and Path(cdl_path).exists():
        print("   CDL 250 m")
        with rasterio.open(cdl_path) as src:
            bufs = road_bufs_metric.to_crs(src.crs).geometry
            roads["cdl_3phase_pct_250m"] = [_raster_pct(src, g, CDL_3PHASE_CODES) for g in bufs]

    # ---- NLCD ----
    for col in ["nlcd_dev_pct_250m", "nlcd_nature_pct_250m", "nlcd_crops_pct_250m"]:
        roads[col] = 0.0
    nlcd_path = paths.get("nlcd")
    if nlcd_path and Path(nlcd_path).exists():
        print("   NLCD 250 m")
        with rasterio.open(nlcd_path) as src:
            bufs = road_bufs_metric.to_crs(src.crs).geometry
            dev, nat, crop = [], [], []
            for g in bufs:
                dev.append( _raster_pct(src, g, NLCD_DEVELOPED))
                nat.append( _raster_pct(src, g, NLCD_NATURE))
                crop.append(_raster_pct(src, g, NLCD_CROPS))
            roads["nlcd_dev_pct_250m"]    = dev
            roads["nlcd_nature_pct_250m"] = nat
            roads["nlcd_crops_pct_250m"]  = crop

    return roads


# ==========================================
#       FEATURE: VECTOR AREA BUFFER
# ==========================================

def _vector_pct_buffer(roads: gpd.GeoDataFrame, path, col: str) -> gpd.GeoDataFrame:
    """% of each road's 250 m buffer area covered by vector polygons."""
    roads = roads.copy()
    roads[col] = 0.0
    data = load_geo(path, METRIC_CRS)
    if data is None or data.empty:
        return roads

    sindex = data.sindex
    pcts = []
    for road_geom in roads.geometry:
        buf   = road_geom.buffer(BUFFER_DIST)
        cands = list(sindex.query(buf, predicate="intersects"))
        if not cands:
            pcts.append(0.0)
            continue
        local   = data.iloc[cands]
        overlap = local.geometry.intersection(buf).area.sum()
        pcts.append(float(overlap / buf.area * 100.0))
    roads[col] = pcts
    return roads


# ==========================================
#        FEATURE: LANID DISTANCE
# ==========================================

def compute_lanid_distance(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """Distance from road centroid to nearest irrigated-field pixel (LANID > 0)."""
    print("   LANID distance")
    roads = roads.copy()
    roads["dist_lanid"] = np.nan
    lanid_path = paths.get("lanid")
    if not lanid_path or not Path(lanid_path).exists():
        return roads

    with rasterio.open(lanid_path) as src:
        data = src.read(1)
        mask = data > 0
        if not mask.any():
            return roads
        pts = [shape(g).centroid for g, _ in rio_shapes(data, mask=mask, transform=src.transform)]
        lanid_crs = src.crs

    lanid_pts = gpd.GeoDataFrame(geometry=pts, crs=lanid_crs).to_crs(METRIC_CRS)
    centroids  = gpd.GeoDataFrame(
        {"road_id": roads["road_id"].values},
        geometry=roads.geometry.centroid,
        crs=METRIC_CRS,
    )
    joined = gpd.sjoin_nearest(centroids, lanid_pts[["geometry"]], how="left", distance_col="dist_lanid")
    joined = joined[~joined.index.duplicated(keep="first")]
    roads["dist_lanid"] = joined["dist_lanid"].values
    return roads


# ==========================================
#    FEATURE: LINEAR OVERLAPS & TERRAIN
# ==========================================

def _overlap_pct(roads: gpd.GeoDataFrame, data: gpd.GeoDataFrame | None, buffer_m: float = 5.0) -> np.ndarray:
    """% of each road's length inside a buffer around linear/polygon features."""
    if data is None or data.empty:
        return np.zeros(len(roads))
    buf     = data.geometry.buffer(buffer_m).unary_union
    lengths = roads.geometry.length.replace(0, np.nan)
    overlap = roads.geometry.intersection(buf).length
    return (overlap / lengths).fillna(0).values * 100.0


def _compute_perpendicular_slope(roads: gpd.GeoDataFrame, dem_path) -> np.ndarray:
    """
    Average terrain slope perpendicular to the road.
    Samples elevation 50 m to each side every 30 m along the road,
    returns mean of arctan(|ΔElev| / 100 m) in degrees.
    """
    STEP_M = 30.0
    PERP_M = 50.0

    if not dem_path or not Path(dem_path).exists():
        return np.full(len(roads), np.nan)

    with rasterio.open(dem_path) as src:
        transformer = pyproj.Transformer.from_crs(METRIC_CRS, src.crs, always_xy=True)
        scores = []

        for line in roads.geometry:
            if line is None or line.length < STEP_M:
                scores.append(np.nan)
                continue

            left_pts, right_pts = [], []
            for d in np.arange(0, line.length, STEP_M):
                pt  = line.interpolate(d)
                pt2 = line.interpolate(min(d + 1.0, line.length))
                dx, dy = pt2.x - pt.x, pt2.y - pt.y
                norm = np.sqrt(dx ** 2 + dy ** 2)
                if norm < 1e-9:
                    continue
                px = -dy / norm * PERP_M
                py =  dx / norm * PERP_M
                left_pts.append( (pt.x + px, pt.y + py))
                right_pts.append((pt.x - px, pt.y - py))

            if not left_pts:
                scores.append(np.nan)
                continue

            all_pts = left_pts + right_pts
            xs, ys  = zip(*all_pts)
            lon, lat = transformer.transform(list(xs), list(ys))

            try:
                elevs = np.array(
                    list(src.sample(zip(lon, lat), indexes=1)), dtype=np.float32
                ).flatten()
                if src.nodata is not None:
                    elevs[elevs == src.nodata] = np.nan

                n = len(left_pts)
                el, er = elevs[:n], elevs[n:]
                valid  = ~(np.isnan(el) | np.isnan(er))
                if valid.sum() < 2:
                    scores.append(np.nan)
                    continue

                slopes = np.abs(el[valid] - er[valid]) / (2.0 * PERP_M)
                scores.append(float(np.degrees(np.arctan(np.nanmean(slopes)))))
            except Exception:
                scores.append(np.nan)

    return np.array(scores)


def compute_linear_features(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    roads = roads.copy()

    # ---- % of road inside a tunnel ----
    print("   tunnel %")
    tunnels = load_geo(paths.get("tunnels"), METRIC_CRS)
    roads["tunnel_pct"] = _overlap_pct(roads, tunnels, buffer_m=5.0)

    # ---- Bridge binary (any bridge ≥ 50 m within 5 m of road) ----
    print("   bridge binary")
    roads["has_bridge"] = 0
    bridges = load_geo(paths.get("bridges"), METRIC_CRS)
    if bridges is not None and not bridges.empty:
        bridges = bridges[bridges.geometry.length >= 50].copy()
        if not bridges.empty:
            road_buf     = roads.geometry.buffer(5)
            bridge_union = bridges.geometry.buffer(1).unary_union
            roads["has_bridge"] = road_buf.intersects(bridge_union).astype(int)

    # ---- % of road overlapping a scenic byway ----
    print("   scenic byway %")
    scenic = load_geo(paths.get("scenic_byways"), METRIC_CRS)
    roads["scenic_byway_pct"] = _overlap_pct(roads, scenic, buffer_m=5.0)

    # ---- Geometric curvature (radians / metre) ----
    print("   curvature")
    roads["curvature_per_m"] = [road_curvature(g) for g in roads.geometry]

    # ---- Perpendicular terrain slope ----
    print("   perpendicular slope")
    roads["perp_slope_deg"] = _compute_perpendicular_slope(roads, paths.get("dem"))

    return roads


# ==========================================
#       FEATURE: PER-NAME ROAD STATS
# ==========================================

def compute_per_name_features(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    For each unique road name (excluding None/empty), compute across all same-name
    segments:
      name_total_length_m   — total combined length
      name_curvature_per_m  — length-weighted curvature
    Roads without a name receive NaN.
    """
    roads = roads.copy()
    roads["name_total_length_m"]  = np.nan
    roads["name_curvature_per_m"] = np.nan

    if "name" not in roads.columns:
        return roads

    named_mask = roads["name"].notna() & (roads["name"].astype(str).str.strip() != "")
    if not named_mask.any():
        return roads

    roads["_len"] = roads.geometry.length

    # curvature_per_m is expected to already be present from compute_linear_features
    if "curvature_per_m" not in roads.columns:
        roads["curvature_per_m"] = [road_curvature(g) for g in roads.geometry]

    grp = roads.loc[named_mask].groupby("name")

    total_len = grp["_len"].sum()
    weighted_curv = grp.apply(
        lambda g: float(
            (g["curvature_per_m"] * g["_len"]).sum() / g["_len"].sum()
        ) if g["_len"].sum() > 0 else 0.0
    )

    roads.loc[named_mask, "name_total_length_m"]  = roads.loc[named_mask, "name"].map(total_len)
    roads.loc[named_mask, "name_curvature_per_m"] = roads.loc[named_mask, "name"].map(weighted_curv)

    roads = roads.drop(columns=["_len"])
    return roads


# ==========================================
#       FEATURE: PARCEL VALUE BUFFER
# ==========================================

_PARCEL_VALUE_COLS = ["imp_val", "lan_val", "tot_val", "tax_amt"]
_PARCEL_BUFFER_M   = 20  # metres along the road


def compute_parcel_values(roads: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """
    Weighted mean of parcel value columns for parcels within 20 m of each road.
    Weight = area of intersection between parcel and the 20 m buffer strip, which
    is proportional to how far the parcel extends alongside the road.
    """
    print("   parcel values (20 m)")
    roads = roads.copy()

    parcels = load_geo(paths.get("parcels"), METRIC_CRS)
    value_cols = [c for c in _PARCEL_VALUE_COLS if parcels is not None and c in parcels.columns] \
        if parcels is not None and not parcels.empty else []

    out_cols = [f"parcel_{c}" for c in _PARCEL_VALUE_COLS]
    for col in out_cols:
        roads[col] = np.nan

    if not value_cols:
        return roads

    for c in value_cols:
        parcels[c] = pd.to_numeric(parcels[c], errors="coerce")

    sindex  = parcels.sindex
    buffers = roads.geometry.buffer(_PARCEL_BUFFER_M)

    results: dict[str, list] = {c: [] for c in value_cols}
    for buf in buffers:
        cands = list(sindex.query(buf, predicate="intersects"))
        if not cands:
            for c in value_cols:
                results[c].append(np.nan)
            continue

        local = parcels.iloc[cands].copy()
        local["_w"] = local.geometry.intersection(buf).area
        total_w = local["_w"].sum()

        if total_w <= 0:
            for c in value_cols:
                results[c].append(np.nan)
            continue

        for c in value_cols:
            valid = local[c].notna()
            if not valid.any():
                results[c].append(0.0)
            else:
                w = local.loc[valid, "_w"]
                v = local.loc[valid, c]
                results[c].append(float((v * w).sum() / w.sum()))

    for c in value_cols:
        roads[f"parcel_{c}"] = results[c]

    return roads


# ==========================================
#          COMPUTE ALL FEATURES
# ==========================================

def compute_all_features(
    roads: gpd.GeoDataFrame,
    abbrev: str,
    is_training: bool,
) -> gpd.GeoDataFrame:
    paths = get_state_paths(abbrev)
    print(f"\n  [{abbrev}] Computing features for {len(roads):,} roads…")

    if is_training:
        roads = compute_grid(roads, paths)

    print("\n  [distances]")
    roads = compute_distances(roads, paths)

    print("\n  [building stats]")
    roads = compute_building_stats(roads, paths)

    print("\n  [raster buffer — CDL / NLCD]")
    roads = compute_raster_buffer(roads, paths)

    print("\n  [wetlands %]")
    roads = _vector_pct_buffer(roads, paths.get("wetlands"), "wetlands_pct_250m")

    print("\n  [VRM %]")
    roads = _vector_pct_buffer(roads, paths.get("vrm"), "vrm_pct_250m")

    print("\n  [LANID distance]")
    roads = compute_lanid_distance(roads, paths)

    print("\n  [linear features — curvature / slope / overlaps]")
    roads = compute_linear_features(roads, paths)

    print("\n  [per-name road stats]")
    roads = compute_per_name_features(roads)

    print("\n  [parcel values]")
    roads = compute_parcel_values(roads, paths)

    return roads


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Road feature generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate-roads.py Massachusetts\n"
            "  python generate-roads.py MA --train\n"
        ),
    )
    parser.add_argument("state", help="State full name or 2-letter abbreviation")
    parser.add_argument(
        "--train", action="store_true",
        help="Training mode — adds the 'grid' binary feature",
    )
    args = parser.parse_args()

    abbrev = _NAME_TO_ABBREV.get(args.state.strip())
    if abbrev is None:
        print(f"[ERROR] Unknown state: '{args.state}'")
        print(f"  Known states: {', '.join(sorted(STATE_FULL_NAMES.keys()))}")
        return

    paths = get_state_paths(abbrev)
    roads_path = paths["roads_clean"]

    if not roads_path.exists():
        print(f"[ERROR] roads_clean not found: {roads_path}")
        print("  → Run  python roads.py <state>  first to generate the road network.")
        return

    print(f"\nLoading {roads_path.name} …")
    roads = gpd.read_file(roads_path).to_crs(METRIC_CRS)
    print(f"  {len(roads):,} road segments")

    if "road_id" not in roads.columns:
        print("[Warning] road_id column missing — assigning sequential IDs")
        roads["road_id"] = [f"R_{i:06d}" for i in range(len(roads))]

    roads_out = compute_all_features(roads, abbrev, is_training=args.train)

    out_path = roads_path.parent / roads_path.name.replace("_roads_clean", "_roads_features")
    roads_out.to_crs(FINAL_CRS).to_file(out_path, driver="GPKG")
    print(
        f"\n[DONE] {out_path.name}"
        f"  ({len(roads_out):,} roads, {roads_out.shape[1]} columns)"
    )


if __name__ == "__main__":
    main()
