"""
generate-tiles.py
Generates prediction and training tiles for a given project.

Prediction tiles: state_border ∩ prediction_extent.gpkg, minus 3-phase grid range
Training tiles:   3-phase grid range extent for each training state
Both sets exclude tiles with ≥30% overlap with state exclusion zones.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from pathlib import Path
import argparse
import warnings

from generate_utils import (
    METRIC_CRS, FINAL_CRS, BASE_DIR, STATE_FULL_NAMES,
    CDL_3PHASE_CODES, NLCD_DEVELOPED, NLCD_NATURE, NLCD_CROPS,
    _find, load_geo, _raster_pct,
)

warnings.filterwarnings("ignore")

# ==========================================
#               CONFIGURATION
# ==========================================

PROJECT = "project-2"
PREDICTION_STATES = ["MA"]      # state(s) for prediction tiles
TRAINING_STATES = ["MA"]        # state(s) for training tiles

TILE_SIZE = 500                 # metres

# Minimum fraction of a tile that must overlap with the valid extent to keep it
EXTENT_OVERLAP_MIN = 0.70

# Maximum fraction of a tile that may overlap with exclusion zones before dropping
EXCLUSION_OVERLAP_MAX = 0.30

STATE_BORDERS_PATH = Path(
    "/Users/kuba/PycharmProjects/grid-research/data/state_borders/cb_2023_us_state_20m.shp"
)

ROAD_CLASSES = ["primary", "secondary", "tertiary", "residential", "service", "osm_grid"]

# ==========================================
#            PATH RESOLUTION
# ==========================================

def get_state_paths(state: str) -> dict:
    raw = BASE_DIR / "raw_data" / state
    s = state
    return {
        "exclusions":               raw / "EXCLUSIONS" / f"{s}_exclusions.gpkg",
        "grid_range":               BASE_DIR / "grid_ranges" / "3_phase" / f"{s}_3phase_range.parquet",
        "grid_lines":               BASE_DIR / "grid_data" / "3_phase" / f"{s}_3phase.parquet",
        # Generators
        "solar":                    _find(raw / "GENERATORS", f"{s}_solar_merged"),
        "wind":                     _find(raw / "GENERATORS", f"{s}_wind_merged"),
        "gas_and_hydro":            _find(raw / "GENERATORS", f"{s}_eia_gas_hydro"),
        "bess_charging_stations":   _find(raw / "GENERATORS", f"{s}_bess_charging_stations"),
        # Industry
        "oil_gas_chemical":         _find(raw / "INDUSTRY", f"{s}_oil_gas_chemical"),
        "industry":                 _find(raw / "INDUSTRY", f"{s}_industry"),
        "works":                    _find(raw / "INDUSTRY", f"{s}_works"),
        # Other vectors
        "mining":                   _find(raw / "MINING", f"{s}_mining_final"),
        "utilities":                _find(raw / "UTILITIES", f"{s}_utilities_merged"),
        "frs_primary":              _find(raw / "FRS", f"{s}_frs_primary_merged"),
        "frs_secondary":            _find(raw / "FRS", f"{s}_frs_secondary_merged"),
        "towers_major":             _find(raw / "TELECOM", f"{s}_towers_major"),
        "towers_minor":             _find(raw / "TELECOM", f"{s}_towers_minor"),
        "agriculture":              _find(raw / "AGRICULTURE", f"{s}_ag_farms_merged"),
        "public_infra":             _find(raw / "PUBLIC", f"{s}_public_infra"),
        "hospitality":              _find(raw / "PUBLIC", f"{s}_hospitality"),
        "big_buildings_and_clusters": _find(raw / "BUILDINGS", f"{s}_fema_building_clusters"),
        "railway":                  _find(raw / "TRANSPORT", f"{s}_rail_network"),
        "substations":              _find(raw / "SUBSTATIONS", f"{s}_substations_final"),
        "transmission_lines":       _find(raw / "GRID", f"{s}_transmission_lines"),
        "dams":                     _find(raw / "TRANSPORT", f"{s}_dams_major"),
        "roads_raw":                _find(raw / "TRANSPORT", f"{s}_roads_raw"),
        "wetlands":                 _find(raw / "LAND", f"{s}_wetlands"),
        "vrm":                      _find(raw / "LAND", f"{s}_vrm"),
        "fema_buildings":           _find(raw / "BUILDINGS", f"{s}_fema_buildings"),
        # Rasters
        "cdl":                      raw / "LAND" / f"{s}_cdl.tif",
        "lanid":                    raw / "LAND" / f"{s}_lanid.tif",
        "dem":                      raw / "LAND" / f"{s}_dem.tif",
        "nlcd":                     raw / "LAND" / f"{s}_nlcd.tif",
    }


# ==========================================
#              HELPERS
# ==========================================

def get_state_border_geom(state: str):
    """Return the state border as a single shapely geometry in METRIC_CRS."""
    states = gpd.read_file(STATE_BORDERS_PATH)
    name = STATE_FULL_NAMES[state]
    border = states[states["NAME"] == name].to_crs(METRIC_CRS)
    return border.geometry.unary_union


def make_grid(extent_geom, tile_size: int) -> gpd.GeoDataFrame:
    """Create a regular grid of square tiles covering extent_geom (METRIC_CRS)."""
    minx, miny, maxx, maxy = extent_geom.bounds
    xs = np.arange(minx, maxx, tile_size)
    ys = np.arange(miny, maxy, tile_size)
    cells = [box(x, y, x + tile_size, y + tile_size) for x in xs for y in ys]
    gdf = gpd.GeoDataFrame({"geometry": cells}, crs=METRIC_CRS)
    gdf["tile_id"] = range(len(gdf))
    return gdf


def filter_extent(tiles: gpd.GeoDataFrame, extent_geom) -> gpd.GeoDataFrame:
    tile_areas = tiles.geometry.area
    inside = tiles.geometry.intersection(extent_geom).area
    return tiles[(inside / tile_areas) >= EXTENT_OVERLAP_MIN].copy().reset_index(drop=True)


def filter_exclusions(tiles: gpd.GeoDataFrame, excl_path) -> gpd.GeoDataFrame:
    if excl_path is None or not Path(excl_path).exists():
        print(f"   [Warning] exclusions not found: {excl_path}")
        return tiles
    excl = gpd.read_file(excl_path).to_crs(METRIC_CRS)
    excl_union = excl.geometry.buffer(0).unary_union
    tile_areas = tiles.geometry.area
    overlap = tiles.geometry.intersection(excl_union).area
    return tiles[(overlap / tile_areas) < EXCLUSION_OVERLAP_MAX].copy().reset_index(drop=True)


def square_15km(geom):
    c = geom.centroid
    h = 7500.0
    return box(c.x - h, c.y - h, c.x + h, c.y + h)


def explode_geom(geom) -> list:
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


def connected_components(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
        labeled, _ = label(mask.astype(np.uint8), structure=np.ones((3, 3), np.uint8))
        return labeled
    except ImportError:
        from skimage.measure import label
        return label(mask, connectivity=2)


# ==========================================
#           TILE GENERATION
# ==========================================

def build_prediction_extent(project_dir: Path, states: list[str]):
    pred = load_geo(project_dir / "prediction_extent.gpkg", METRIC_CRS)
    pred_union = pred.geometry.unary_union

    extents = []
    for state in states:
        border = get_state_border_geom(state)
        clipped = border.intersection(pred_union)

        grid_range = load_geo(
            BASE_DIR / "grid_ranges" / "3_phase" / f"{state}_3phase_range.parquet", METRIC_CRS
        )
        if grid_range is not None and not grid_range.empty:
            clipped = clipped.difference(grid_range.geometry.unary_union)

        extents.append(clipped)

    return unary_union(extents)


def generate_prediction_tiles(project_dir: Path, states: list[str]) -> gpd.GeoDataFrame:
    print("[PREDICTION] Building extent...")
    extent = build_prediction_extent(project_dir, states)

    tiles = make_grid(extent, TILE_SIZE)
    tiles = filter_extent(tiles, extent)
    print(f"[PREDICTION] {len(tiles)} tiles after extent filter")

    for state in states:
        paths = get_state_paths(state)
        tiles = filter_exclusions(tiles, paths["exclusions"])

    tiles["tile_id"] = range(len(tiles))
    print(f"[PREDICTION] {len(tiles)} tiles after exclusion filter")
    return tiles


def generate_training_tiles(states: list[str]) -> gpd.GeoDataFrame:
    print("[TRAINING] Generating tiles per state...")
    all_tiles = []
    offset = 0
    for state in states:
        grid_range = load_geo(
            BASE_DIR / "grid_ranges" / "3_phase" / f"{state}_3phase_range.parquet", METRIC_CRS
        )
        if grid_range is None:
            print(f"  [{state}] grid range not found — skipping")
            continue

        extent = grid_range.geometry.unary_union
        tiles = make_grid(extent, TILE_SIZE)
        tiles = filter_extent(tiles, extent)

        paths = get_state_paths(state)
        tiles = filter_exclusions(tiles, paths["exclusions"])

        tiles["state"] = state
        tiles["tile_id"] = range(offset, offset + len(tiles))
        offset += len(tiles)
        all_tiles.append(tiles)
        print(f"  [{state}] {len(tiles)} training tiles")

    combined = gpd.GeoDataFrame(pd.concat(all_tiles, ignore_index=True), crs=METRIC_CRS)
    print(f"[TRAINING] Total: {len(combined)} tiles")
    return combined


# ==========================================
#          FEATURE: INFRASTRUCTURE
# ==========================================

# (type, path_key)
INFRA_CONFIG: dict[str, tuple[str, str]] = {
    "solar":                    ("continuous", "solar"),
    "wind":                     ("discrete",   "wind"),
    "gas_and_hydro":            ("binary",     "gas_and_hydro"),
    "bess_charging_stations":   ("discrete",   "bess_charging_stations"),
    "oil_gas_chemical":         ("binary",     "oil_gas_chemical"),
    "mining":                   ("discrete",   "mining"),
    "industry":                 ("binary",     "industry"),
    "works":                    ("discrete",   "works"),
    "utilities":                ("discrete",   "utilities"),
    "frs_primary":              ("discrete",   "frs_primary"),
    "frs_secondary":            ("discrete",   "frs_secondary"),
    "towers_major":             ("binary",     "towers_major"),
    "towers_minor":             ("discrete",   "towers_minor"),
    "agriculture":              ("binary",     "agriculture"),
    "public_infra":             ("discrete",   "public_infra"),
    "hospitality":              ("discrete",   "hospitality"),
    "big_buildings_and_clusters": ("discrete", "big_buildings_and_clusters"),
    "railway":                  ("binary",     "railway"),
    "substations":              ("binary",     "substations"),
    "transmission_lines":       ("binary",     "transmission_lines"),
    "dams":                     ("discrete",   "dams"),
}


def _count_blobs(tile_geom, data: gpd.GeoDataFrame, sindex, buffer: float = 25.0) -> int:
    cands = list(sindex.query(tile_geom, predicate="intersects"))
    if not cands:
        return 0
    local = data.iloc[cands]
    if local.empty:
        return 0
    merged = unary_union(local.geometry.buffer(buffer))
    return len(explode_geom(merged))


def compute_infra(tiles: gpd.GeoDataFrame, paths: dict, is_training: bool) -> gpd.GeoDataFrame:
    tiles = tiles.copy()

    # Grid — training only
    if is_training:
        print("   grid")
        grid_data = load_geo(paths["grid_lines"], METRIC_CRS)
        if grid_data is not None and not grid_data.empty:
            joined = gpd.sjoin(tiles[["tile_id", "geometry"]], grid_data, how="inner", predicate="intersects")
            tiles["grid"] = tiles["tile_id"].isin(joined["tile_id"]).astype(int)
        else:
            tiles["grid"] = 0

    for name, (mode, key) in INFRA_CONFIG.items():
        print(f"   {name}")
        path = paths.get(key)
        data = load_geo(path, METRIC_CRS)
        if data is None or data.empty:
            tiles[name] = 0
            continue
        data = data[["geometry"]].copy().reset_index(drop=True)
        # buffer(0) fixes invalid polygon topology but destroys points/lines — apply only to polygons
        geom_types = data.geometry.geom_type.str.lower()
        is_poly = geom_types.str.contains("polygon")
        if is_poly.any():
            data.loc[is_poly, "geometry"] = data.loc[is_poly, "geometry"].buffer(0)

        if mode == "binary":
            joined = gpd.sjoin(tiles[["tile_id", "geometry"]], data, how="inner", predicate="intersects")
            tiles[name] = tiles["tile_id"].isin(joined["tile_id"]).astype(int)

        elif mode == "discrete":
            sindex = data.sindex
            tiles[name] = [_count_blobs(g, data, sindex) for g in tiles.geometry]

        elif mode == "continuous":
            sindex = data.sindex
            areas = []
            for geom in tiles.geometry:
                cands = list(sindex.query(geom, predicate="intersects"))
                local = data.iloc[cands] if cands else data.iloc[[]]
                areas.append(float(local.geometry.intersection(geom).area.sum()))
            tiles[name] = areas

    return tiles


# ==========================================
#            FEATURE: ROADS
# ==========================================

def compute_roads(tiles: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    print("   roads")
    tiles = tiles.copy()
    road_cols = [f"road_m__{c}" for c in ROAD_CLASSES]
    for col in road_cols:
        tiles[col] = 0.0

    path = paths.get("roads_raw")
    if path is None or not Path(path).exists():
        print("   [Skipping] roads_raw not found")
        return tiles

    roads = load_geo(path, METRIC_CRS)
    roads = roads[roads["highway"].isin(ROAD_CLASSES)][["highway", "geometry"]].copy().reset_index(drop=True)
    if roads.empty:
        return tiles

    tiles_sub = tiles[["tile_id", "geometry"]].copy()
    joined = gpd.sjoin(roads, tiles_sub, how="inner", predicate="intersects")
    if joined.empty:
        return tiles

    # reset_index so each (road, tile) pair has a unique integer index;
    # without this, a road matching N tiles has N rows with the same index and
    # GeoSeries.intersection aligns by index label, causing wrong geometry pairs
    joined = joined.reset_index(drop=True)
    tile_geom_map = tiles.set_index("tile_id")["geometry"]
    joined["tile_geom"] = joined["tile_id"].map(tile_geom_map)
    joined["len_m"] = joined.geometry.intersection(gpd.GeoSeries(joined["tile_geom"], crs=METRIC_CRS)).length

    sums = joined.groupby(["tile_id", "highway"])["len_m"].sum().reset_index()
    wide = sums.pivot_table(index="tile_id", columns="highway", values="len_m", fill_value=0.0)
    wide.columns = [f"road_m__{c}" for c in wide.columns]

    # drop pre-initialised columns and re-attach from pivot
    tiles = tiles.drop(columns=[c for c in road_cols if c in tiles.columns])
    tiles = tiles.merge(wide.reset_index(), on="tile_id", how="left")
    for col in road_cols:
        if col not in tiles.columns:
            tiles[col] = 0.0
        else:
            tiles[col] = tiles[col].fillna(0.0)

    return tiles


# ==========================================
#          FEATURE: RASTERS
# ==========================================

def compute_nlcd(tiles: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    print("   NLCD")
    tiles = tiles.copy()
    nlcd_path = paths.get("nlcd")
    cols = ["pct_developed_tile", "pct_nature_tile", "pct_nature_15km", "pct_crops_tile", "pct_crops_15km"]
    for col in cols:
        tiles[col] = 0.0

    if not nlcd_path or not Path(nlcd_path).exists():
        return tiles

    with rasterio.open(nlcd_path) as src:
        tiles_proj = tiles.to_crs(src.crs)
        dev, nat_t, nat_15, crop_t, crop_15 = [], [], [], [], []
        total = len(tiles_proj)
        for i, geom in enumerate(tiles_proj.geometry):
            dev.append(_raster_pct(src, geom, NLCD_DEVELOPED))
            nat_t.append(_raster_pct(src, geom, NLCD_NATURE))
            crop_t.append(_raster_pct(src, geom, NLCD_CROPS))
            sq = square_15km(geom)
            nat_15.append(_raster_pct(src, sq, NLCD_NATURE))
            crop_15.append(_raster_pct(src, sq, NLCD_CROPS))
            if (i + 1) % 500 == 0:
                print(f"      {i+1}/{total}")

    tiles["pct_developed_tile"] = dev
    tiles["pct_nature_tile"] = nat_t
    tiles["pct_nature_15km"] = nat_15
    tiles["pct_crops_tile"] = crop_t
    tiles["pct_crops_15km"] = crop_15
    return tiles


def compute_terrain(tiles: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    print("   terrain")
    tiles = tiles.copy()
    dem_path = paths.get("dem")
    tiles["undulation_score"] = np.nan
    if not dem_path or not Path(dem_path).exists():
        return tiles

    TARGET_RES = 30.0

    def _score(src, geom_m):
        try:
            bounds_src = transform_bounds(METRIC_CRS, src.crs, *geom_m.bounds, densify_pts=21)
            window = src.window(*bounds_src)
            raw = src.read(1, window=window, masked=False).astype("float32")
            src_tf = src.window_transform(window)
            nodata = src.nodata if src.nodata is not None else -9999.0
            minx, miny, maxx, maxy = geom_m.bounds
            w = max(1, int(np.ceil((maxx - minx) / TARGET_RES)))
            h = max(1, int(np.ceil((maxy - miny) / TARGET_RES)))
            dst_tf = rasterio.transform.from_origin(minx, maxy, TARGET_RES, TARGET_RES)
            dem_out = np.full((h, w), nodata, dtype="float32")
            reproject(
                raw, dem_out,
                src_transform=src_tf, src_crs=src.crs,
                dst_transform=dst_tf, dst_crs=METRIC_CRS,
                resampling=Resampling.bilinear,
                src_nodata=nodata, dst_nodata=nodata,
            )
            dem_out[dem_out == nodata] = np.nan
            mask = rasterize(
                [(mapping(geom_m), 1)], out_shape=(h, w), transform=dst_tf, fill=0
            ).astype(bool)
            dem_out = np.where(mask, dem_out, np.nan)
            dz_dy, dz_dx = np.gradient(dem_out, TARGET_RES, TARGET_RES)
            slopes = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
            s = slopes[np.isfinite(slopes)]
            if s.size == 0:
                return np.nan
            raw_score = (0.7 * np.mean(s) + 0.3 * np.percentile(s, 90)) / 30.0
            return float(np.clip(raw_score, 0.0, 1.0) * 100.0)
        except Exception:
            return np.nan

    with rasterio.open(dem_path) as src:
        scores = []
        total = len(tiles)
        for i, geom in enumerate(tiles.geometry):
            scores.append(_score(src, geom))
            if (i + 1) % 250 == 0:
                print(f"      {i+1}/{total}")

    tiles["undulation_score"] = scores
    return tiles


def compute_cdl(tiles: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    print("   CDL")
    tiles = tiles.copy()
    tiles["pct_3phase_crops"] = 0.0
    tiles["mean_field_size_m2"] = 0.0

    cdl_path = paths.get("cdl")
    if not cdl_path or not Path(cdl_path).exists():
        return tiles

    def _process(src, geom):
        try:
            window = geometry_window(src, [mapping(geom)])
            data = src.read(1, window=window)
            tf = src.window_transform(window)
            pixel_area = abs(src.transform.a * src.transform.e)
            tile_mask = rasterize(
                [(mapping(geom), 1)], out_shape=data.shape, transform=tf, fill=0, all_touched=False
            ).astype(bool)
            if src.nodata is not None:
                tile_mask &= (data != src.nodata)
            denom = tile_mask.sum()
            if denom == 0:
                return 0.0, 0.0
            field_mask = tile_mask & np.isin(data, CDL_3PHASE_CODES)
            num = field_mask.sum()
            pct = float(num / denom * 100.0)
            if num == 0:
                return pct, 0.0
            labeled = connected_components(field_mask)
            _, counts = np.unique(labeled[labeled > 0], return_counts=True)
            mean_size = float(counts.mean()) * pixel_area if len(counts) > 0 else 0.0
            return pct, mean_size
        except Exception:
            return 0.0, 0.0

    with rasterio.open(cdl_path) as src:
        tiles_proj = tiles.to_crs(src.crs)
        pcts, means = [], []
        total = len(tiles_proj)
        for i, geom in enumerate(tiles_proj.geometry):
            p, m = _process(src, geom)
            pcts.append(p)
            means.append(m)
            if (i + 1) % 500 == 0:
                print(f"      {i+1}/{total}")

    tiles["pct_3phase_crops"] = pcts
    tiles["mean_field_size_m2"] = means
    return tiles


def compute_lanid(tiles: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    print("   LANID")
    tiles = tiles.copy()
    tiles["center_pivot_irrigation"] = 0
    lanid_path = paths.get("lanid")
    if not lanid_path or not Path(lanid_path).exists():
        return tiles

    with rasterio.open(lanid_path) as src:
        tiles_proj = tiles.to_crs(src.crs)
        results = []
        for geom in tiles_proj.geometry:
            try:
                window = geometry_window(src, [mapping(geom)])
                data = src.read(1, window=window)
                tf = src.window_transform(window)
                mask = rasterize(
                    [(mapping(geom), 1)], out_shape=data.shape, transform=tf, fill=0
                ).astype(bool)
                if src.nodata is not None:
                    mask &= (data != src.nodata)
                results.append(1 if (mask & (data > 0)).any() else 0)
            except Exception:
                results.append(0)

    tiles["center_pivot_irrigation"] = results
    return tiles


# ==========================================
#       FEATURE: SIMPLE VECTOR AREAS
# ==========================================

def _vector_area(tiles: gpd.GeoDataFrame, path, col: str) -> gpd.GeoDataFrame:
    tiles = tiles.copy()
    data = load_geo(path, METRIC_CRS)
    if data is None or data.empty:
        tiles[col] = 0.0
        return tiles
    sindex = data.sindex
    areas = []
    for geom in tiles.geometry:
        cands = list(sindex.intersection(geom.bounds))
        if not cands:
            areas.append(0.0)
            continue
        local = data.iloc[cands]
        local = local[local.intersects(geom)]
        areas.append(float(local.geometry.intersection(geom).area.sum()))
    tiles[col] = areas
    return tiles


# ==========================================
#         FEATURE: FEMA BUILDINGS
# ==========================================

def compute_fema_buildings(tiles: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    print("   FEMA buildings")
    tiles = tiles.copy()
    for col in ["bldg_area_per_km2", "bldg_median_m2", "bldg_std_m2"]:
        tiles[col] = 0.0

    path = paths.get("fema_buildings")
    bldgs = load_geo(path, METRIC_CRS)
    if bldgs is None or bldgs.empty:
        return tiles

    bldgs["area_m2"] = bldgs.geometry.area
    sindex = bldgs.sindex
    tile_area_km2 = (TILE_SIZE ** 2) / 1e6

    per_km2, medians, stds = [], [], []
    for geom in tiles.geometry:
        cands = list(sindex.intersection(geom.bounds))
        if not cands:
            per_km2.append(0.0); medians.append(0.0); stds.append(0.0)
            continue
        local = bldgs.iloc[cands]
        local = local[local.intersects(geom)]
        if local.empty:
            per_km2.append(0.0); medians.append(0.0); stds.append(0.0)
            continue
        clipped_sum = float(local.geometry.intersection(geom).area.sum())
        sizes = local["area_m2"]
        per_km2.append(clipped_sum / tile_area_km2)
        medians.append(float(sizes.median()))
        stds.append(float(sizes.std()) if len(sizes) > 1 else 0.0)

    tiles["bldg_area_per_km2"] = per_km2
    tiles["bldg_median_m2"] = medians
    tiles["bldg_std_m2"] = stds
    return tiles


# ==========================================
#        COMPUTE ALL FEATURES
# ==========================================

def compute_features(tiles: gpd.GeoDataFrame, state: str, is_training: bool) -> gpd.GeoDataFrame:
    paths = get_state_paths(state)
    print(f"\n  [{state}] Computing features for {len(tiles)} tiles...")

    tiles = compute_infra(tiles, paths, is_training)
    tiles = compute_roads(tiles, paths)
    tiles = compute_nlcd(tiles, paths)
    tiles = compute_terrain(tiles, paths)
    tiles = compute_cdl(tiles, paths)
    tiles = compute_lanid(tiles, paths)

    print("   wetlands")
    tiles = _vector_area(tiles, paths.get("wetlands"), "wetlands_area_m2")
    print("   VRM")
    tiles = _vector_area(tiles, paths.get("vrm"), "vrm_area_m2")

    tiles = compute_fema_buildings(tiles, paths)
    return tiles


def _assign_state(tiles: gpd.GeoDataFrame, states: list[str]) -> gpd.GeoDataFrame:
    """Assign each tile to a state using centroid intersection."""
    tiles = tiles.copy()
    tiles["state"] = None
    centroids = tiles.geometry.centroid
    for state in states:
        border = get_state_border_geom(state)
        mask = centroids.intersects(border)
        tiles.loc[mask & tiles["state"].isna(), "state"] = state
    # fallback: nearest state for any unassigned tiles
    tiles["state"] = tiles["state"].fillna(states[0])
    return tiles


# ==========================================
#                  MAIN
# ==========================================

def main():
    project_dir = BASE_DIR / PROJECT
    out_dir = project_dir / f"{TILE_SIZE}_tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- PREDICTION ----
    print("\n" + "=" * 60)
    print("PREDICTION TILES")
    print("=" * 60)

    pred_tiles = generate_prediction_tiles(project_dir, PREDICTION_STATES)

    if len(PREDICTION_STATES) == 1:
        pred_tiles["state"] = PREDICTION_STATES[0]
        pred_with_features = compute_features(pred_tiles, PREDICTION_STATES[0], is_training=False)
    else:
        pred_tiles = _assign_state(pred_tiles, PREDICTION_STATES)
        parts = []
        for state in PREDICTION_STATES:
            subset = pred_tiles[pred_tiles["state"] == state].copy()
            if not subset.empty:
                parts.append(compute_features(subset, state, is_training=False))
        pred_with_features = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), crs=METRIC_CRS
        )

    out_pred = out_dir / "prediction_tiles.gpkg"
    pred_with_features.to_crs(FINAL_CRS).to_file(out_pred, driver="GPKG")
    print(f"\n[DONE] Prediction tiles → {out_pred}  ({len(pred_with_features)} tiles)")

    # ---- TRAINING ----
    print("\n" + "=" * 60)
    print("TRAINING TILES")
    print("=" * 60)

    train_tiles = generate_training_tiles(TRAINING_STATES)

    parts = []
    for state in TRAINING_STATES:
        subset = train_tiles[train_tiles["state"] == state].copy()
        if not subset.empty:
            parts.append(compute_features(subset, state, is_training=True))

    train_with_features = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=METRIC_CRS
    )

    out_train = out_dir / "training_tiles.gpkg"
    train_with_features.to_crs(FINAL_CRS).to_file(out_train, driver="GPKG")
    print(f"\n[DONE] Training tiles → {out_train}  ({len(train_with_features)} tiles)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prediction and training tiles.")
    parser.add_argument("--project",     default=PROJECT,          help="Project folder name (default: %(default)s)")
    parser.add_argument("--predict",     nargs="+", default=PREDICTION_STATES, metavar="STATE", help="State(s) for prediction tiles")
    parser.add_argument("--train",       nargs="+", default=TRAINING_STATES,   metavar="STATE", help="State(s) for training tiles")
    parser.add_argument("--tile-size",   type=int,  default=TILE_SIZE,         help="Tile size in metres (default: %(default)s)")
    args = parser.parse_args()

    PROJECT           = args.project
    PREDICTION_STATES = args.predict
    TRAINING_STATES   = args.train
    TILE_SIZE         = args.tile_size

    main()