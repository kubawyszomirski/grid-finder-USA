#!/usr/bin/env python3
"""
generate-edges-paths.py
Network path-analysis pipeline for a single state within a project.

Pipeline:
  1. Load {state}_edges_raw.gpkg  — the raw network edges
  2. Clip edges to the project's prediction_extent.gpkg
  3. Infer nodes from edge endpoints (or use 'u'/'v' columns if present)
     and build a NetworkX graph
  4. Snap cluster polygon centroids to network nodes  →  source nodes
  5. Snap substation points to network nodes          →  target nodes
  6. Pass 1 — standard Dijkstra: every cluster → nearest N substations
                                               → nearest M other clusters
  7. Pass 2 — same, but edges that cross anti-cluster zones are removed
  8. Aggregate per-edge path-usage statistics
  9. Add 'in_cluster' binary  (≥50 % of edge length inside any cluster polygon)
 10. Save {state}_edges_clean.gpkg to the project folder

Reads:
  raw_data/<S>/TRANSPORT/<S>_edges_raw.gpkg
  <project>/prediction_extent.gpkg
  <project>/clusters/<S>_cluster_polygons.geojson     major clusters
  <project>/clusters/<S>_cluster_polygons_minor.geojson  minor (optional)
  <project>/clusters/<S>_anti_cluster_polygons.geojson   optional
  raw_data/<S>/SUBSTATIONS/<S>_substations_final.*   optional

Writes:
  <project>/edges/{state}_edges_clean.gpkg

Usage:
  python generate-edges-paths.py Massachusetts
  python generate-edges-paths.py MA
  python generate-edges-paths.py MA --project project-ma
"""
from __future__ import annotations

import argparse
import heapq
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, shape
from shapely.strtree import STRtree

from generate_utils import (
    METRIC_CRS, FINAL_CRS, BASE_DIR, STATE_FULL_NAMES, _NAME_TO_ABBREV,
    CDL_3PHASE_CODES, _find, load_geo, _raster_pct,
)

warnings.filterwarnings("ignore")


# ==========================================
#               CONFIGURATION
# ==========================================

PROJECT = "project-ma"

# ---- Pathfinding ----
N_NEAREST_SUBS_P1     = 2
N_NEAREST_CLUSTERS_P1 = 15
N_NEAREST_SUBS_P2     = 2
N_NEAREST_CLUSTERS_P2 = 10

# ---- Node snapping ----
SNAP_TOL_M         = 1.0   # metres — endpoint tolerance for topology inference
K_NODES_PER_CLUSTER = 3    # network nodes sampled per cluster polygon

# ---- In-cluster flag ----
CLUSTER_OVERLAP_MIN = 0.50  # fraction of edge length inside cluster → in_cluster = 1

# ---- Edge feature computation ----
_EDGE_BUF_M      = 250   # buffer radius for building / land cover / raster features
_PARCEL_BUF_M    = 20    # buffer for parcel value features
_PARCEL_COLS     = ["imp_val", "lan_val", "tot_val", "tax_amt"]

# Direct distance features: (output_col, path_key)
_DIST_DIRECT = [
    ("dist_gas_and_hydro",      "gas_and_hydro"),
    ("dist_oil_gas_chemical",   "oil_gas_chemical"),
    ("dist_mining",             "mining"),
    ("dist_frs_primary",        "frs_primary"),
    ("dist_works",              "works"),
    ("dist_railway",            "railway"),
    ("dist_substations",        "substations"),
    ("dist_dams",               "dams"),
    ("dist_transmission_lines", "transmission_lines"),
]

# Merged distance features: (output_col, [path_keys]) → min distance across group
_DIST_MERGED = [
    ("dist_ren_power",    ["solar", "wind", "bess_charging_stations"]),
    ("dist_all_industry", ["utilities", "industry", "frs_secondary", "agriculture"]),
    ("dist_towers",       ["towers_major", "towers_minor"]),
    ("dist_public",       ["hospitality", "big_buildings_and_clusters", "public_infra"]),
]


# ==========================================
#            PATH RESOLUTION
# ==========================================

def get_state_paths(s: str) -> dict:
    raw = BASE_DIR / "raw_data" / s
    t   = raw / "TRANSPORT"
    return {
        "edges_raw":   t / f"{s}_edges_raw.gpkg",
        # ---- Generators ----
        "solar":                      _find(raw / "GENERATORS",  f"{s}_solar_merged"),
        "wind":                       _find(raw / "GENERATORS",  f"{s}_wind_merged"),
        "gas_and_hydro":              _find(raw / "GENERATORS",  f"{s}_eia_gas_hydro"),
        "bess_charging_stations":     _find(raw / "GENERATORS",  f"{s}_bess_charging_stations"),
        # ---- Industry ----
        "oil_gas_chemical":           _find(raw / "INDUSTRY",    f"{s}_oil_gas_chemical"),
        "industry":                   _find(raw / "INDUSTRY",    f"{s}_industry"),
        "works":                      _find(raw / "INDUSTRY",    f"{s}_works"),
        # ---- Other infra ----
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
        "railway":                    _find(t,                   f"{s}_rail_network"),
        "substations":                _find(raw / "SUBSTATIONS", f"{s}_substations_final"),
        "dams":                       _find(t,                   f"{s}_dams_major"),
        "transmission_lines":         _find(raw / "GRID",        f"{s}_transmission_lines"),
        # ---- Linear overlaps ----
        "tunnels":                    _find(t,                   f"{s}_tunnels_major"),
        "bridges":                    _find(t,                   f"{s}_bridges_lines"),
        "scenic_byways":              _find(raw / "LAND",        f"{s}_scenic_byways"),
        # ---- Buildings & land ----
        "fema_buildings":             _find(raw / "BUILDINGS",   f"{s}_fema_buildings"),
        "wetlands":                   _find(raw / "LAND",        f"{s}_wetlands"),
        "vrm":                        _find(raw / "LAND",        f"{s}_vrm"),
        "parcels":                    _find(raw / "LAND",        f"{s}_parcels"),
        # ---- Rasters ----
        "cdl":   raw / "LAND" / f"{s}_cdl.tif",
        "dem":   raw / "LAND" / f"{s}_dem.tif",
        "lanid": raw / "LAND" / f"{s}_lanid.tif",
        # ---- DSO service territories ----
        "dso_boundaries": _find(raw / "LAND", f"{s}_dso_boundaries"),
    }


def get_project_paths(s: str, project_dir: Path) -> dict:
    """Cluster outputs are project-specific (depend on prediction_extent)."""
    clu = project_dir / "clusters"
    return {
        "clusters":       clu / f"{s}_cluster_polygons.geojson",
        "clusters_minor": clu / f"{s}_cluster_polygons_minor.geojson",
        "anti_clusters":  clu / f"{s}_anti_cluster_polygons.geojson",
    }


# ==========================================
#                HELPERS
# ==========================================

def _shannon_entropy(classes: list) -> float:
    if not classes:
        return 0.0
    counts = Counter(classes)
    total  = len(classes)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _entropy_and_mode(terminals: set[int], class_map: dict) -> tuple[float, str | None]:
    classes = [class_map.get(cid, "unknown") for cid in terminals]
    return (
        round(_shannon_entropy(classes), 4),
        Counter(classes).most_common(1)[0][0] if classes else None,
    )


# ==========================================
#        EDGE FEATURE COMPUTATION
# ==========================================

def _snap_to_points(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace polygon / line geometries with centroids for distance queries."""
    data = data[["geometry"]].copy().reset_index(drop=True)
    non_point = ~data.geometry.geom_type.isin(["Point", "MultiPoint"])
    if non_point.any():
        data.loc[non_point, "geometry"] = data.loc[non_point, "geometry"].centroid
    return data


def _line_curvature(line) -> float:
    """Total turning angle (radians) per metre of line length."""
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


def _overlap_pct(edges: gpd.GeoDataFrame, data: gpd.GeoDataFrame | None,
                 buffer_m: float = 5.0) -> np.ndarray:
    """% of each edge's length that overlaps a buffer around features."""
    if data is None or data.empty:
        return np.zeros(len(edges))
    buf     = data.geometry.buffer(buffer_m).unary_union
    lengths = edges.geometry.length.replace(0, np.nan)
    overlap = edges.geometry.intersection(buf).length
    return (overlap / lengths).fillna(0).values * 100.0


def _perp_slope(edges: gpd.GeoDataFrame, dem_path) -> np.ndarray:
    """Average terrain slope (degrees) perpendicular to each edge."""
    STEP_M = 30.0
    PERP_M = 50.0
    if not dem_path or not Path(dem_path).exists():
        return np.full(len(edges), np.nan)
    with rasterio.open(dem_path) as src:
        transformer = pyproj.Transformer.from_crs(METRIC_CRS, src.crs, always_xy=True)
        scores = []
        for line in edges.geometry:
            if line is None or line.length < STEP_M:
                scores.append(np.nan)
                continue
            left_pts, right_pts = [], []
            for d in np.arange(0, line.length, STEP_M):
                pt  = line.interpolate(d)
                pt2 = line.interpolate(min(d + 1.0, line.length))
                dx, dy = pt2.x - pt.x, pt2.y - pt.y
                norm = np.sqrt(dx**2 + dy**2)
                if norm < 1e-9:
                    continue
                px = -dy / norm * PERP_M
                py =  dx / norm * PERP_M
                left_pts.append( (pt.x + px, pt.y + py))
                right_pts.append((pt.x - px, pt.y - py))
            if not left_pts:
                scores.append(np.nan)
                continue
            xs, ys   = zip(*(left_pts + right_pts))
            lon, lat = transformer.transform(list(xs), list(ys))
            try:
                elevs = np.array(list(src.sample(zip(lon, lat), indexes=1)),
                                 dtype=np.float32).flatten()
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


def compute_edge_features(edges: gpd.GeoDataFrame, paths: dict) -> gpd.GeoDataFrame:
    """
    Assign road-style ML features to every edge geometry.
    Works identically to generate-roads feature logic but on edge LineStrings.
    """
    print("\n[feat] Computing edge features...")
    edges = edges.copy()

    centroids = gpd.GeoDataFrame(geometry=edges.geometry.centroid, crs=METRIC_CRS, index=edges.index)

    # ------------------------------------------------------------------
    # Distances — direct
    # ------------------------------------------------------------------
    print("  [feat] distances (direct)")
    for col, key in _DIST_DIRECT:
        data = load_geo(paths.get(key), METRIC_CRS)
        if data is None or data.empty:
            edges[col] = np.nan
            continue
        pts    = _snap_to_points(data)
        joined = gpd.sjoin_nearest(centroids, pts, how="left", distance_col=col)
        joined = joined[~joined.index.duplicated(keep="first")]
        edges[col] = joined[col].reindex(edges.index)

    # ------------------------------------------------------------------
    # Distances — merged (min across group)
    # ------------------------------------------------------------------
    print("  [feat] distances (merged)")
    for col, keys in _DIST_MERGED:
        tmp_cols = []
        for key in keys:
            data = load_geo(paths.get(key), METRIC_CRS)
            if data is None or data.empty:
                continue
            pts    = _snap_to_points(data)
            tcol   = f"_tmp_{key}"
            joined = gpd.sjoin_nearest(centroids, pts, how="left", distance_col=tcol)
            joined = joined[~joined.index.duplicated(keep="first")]
            edges[tcol] = joined[tcol].reindex(edges.index)
            tmp_cols.append(tcol)
        if tmp_cols:
            edges[col] = edges[tmp_cols].min(axis=1)
            edges.drop(columns=tmp_cols, inplace=True)
        else:
            edges[col] = np.nan

    # ------------------------------------------------------------------
    # Building stats (250 m buffer)
    # ------------------------------------------------------------------
    print("  [feat] building stats")
    for c in ["bldg_total_area_250m", "bldg_median_m2_250m"]:
        edges[c] = 0.0
    bldgs = load_geo(paths.get("fema_buildings"), METRIC_CRS)
    if bldgs is not None and not bldgs.empty:
        bldgs = bldgs.copy()
        bldgs["_area"] = bldgs.geometry.area
        sindex  = bldgs.sindex
        buffers = edges.geometry.buffer(_EDGE_BUF_M)
        total_areas, medians = [], []
        for buf in buffers:
            cands = list(sindex.query(buf, predicate="intersects"))
            if not cands:
                total_areas.append(0.0); medians.append(0.0)
                continue
            areas = bldgs.iloc[cands]["_area"]
            total_areas.append(float(areas.sum()))
            medians.append(float(areas.median()))
        edges["bldg_total_area_250m"] = total_areas
        edges["bldg_median_m2_250m"]  = medians

    # ------------------------------------------------------------------
    # CDL 3-phase % (250 m buffer)
    # ------------------------------------------------------------------
    print("  [feat] CDL raster")
    edges["cdl_3phase_pct_250m"] = 0.0
    cdl_path = paths.get("cdl")
    if cdl_path and Path(cdl_path).exists():
        bufs = edges.geometry.buffer(_EDGE_BUF_M)
        with rasterio.open(cdl_path) as src:
            bufs_repr = bufs.to_crs(src.crs) if edges.crs.to_epsg() != src.crs.to_epsg() else bufs
            edges["cdl_3phase_pct_250m"] = [_raster_pct(src, g, CDL_3PHASE_CODES) for g in bufs_repr]

    # ------------------------------------------------------------------
    # Wetlands / VRM % (250 m buffer)
    # ------------------------------------------------------------------
    print("  [feat] wetlands / VRM")
    buffers = edges.geometry.buffer(_EDGE_BUF_M)
    for col, key in [("wetlands_pct_250m", "wetlands"), ("vrm_pct_250m", "vrm")]:
        data = load_geo(paths.get(key), METRIC_CRS)
        if data is None or data.empty:
            edges[col] = 0.0
            continue
        sindex = data.sindex
        pcts   = []
        for buf in buffers:
            cands = list(sindex.query(buf, predicate="intersects"))
            if not cands:
                pcts.append(0.0)
                continue
            overlap = data.iloc[cands].geometry.intersection(buf).area.sum()
            pcts.append(float(overlap / buf.area * 100.0))
        edges[col] = pcts

    # ------------------------------------------------------------------
    # LANID distance
    # ------------------------------------------------------------------
    print("  [feat] LANID distance")
    edges["dist_lanid"] = np.nan
    lanid_path = paths.get("lanid")
    if lanid_path and Path(lanid_path).exists():
        with rasterio.open(lanid_path) as src:
            data = src.read(1)
            mask = data > 0
            if mask.any():
                pts_lanid = [shape(g).centroid
                             for g, _ in rio_shapes(data, mask=mask, transform=src.transform)]
                lanid_pts = gpd.GeoDataFrame(geometry=pts_lanid, crs=src.crs).to_crs(METRIC_CRS)
                joined    = gpd.sjoin_nearest(centroids, lanid_pts[["geometry"]],
                                              how="left", distance_col="dist_lanid")
                joined    = joined[~joined.index.duplicated(keep="first")]
                edges["dist_lanid"] = joined["dist_lanid"].reindex(edges.index)

    # ------------------------------------------------------------------
    # Linear overlaps, curvature, slope
    # ------------------------------------------------------------------
    print("  [feat] linear overlaps / curvature / slope")
    edges["tunnel_pct"]      = _overlap_pct(edges, load_geo(paths.get("tunnels"),      METRIC_CRS), 5.0)
    edges["scenic_byway_pct"]= _overlap_pct(edges, load_geo(paths.get("scenic_byways"),METRIC_CRS), 5.0)

    edges["has_bridge"] = 0
    bridges = load_geo(paths.get("bridges"), METRIC_CRS)
    if bridges is not None and not bridges.empty:
        bridges = bridges[bridges.geometry.length >= 50].copy()
        if not bridges.empty:
            bridge_union = bridges.geometry.buffer(1).unary_union
            edges["has_bridge"] = edges.geometry.buffer(5).intersects(bridge_union).astype(int)

    edges["curvature_per_m"] = [_line_curvature(g) for g in edges.geometry]
    edges["perp_slope_deg"]  = _perp_slope(edges, paths.get("dem"))

    # ------------------------------------------------------------------
    # Parcel values (20 m buffer, area-weighted mean)
    # ------------------------------------------------------------------
    print("  [feat] parcel values")
    out_cols = [f"parcel_{c}" for c in _PARCEL_COLS]
    for c in out_cols:
        edges[c] = np.nan
    parcels = load_geo(paths.get("parcels"), METRIC_CRS)
    if parcels is not None and not parcels.empty:
        value_cols = [c for c in _PARCEL_COLS if c in parcels.columns]
        for c in value_cols:
            parcels[c] = pd.to_numeric(parcels[c], errors="coerce")
        sindex  = parcels.sindex
        buffers = edges.geometry.buffer(_PARCEL_BUF_M)
        results = {c: [] for c in value_cols}
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
            edges[f"parcel_{c}"] = results[c]

    # ------------------------------------------------------------------
    # DSO boundary crossings
    # ------------------------------------------------------------------
    print("  [feat] DSO boundary crossings")
    edges["crosses_dso_boundary"] = 0
    dso = load_geo(paths.get("dso_boundaries"), METRIC_CRS)
    if dso is not None and not dso.empty:
        dso = dso[["geometry"]].reset_index(drop=True).copy()
        dso["_dso_id"] = dso.index.astype(float)

        # One GeoDataFrame of all unique edge endpoints (starts then ends)
        starts = [Point(g.coords[0])  for g in edges.geometry]
        ends   = [Point(g.coords[-1]) for g in edges.geometry]
        n      = len(edges)
        pts_gdf = gpd.GeoDataFrame(
            geometry=starts + ends, crs=METRIC_CRS,
        ).reset_index(drop=True)

        # Assign DSO ID to each point; unmatched (outside all polygons) → NaN
        joined = gpd.sjoin(pts_gdf, dso, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
        dso_ids = joined["_dso_id"].reindex(range(len(pts_gdf))).values

        start_dso = pd.array(dso_ids[:n], dtype="Float64")
        end_dso   = pd.array(dso_ids[n:], dtype="Float64")

        crosses = []
        for s, e in zip(start_dso, end_dso):
            s_na, e_na = pd.isna(s), pd.isna(e)
            if s_na and e_na:
                crosses.append(0)       # both outside all DSOs
            elif s_na or e_na:
                crosses.append(1)       # one inside, one outside
            else:
                crosses.append(int(s != e))   # different DSOs
        edges["crosses_dso_boundary"] = crosses
        n_cross = sum(crosses)
        print(f"    {n_cross:,} edges cross a DSO boundary")

    print(f"  [feat] done — {len(edges):,} edges")
    return edges


# ==========================================
#          STEP 2: GRAPH BUILDING
# ==========================================

def build_graph(
    edges_gdf: gpd.GeoDataFrame,
) -> tuple[
    object,                          # nx.Graph  (G)
    dict[int, Point],                # node_geom_by_id
    dict[int, int],                  # edge_row_idx → node_u
    dict[int, int],                  # edge_row_idx → node_v
    np.ndarray,                      # node_ids  (sorted)
    np.ndarray,                      # node_xy   (N × 2)
    dict[tuple[int, int], list[int]], # canonical_edge → [original row indices]
]:
    """
    Build a NetworkX graph from a GeoDataFrame of LineString edges.

    If the GeoDataFrame has 'u' and 'v' integer columns (OSMnx-style),
    they are used directly.  Otherwise topology is inferred by snapping
    endpoint coordinates within SNAP_TOL_M.
    """
    has_uv = "u" in edges_gdf.columns and "v" in edges_gdf.columns

    node_geom_by_id: dict[int, Point] = {}
    row_to_u: dict[int, int] = {}
    row_to_v: dict[int, int] = {}

    if has_uv:
        for idx, row in edges_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            u, v = int(row["u"]), int(row["v"])
            node_geom_by_id.setdefault(u, Point(coords[0]))
            node_geom_by_id.setdefault(v, Point(coords[-1]))
            row_to_u[idx] = u
            row_to_v[idx] = v
    else:
        coord_to_nid: dict[tuple[int, int], int] = {}
        next_id = 0
        factor  = 1.0 / SNAP_TOL_M

        for idx, row in edges_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            for raw_xy, role in [(coords[0], "u"), (coords[-1], "v")]:
                key = (int(round(raw_xy[0] * factor)), int(round(raw_xy[1] * factor)))
                if key not in coord_to_nid:
                    coord_to_nid[key] = next_id
                    node_geom_by_id[next_id] = Point(raw_xy)
                    next_id += 1
                nid = coord_to_nid[key]
                if role == "u":
                    row_to_u[idx] = nid
                else:
                    row_to_v[idx] = nid

    # Build graph
    G = nx.Graph()
    for nid, pt in node_geom_by_id.items():
        G.add_node(nid, x=float(pt.x), y=float(pt.y))

    pair_to_rows: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, row in edges_gdf.iterrows():
        u = row_to_u.get(idx)
        v = row_to_v.get(idx)
        if u is None or v is None or u == v:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Edge weight: prefer stored length column, fall back to geometric length
        w = None
        for lc in ("length", "edge_length_m", "shape_length"):
            if lc in edges_gdf.columns and pd.notna(row[lc]):
                w = float(row[lc])
                break
        if w is None or w <= 0:
            w = max(float(geom.length), 1.0)

        e = (min(u, v), max(u, v))
        pair_to_rows[e].append(idx)
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=w)

    node_ids = np.array(sorted(node_geom_by_id.keys()), dtype=np.int64)
    node_xy  = np.array(
        [(node_geom_by_id[n].x, node_geom_by_id[n].y) for n in node_ids],
        dtype=np.float64,
    )
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, node_geom_by_id, row_to_u, row_to_v, node_ids, node_xy, dict(pair_to_rows)


# ==========================================
#    STEP 3: CLUSTER NODES
# ==========================================

def build_cluster_node_map(
    clusters_gdf: gpd.GeoDataFrame,
    node_ids:     np.ndarray,
    node_xy:      np.ndarray,
    G,
) -> dict[int, list[int]]:
    """
    For each cluster polygon, sample up to K_NODES_PER_CLUSTER representative
    network nodes (centroid + boundary samples, spread-selected).

    Returns {cluster_id: [node_id, ...]}
    """
    tree   = cKDTree(node_xy)
    id_col = next(
        (c for c in ("cluster_id", "anti_cluster_id", "id") if c in clusters_gdf.columns),
        None,
    )
    result: dict[int, list[int]] = {}

    for idx, row in clusters_gdf.iterrows():
        cid  = int(row[id_col]) if id_col is not None else int(idx)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Sample candidate points: centroid + evenly-spaced boundary points
        candidate_pts: list[Point] = [geom.centroid, geom.representative_point()]
        try:
            boundary = geom.boundary
            for frac in np.linspace(0, 1, K_NODES_PER_CLUSTER * 6, endpoint=False):
                candidate_pts.append(boundary.interpolate(frac, normalized=True))
        except Exception:
            pass

        chosen: list[int] = []
        seen:   set[int]  = set()
        for pt in candidate_pts:
            _, idxs = tree.query([pt.x, pt.y], k=min(5, len(node_ids)))
            for i in np.atleast_1d(idxs):
                nid = int(node_ids[i])
                if nid in G and nid not in seen:
                    # Enforce minimum spread: 50 m from already-chosen nodes
                    xy  = node_xy[i]
                    too_close = any(
                        np.linalg.norm(xy - node_xy[np.searchsorted(node_ids, c)]) < 50.0
                        for c in chosen
                    )
                    if not too_close:
                        chosen.append(nid)
                        seen.add(nid)
                if len(chosen) >= K_NODES_PER_CLUSTER:
                    break
            if len(chosen) >= K_NODES_PER_CLUSTER:
                break

        # Fallback: just take the k nearest if spread selection found nothing
        if not chosen:
            _, idxs = tree.query(
                [geom.centroid.x, geom.centroid.y],
                k=min(K_NODES_PER_CLUSTER, len(node_ids)),
            )
            for i in np.atleast_1d(idxs):
                nid = int(node_ids[i])
                if nid in G:
                    chosen.append(nid)
                if len(chosen) >= K_NODES_PER_CLUSTER:
                    break

        if chosen:
            result[cid] = chosen

    return result


# ==========================================
#    STEP 4 / 6: GRAPH CONSTRUCTION
# ==========================================

def build_anti_cluster_graph(G_base, anti_clusters: gpd.GeoDataFrame | None,
                               node_geom_by_id: dict) -> object:
    """Return a copy of G_base with edges crossing anti-cluster zones removed."""
    G2 = G_base.copy()
    if anti_clusters is None or anti_clusters.empty:
        return G2

    ac_geoms = list(anti_clusters.geometry.values)
    ac_tree  = STRtree(ac_geoms)
    removed  = 0

    for u, v in list(G2.edges()):
        pu = node_geom_by_id.get(u)
        pv = node_geom_by_id.get(v)
        if pu is None or pv is None:
            continue
        edge_line = LineString([(pu.x, pu.y), (pv.x, pv.y)])
        hits = ac_tree.query(edge_line)
        for ac in hits:
            ac_geom = ac_geoms[ac] if isinstance(ac, (int, np.integer)) else ac
            if edge_line.intersects(ac_geom):
                G2.remove_edge(u, v)
                removed += 1
                break

    print(f"  Pass-2 graph: {removed:,} edges removed  "
          f"({G2.number_of_edges():,} remaining)")
    return G2


# ==========================================
#    STEP 5 / 6: PATHFINDING
# ==========================================

def _dijkstra_n_targets(
    G,
    source:      int,
    targets_set: set[int],
    n:           int,
) -> list[tuple[int, float, list[int]]]:
    """Dijkstra returning paths to the n nearest members of targets_set."""
    if n <= 0:
        return []
    queue = [(0.0, int(source))]
    dists: dict[int, float] = {int(source): 0.0}
    pred:  dict[int, int | None] = {int(source): None}
    found:   list[tuple[int, float, list[int]]] = []
    visited: set[int] = set()

    while queue and len(found) < n:
        dist, u = heapq.heappop(queue)
        if dist > dists.get(u, float("inf")):
            continue
        if u in targets_set and u not in visited:
            path, cur = [], u
            while cur is not None:
                path.append(cur)
                cur = pred.get(cur)
            path.reverse()
            found.append((u, dist, path))
            visited.add(u)
            if len(found) == n:
                break
        for v, edata in G[u].items():
            v   = int(v)
            alt = dist + float(edata.get("weight", 1.0))
            if alt < dists.get(v, float("inf")):
                dists[v] = alt
                pred[v]  = u
                heapq.heappush(queue, (alt, v))

    return found


def run_pathfinding_pass(
    G,
    cluster_node_map: dict[int, list[int]],
    sub_nodes_set:    set[int],
    class_map:        dict[int, str],
    n_subs:           int,
    n_clusters:       int,
    all_edge_pairs:   set[tuple[int, int]],
) -> dict[tuple[int, int], dict]:
    """
    For every cluster, find n_subs nearest substations and n_clusters nearest
    other clusters.  Returns per-edge usage statistics.
    """
    usage: dict[tuple[int, int], dict] = {
        e: {"sub_path_count": 0, "tt_path_count": 0, "terminals": set()}
        for e in all_edge_pairs
    }

    for cid, nodes_list in cluster_node_map.items():
        valid = [n for n in nodes_list if n in G]
        if not valid:
            continue
        src = valid[0]

        # ---- To substations ----
        if sub_nodes_set:
            for _, _, path in _dijkstra_n_targets(G, src, sub_nodes_set, n_subs):
                for a, b in zip(path[:-1], path[1:]):
                    e = (min(a, b), max(a, b))
                    if e in usage:
                        usage[e]["sub_path_count"] += 1
                        usage[e]["terminals"].add(cid)

        # ---- To other clusters ----
        other_node_to_cids: dict[int, list[int]] = {}
        for other_cid, other_nodes in cluster_node_map.items():
            if other_cid == cid:
                continue
            tgt = [n for n in other_nodes if n in G]
            if tgt:
                other_node_to_cids.setdefault(tgt[0], []).append(other_cid)

        for tgt_node, _, path in _dijkstra_n_targets(
            G, src, set(other_node_to_cids.keys()), n_clusters
        ):
            target_cids = other_node_to_cids.get(tgt_node, [])
            for a, b in zip(path[:-1], path[1:]):
                e = (min(a, b), max(a, b))
                if e in usage:
                    usage[e]["tt_path_count"] += 1
                    usage[e]["terminals"].update([cid] + target_cids)

    return usage


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge/path pipeline — graph from edges_raw, "
                    "cluster pathfinding, outputs edges_clean to project folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate-edges-paths.py Massachusetts\n"
            "  python generate-edges-paths.py MA\n"
            "  python generate-edges-paths.py MA --project project-ma\n"
        ),
    )
    parser.add_argument("state", help="Full state name or 2-letter abbreviation")
    parser.add_argument("--project", default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    args = parser.parse_args()

    abbrev = _NAME_TO_ABBREV.get(args.state.strip())
    if abbrev is None:
        print(f"[ERROR] Unknown state: '{args.state}'")
        print(f"  Known: {', '.join(sorted(STATE_FULL_NAMES.keys()))}")
        return

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    paths = {**get_state_paths(abbrev), **get_project_paths(abbrev, project_dir)}

    print(f"\n{'='*60}")
    print(f" Edge/Path Pipeline — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f" Project: {args.project}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load edges_raw
    # ------------------------------------------------------------------
    print("\n[1] Loading edges_raw...")
    if not paths["edges_raw"].exists():
        print(f"[ERROR] Not found: {paths['edges_raw']}")
        return
    edges = gpd.read_file(paths["edges_raw"]).to_crs(METRIC_CRS)
    print(f"  {len(edges):,} edges loaded")

    # ------------------------------------------------------------------
    # 1b. Clip to prediction extent
    # ------------------------------------------------------------------
    extent_path = project_dir / "prediction_extent.gpkg"
    if extent_path.exists():
        print(f"\n[1b] Clipping to prediction extent...")
        extent = gpd.read_file(extent_path).to_crs(METRIC_CRS)
        extent_union = extent.geometry.union_all()
        edges = edges[edges.geometry.intersects(extent_union)].copy()
        edges = gpd.clip(edges, extent_union)
        print(f"  {len(edges):,} edges after clipping")
    else:
        print(f"\n[1b] No prediction_extent.gpkg in {project_dir} — using all edges")

    # ------------------------------------------------------------------
    # 2. Build network graph
    # ------------------------------------------------------------------
    print("\n[2] Building network graph...")
    G, node_geom_by_id, row_to_u, row_to_v, node_ids, node_xy, pair_to_rows = \
        build_graph(edges)

    all_edge_pairs: set[tuple[int, int]] = set(pair_to_rows.keys())

    # ------------------------------------------------------------------
    # 3. Load cluster polygons
    # ------------------------------------------------------------------
    print("\n[3] Loading cluster polygons...")
    major  = load_geo(paths["clusters"],       METRIC_CRS)
    minor  = load_geo(paths["clusters_minor"], METRIC_CRS)

    if major is None or major.empty:
        print("[ERROR] No major cluster polygons found — aborting.")
        return

    n_minor = len(minor) if minor is not None else 0
    print(f"  {len(major)} major + {n_minor} minor clusters")

    # Union of all clusters for in_cluster computation
    all_clusters = pd.concat(
        [df for df in [major, minor] if df is not None and not df.empty],
        ignore_index=True,
    )
    cluster_union = all_clusters.geometry.union_all()

    # Cluster class map for entropy
    id_col    = next((c for c in ("cluster_id", "id") if c in major.columns), None)
    class_col = next((c for c in ("dominant_class", "class", "type") if c in major.columns), None)
    class_map: dict[int, str] = {}
    if id_col and class_col:
        for _, row in major.iterrows():
            class_map[int(row[id_col])] = str(row[class_col])

    # ------------------------------------------------------------------
    # 4. Snap cluster centroids → network nodes
    # ------------------------------------------------------------------
    print("\n[4] Mapping cluster polygons to network nodes...")
    cluster_node_map = build_cluster_node_map(major, node_ids, node_xy, G)
    print(f"  {len(cluster_node_map)} clusters mapped")

    # ------------------------------------------------------------------
    # 5. Snap substations → network nodes
    # ------------------------------------------------------------------
    print("\n[5] Loading substations...")
    subs = load_geo(paths["substations"], METRIC_CRS)
    sub_nodes_set: set[int] = set()
    if subs is not None and not subs.empty:
        tree = cKDTree(node_xy)
        sub_xy = np.c_[subs.geometry.centroid.x.values,
                        subs.geometry.centroid.y.values]
        _, idxs = tree.query(sub_xy, k=1)
        sub_nodes_set = {int(node_ids[i]) for i in idxs if int(node_ids[i]) in G}
        print(f"  {len(sub_nodes_set)} substation nodes")
    else:
        print("  No substations found — cluster-to-cluster paths only")

    # ------------------------------------------------------------------
    # 6. Build Pass-2 graph (anti-cluster avoidance)
    # ------------------------------------------------------------------
    print("\n[6] Building anti-cluster graph (Pass 2)...")
    anti = load_geo(paths["anti_clusters"], METRIC_CRS)
    G2   = build_anti_cluster_graph(G, anti, node_geom_by_id)

    # ------------------------------------------------------------------
    # 7. Pathfinding Pass 1 — standard
    # ------------------------------------------------------------------
    print("\n[7] Pathfinding — Pass 1 (standard)...")
    usage_p1 = run_pathfinding_pass(
        G, cluster_node_map, sub_nodes_set, class_map,
        N_NEAREST_SUBS_P1, N_NEAREST_CLUSTERS_P1, all_edge_pairs,
    )
    n_used_p1 = sum(
        1 for u in usage_p1.values()
        if u["sub_path_count"] + u["tt_path_count"] > 0
    )
    print(f"  {n_used_p1:,} edges used")

    # ------------------------------------------------------------------
    # 8. Pathfinding Pass 2 — avoids anti-clusters
    # ------------------------------------------------------------------
    print("\n[8] Pathfinding — Pass 2 (anti-cluster avoidance)...")
    usage_p2 = run_pathfinding_pass(
        G2, cluster_node_map, sub_nodes_set, class_map,
        N_NEAREST_SUBS_P2, N_NEAREST_CLUSTERS_P2, all_edge_pairs,
    )
    n_used_p2 = sum(
        1 for u in usage_p2.values()
        if u["sub_path_count"] + u["tt_path_count"] > 0
    )
    print(f"  {n_used_p2:,} edges used")

    # ------------------------------------------------------------------
    # 9. Assign stats + in_cluster to edges
    # ------------------------------------------------------------------
    print("\n[9] Assigning path stats to edges...")

    n = len(edges)
    new_cols: dict[str, list] = {
        "p1_sub_path_count":   [0] * n,
        "p1_tt_path_count":    [0] * n,
        "p1_unique_terminals": [0] * n,
        "p1_usage_entropy":    [0.0] * n,
        "p1_usage_mode":       [None] * n,
        "p2_sub_path_count":   [0] * n,
        "p2_tt_path_count":    [0] * n,
        "p2_unique_terminals": [0] * n,
        "p2_usage_entropy":    [0.0] * n,
        "p2_usage_mode":       [None] * n,
        "is_used_by_any_path": [0] * n,
        "in_cluster":          [0] * n,
    }

    for i, (row_idx, row) in enumerate(edges.iterrows()):
        u = row_to_u.get(row_idx)
        v = row_to_v.get(row_idx)
        if u is None or v is None:
            continue
        e = (min(u, v), max(u, v))

        s1 = usage_p1.get(e)
        s2 = usage_p2.get(e)

        if s1 is not None:
            ent1, mode1 = _entropy_and_mode(s1["terminals"], class_map)
            new_cols["p1_sub_path_count"][i]   = int(s1["sub_path_count"])
            new_cols["p1_tt_path_count"][i]    = int(s1["tt_path_count"])
            new_cols["p1_unique_terminals"][i] = int(len(s1["terminals"]))
            new_cols["p1_usage_entropy"][i]    = float(ent1)
            new_cols["p1_usage_mode"][i]       = mode1

        if s2 is not None:
            ent2, mode2 = _entropy_and_mode(s2["terminals"], class_map)
            new_cols["p2_sub_path_count"][i]   = int(s2["sub_path_count"])
            new_cols["p2_tt_path_count"][i]    = int(s2["tt_path_count"])
            new_cols["p2_unique_terminals"][i] = int(len(s2["terminals"]))
            new_cols["p2_usage_entropy"][i]    = float(ent2)
            new_cols["p2_usage_mode"][i]       = mode2

        if s1 is not None and s2 is not None:
            total = (s1["sub_path_count"] + s1["tt_path_count"]
                     + s2["sub_path_count"] + s2["tt_path_count"])
            new_cols["is_used_by_any_path"][i] = int(total > 0)

        # in_cluster: ≥ CLUSTER_OVERLAP_MIN of edge length inside cluster union
        geom = row.geometry
        if geom is not None and not geom.is_empty and cluster_union is not None:
            try:
                overlap = geom.intersection(cluster_union).length
                new_cols["in_cluster"][i] = int(
                    overlap / max(geom.length, 1e-9) >= CLUSTER_OVERLAP_MIN
                )
            except Exception:
                pass

    edges_out = edges.copy()
    for col, vals in new_cols.items():
        edges_out[col] = vals

    # ------------------------------------------------------------------
    # 10. Assign road-style features
    # ------------------------------------------------------------------
    edges_out = compute_edge_features(edges_out, paths)

    # ------------------------------------------------------------------
    # 11. Save
    # ------------------------------------------------------------------
    print("\n[11] Saving edges_clean...")
    out_dir = project_dir / "edges"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{abbrev}_edges_clean.gpkg"
    edges_out.to_crs(FINAL_CRS).to_file(out, driver="GPKG")

    used_any   = sum(new_cols["is_used_by_any_path"])
    in_cluster = sum(new_cols["in_cluster"])
    print(f"  {len(edges_out):,} edges  →  {out.name}")
    print(f"  {used_any:,} used by at least one path")
    print(f"  {in_cluster:,} lie within a cluster (≥{int(CLUSTER_OVERLAP_MIN*100)}% overlap)")

    print(f"\n{'='*60}")
    print(f" DONE — {STATE_FULL_NAMES[abbrev]} ({abbrev})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()