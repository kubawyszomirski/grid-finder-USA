#!/usr/bin/env python3
"""
Road network processing pipeline.

Produces two complete sets of road network outputs for a given US state:

  Set 1 — "raw":   OSM grid integrated, roads stitched/split,
                   exclusion zones applied, artificial paths injected.

  Set 2 — "clean": Same as raw, plus dead-end stump removal,
                   parking/cemetery cuts, and isolated cluster pruning.

Both sets output three GeoPackage layers: roads, edges, nodes.
Roads carry road_id; edges carry edge_id + road_id + start_node/end_node;
nodes carry node_id + connected topology. Both sets share the same column schema.

Usage:
    python roads.py Massachusetts
    python roads.py "New York"
"""
from __future__ import annotations

import argparse
import itertools
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring
from tqdm import tqdm

from fetch_data.fetch_utils import STATE_ABBREV, STATE_CRS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
OVERLAP_BUFFER     = 12      # m  — grid/road proximity for overlap detection
OVERLAP_THRESHOLD  = 0.80    # 80 % grid-line overlap → treat as road, drop grid line
MAX_EXTEND_DIST    = 300     # m  — max endpoint snap for isolated grid lines
MAX_STITCH_LEN     = 3500    # m  — maximum merged segment length
SPLIT_THRESHOLD    = 5000    # m  — split roads longer than this
MIN_CLUSTER_SIZE   = 100     # segments — drop isolated clusters smaller than this
STUMP_LEN_PASS1    = 250     # m  — first stump-removal pass threshold (clean only)
STUMP_LEN_PASS2    = 200     # m  — second stump-removal pass threshold (clean only)
ART_NODE_AREA_MIN  = 5_000   # m² — exclusion polygon must exceed this to get nodes
ART_NODE_DENSITY   = 300_000 # m² per artificial node
ART_EDGE_MAX_LEN   = 2_000   # m  — hard cap on internal artificial edges
ART_BRIDGE_MAX_LEN = 1_000   # m  — max bridge from real node to artificial node

DATA_ROOT = Path(__file__).parent / "raw_data"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _paths(abbrev: str) -> dict[str, Path]:
    t = DATA_ROOT / abbrev / "TRANSPORT"
    p = DATA_ROOT / abbrev / "PUBLIC"
    g = DATA_ROOT / abbrev / "GRID"
    e = DATA_ROOT / abbrev / "EXCLUSIONS"
    return {
        "roads_parquet":   t / f"{abbrev}_roads.parquet",
        "grid":            g / f"{abbrev}_osm_distribution_lines.geojson",
        "exclusions":      e / f"{abbrev}_exclusions.gpkg",
        "parkings":        t / f"{abbrev}_parkings.geojson",
        "cemeteries":      p / f"{abbrev}_cemeteries.geojson",
        "out_roads_raw":   t / f"{abbrev}_roads_raw.gpkg",
        "out_edges_raw":   t / f"{abbrev}_edges_raw.gpkg",
        "out_nodes_raw":   t / f"{abbrev}_nodes_raw.gpkg",
        "out_roads_clean": t / f"{abbrev}_roads_clean.gpkg",
        "out_edges_clean": t / f"{abbrev}_edges_clean.gpkg",
        "out_nodes_clean": t / f"{abbrev}_nodes_clean.gpkg",
    }


# ---------------------------------------------------------------------------
# Step 1 — Load and prepare raw roads
# ---------------------------------------------------------------------------

def load_roads(parquet_path: Path, local_crs: str) -> gpd.GeoDataFrame:
    """
    Load roads parquet, drop mostly-empty columns, remove polygons/non-lines,
    normalise highway classes, and project to the local metric CRS.
    """
    log.info("Loading roads parquet…")
    gdf = gpd.read_parquet(parquet_path)
    log.info(f"  {len(gdf):,} rows, {gdf.shape[1]} columns")

    # Drop columns with <10 % valid values
    min_valid = max(1, int(len(gdf) * 0.10))
    gdf = gdf.dropna(axis=1, thresh=min_valid)

    # Remove polygons early — explode first, then keep only LineStrings
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.type == "LineString"].copy()
    log.info(f"  After polygon removal: {len(gdf):,} LineStrings, {gdf.shape[1]} columns")

    if "highway" in gdf.columns:
        gdf = gdf[~gdf["highway"].isin(["motorway", "motorway_link"])].copy()
        gdf["highway"] = (
            gdf["highway"]
            .fillna("service")
            .replace("", "service")
            .replace({
                "trunk":        "primary",
                "trunk_link":   "primary",
                "primary_link": "primary",
                "unclassified": "service",
            })
        )
    else:
        gdf["highway"] = "service"

    gdf["is_grid"] = "no"
    gdf = gdf.to_crs(local_crs)
    return gdf


# ---------------------------------------------------------------------------
# Step 2 — Integrate OSM grid
# ---------------------------------------------------------------------------

def _extend_line_to_road(line: LineString, roads: gpd.GeoDataFrame, sindex, max_dist: float) -> LineString:
    """Extend each endpoint of *line* along its direction until it snaps to a road."""
    if not isinstance(line, LineString) or len(line.coords) < 2:
        return line

    coords = list(line.coords)
    new_coords = coords.copy()

    for is_start in (True, False):
        if is_start:
            p1, p2 = np.array(coords[1]), np.array(coords[0])
        else:
            p1, p2 = np.array(coords[-2]), np.array(coords[-1])

        vec = p2 - p1
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue

        ray = LineString([p2, p2 + (vec / norm) * max_dist])
        hits = list(sindex.intersection(ray.bounds))
        if not hits:
            continue

        intersections = roads.iloc[hits].intersection(ray)
        valid = intersections[~intersections.is_empty]
        if valid.empty:
            continue

        closest_pt, min_d = None, float("inf")
        for geom in valid:
            pts = list(geom.geoms) if geom.geom_type == "MultiPoint" else ([geom] if geom.geom_type == "Point" else [])
            for pt in pts:
                d = Point(p2).distance(pt)
                if d < min_d:
                    min_d, closest_pt = d, pt

        if closest_pt is not None:
            new_coords[0 if is_start else -1] = (closest_pt.x, closest_pt.y)

    return LineString(new_coords)


def integrate_osm_grid(roads: gpd.GeoDataFrame, grid_path: Path) -> gpd.GeoDataFrame:
    """
    Merge OSM distribution grid lines into the road network.
    Grid lines that heavily overlap existing roads are dropped (their roads are
    tagged is_grid='yes'). The remaining isolated grid lines are endpoint-snapped
    to the road network before merging.
    """
    log.info("Loading OSM grid…")
    grid = gpd.read_file(grid_path).to_crs(roads.crs)
    grid = grid.explode(index_parts=False).reset_index(drop=True)
    grid = grid[grid.geometry.type == "LineString"].copy()
    grid["highway"] = "osm_grid"
    grid["is_grid"]  = "yes"
    log.info(f"  {len(grid):,} grid lines")

    sindex = roads.sindex
    drop_idx = []

    for idx, row in grid.iterrows():
        line = row.geometry
        candidates_iloc = list(sindex.intersection(line.bounds))
        if not candidates_iloc:
            continue
        nearby = roads.iloc[candidates_iloc]
        nearby = nearby[nearby.intersects(line.buffer(OVERLAP_BUFFER))]
        if nearby.empty:
            continue

        road_buf_union = nearby.geometry.buffer(OVERLAP_BUFFER).unary_union
        if line.intersection(road_buf_union).length / line.length >= OVERLAP_THRESHOLD:
            for ridx in nearby.index:
                if roads.at[ridx, "geometry"].distance(line) <= OVERLAP_BUFFER:
                    roads.at[ridx, "is_grid"] = "yes"
            drop_idx.append(idx)

    isolated = grid.drop(index=drop_idx).copy()
    log.info(f"  Removed {len(drop_idx):,} overlapping lines. {len(isolated):,} isolated lines remain.")

    isolated["geometry"] = isolated["geometry"].apply(
        lambda g: _extend_line_to_road(g, roads, sindex, MAX_EXTEND_DIST)
    )

    keep = {"geometry", "highway", "is_grid"}
    r_cols = [c for c in roads.columns   if c in keep]
    g_cols = [c for c in isolated.columns if c in keep]

    merged = pd.concat([roads[r_cols], isolated[g_cols]], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs=roads.crs)
    log.info(f"  Merged: {len(merged):,} total features")
    return merged


# ---------------------------------------------------------------------------
# Step 3 — Apply exclusion zones
# ---------------------------------------------------------------------------

def _build_exclusion_union(exclusions: gpd.GeoDataFrame):
    """Return a clean, unioned exclusion polygon geometry."""
    exc = exclusions.copy()
    exc["geometry"] = exc.geometry.make_valid().buffer(0)
    exc = exc[~exc.geometry.is_empty]
    return exc.geometry.union_all() if hasattr(exc.geometry, "union_all") else exc.geometry.unary_union


def apply_exclusion_zones(roads: gpd.GeoDataFrame, exclusion_union) -> gpd.GeoDataFrame:
    """
    Difference roads against exclusion polygons.
    Roads with >50 % overlap are dropped; the rest are trimmed.
    """
    exc_gdf = gpd.GeoDataFrame(geometry=[exclusion_union], crs=roads.crs)
    hit = gpd.sjoin(roads, exc_gdf, how="inner", predicate="intersects").index.unique()

    safe   = roads[~roads.index.isin(hit)].copy()
    to_cut = roads[ roads.index.isin(hit)].copy()
    log.info(f"  {len(to_cut):,} roads interact with exclusions")

    to_cut["_orig"] = to_cut.geometry.length
    to_cut["geometry"] = to_cut["geometry"].difference(exclusion_union)
    to_cut["_new"]  = to_cut.geometry.length
    to_cut["_pct"]  = (to_cut["_orig"] - to_cut["_new"]) / to_cut["_orig"].clip(lower=1e-9)

    dropped = (to_cut["_pct"] > 0.50).sum()
    log.info(f"  Dropping {dropped:,} roads (>50 % inside exclusions)")

    to_cut = to_cut[(to_cut["_pct"] <= 0.50) & ~to_cut.geometry.is_empty]
    to_cut = to_cut.drop(columns=["_orig", "_new", "_pct"])

    result = gpd.GeoDataFrame(pd.concat([safe, to_cut], ignore_index=True), crs=roads.crs)
    log.info(f"  After exclusions: {len(result):,} roads")
    return result


# ---------------------------------------------------------------------------
# Cleaning helpers (Set 2 only)
# ---------------------------------------------------------------------------

def _remove_stumps(roads: gpd.GeoDataFrame, max_len: float, highway_types: list[str]) -> gpd.GeoDataFrame:
    """
    Remove short dead-end stubs: service/residential roads shorter than *max_len*
    that do not connect at both endpoints.
    """
    candidates = roads[roads["highway"].isin(highway_types) & (roads.geometry.length < max_len)].copy()
    if candidates.empty:
        return roads

    endpoints = candidates.geometry.boundary
    ep_gdf = gpd.GeoDataFrame(geometry=endpoints, crs=roads.crs).explode(index_parts=False)
    ep_gdf["ep_id"]     = range(len(ep_gdf))
    ep_gdf["parent_id"] = ep_gdf.index
    ep_gdf["geometry"]  = ep_gdf.geometry.buffer(0.1)

    conn = gpd.sjoin(ep_gdf, roads, how="inner", predicate="intersects")
    valid_conn = conn[conn["parent_id"] != conn["index_right"]]
    connected_eps = valid_conn["ep_id"].unique()

    ep_gdf["is_connected"] = ep_gdf["ep_id"].isin(connected_eps)
    connectivity = ep_gdf.groupby("parent_id")["is_connected"].all()
    stump_ids = connectivity[~connectivity].index

    result = roads[~roads.index.isin(stump_ids)].reset_index(drop=True)
    log.info(f"  Removed {len(stump_ids):,} stubs")
    return result


def _cut_polygon_features(
    roads: gpd.GeoDataFrame,
    poly_feats: gpd.GeoDataFrame,
    polygon_union,
    highway_types: list[str],
) -> gpd.GeoDataFrame:
    """
    Cut service/residential roads against parking/cemetery polygons (50 % rule).
    *poly_feats* is the individual-polygon GeoDataFrame used for efficient spatial
    index filtering; *polygon_union* is the pre-built union used for the actual
    difference operation.
    Non-matching highway types are left untouched.
    """
    if polygon_union is None or (hasattr(polygon_union, "is_empty") and polygon_union.is_empty):
        return roads

    # Filter with individual polygons so the spatial index is effective
    hit_idx = gpd.sjoin(roads, poly_feats[["geometry"]], how="inner", predicate="intersects").index.unique()

    target = roads.index.isin(hit_idx) & roads["highway"].isin(highway_types)
    safe   = roads[~target].copy()
    to_cut = roads[ target].copy()
    log.info(f"  {target.sum():,} service/residential segments to evaluate")

    if len(to_cut) > 0:
        to_cut["_orig"] = to_cut.geometry.length
        to_cut["geometry"] = to_cut["geometry"].difference(polygon_union)
        to_cut["_new"]  = to_cut.geometry.length
        to_cut["_pct"]  = (to_cut["_orig"] - to_cut["_new"]) / to_cut["_orig"].clip(lower=1e-9)

        dropped = (to_cut["_pct"] > 0.50).sum()
        log.info(f"  Dropping {dropped:,} segments (>50 % inside polygon features)")

        to_cut = to_cut[(to_cut["_pct"] <= 0.50) & ~to_cut.geometry.is_empty]
        to_cut = to_cut.drop(columns=["_orig", "_new", "_pct"])

    return gpd.GeoDataFrame(pd.concat([safe, to_cut], ignore_index=True), crs=roads.crs)


def prune_network(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Three-pass clean:
      1. Explode multi-part geometries
      2. Second stump removal (< STUMP_LEN_PASS2)
      3. Drop isolated clusters smaller than MIN_CLUSTER_SIZE
    """
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geometry.type == "LineString"].copy()
    log.info(f"  After explode: {len(roads):,} segments")

    roads = _remove_stumps(roads, STUMP_LEN_PASS2, ["service", "residential"])

    buf = gpd.GeoDataFrame(geometry=roads.geometry.buffer(0.1), crs=roads.crs)
    hits = gpd.sjoin(buf, roads, how="inner", predicate="intersects")
    G = nx.Graph()
    G.add_edges_from(zip(hits.index, hits["index_right"]))

    clusters = list(nx.connected_components(G))
    keep_idx = [i for c in clusters if len(c) >= MIN_CLUSTER_SIZE for i in c]
    dropped  = sum(1 for c in clusters if len(c) < MIN_CLUSTER_SIZE)

    roads = roads.loc[keep_idx].reset_index(drop=True)
    log.info(f"  Dropped {dropped:,} small clusters. Network: {len(roads):,} segments")
    return roads


# ---------------------------------------------------------------------------
# Step 4 — Stitch and split
# ---------------------------------------------------------------------------

def stitch_and_split(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge contiguous same-type/same-name degree-2 segments (max MAX_STITCH_LEN m,
    unlimited for osm_grid), then split any road longer than SPLIT_THRESHOLD m.
    """
    def names_match(n1, n2) -> bool:
        return (pd.isna(n1) and pd.isna(n2)) or str(n1) == str(n2)

    pt_to_roads: dict[tuple, list] = defaultdict(list)
    for row in roads.itertuples():
        g = row.geometry
        if g is None or g.is_empty or g.geom_type != "LineString":
            continue
        c = list(g.coords)
        pt_to_roads[c[0]].append(row.Index)
        pt_to_roads[c[-1]].append(row.Index)

    G = nx.Graph()
    for pt, ids in pt_to_roads.items():
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        r1, r2 = ids
        hw1, hw2 = roads.at[r1, "highway"], roads.at[r2, "highway"]
        if hw1 != hw2:
            continue
        n1 = roads.at[r1, "name"] if "name" in roads.columns else None
        n2 = roads.at[r2, "name"] if "name" in roads.columns else None
        if hw1 == "osm_grid" or names_match(n1, n2):
            G.add_edge(r1, r2)

    drop_set: set = set()
    new_rows: list = []

    for comp in nx.connected_components(G):
        sg    = G.subgraph(comp)
        start = next((n for n, d in sg.degree() if d == 1), list(comp)[0])
        path  = list(nx.dfs_preorder_nodes(sg, start))
        is_grid = roads.at[path[0], "highway"] == "osm_grid"

        chunk: list = []
        chunk_len = 0.0

        for r_idx in path:
            seg_len = roads.at[r_idx, "geometry"].length
            if not is_grid and chunk_len + seg_len > MAX_STITCH_LEN and chunk:
                if len(chunk) > 1:
                    drop_set.update(chunk)
                    merged = linemerge(MultiLineString([roads.at[i, "geometry"] for i in chunk]))
                    row = roads.loc[chunk[0]].copy()
                    row["geometry"] = merged
                    new_rows.append(row)
                chunk = [r_idx]
                chunk_len = seg_len
            else:
                chunk.append(r_idx)
                chunk_len += seg_len

        if len(chunk) > 1:
            drop_set.update(chunk)
            merged = linemerge(MultiLineString([roads.at[i, "geometry"] for i in chunk]))
            row = roads.loc[chunk[0]].copy()
            row["geometry"] = merged
            new_rows.append(row)

    log.info(f"  Stitched {len(drop_set):,} fragments → {len(new_rows):,} merged lines")
    final = roads.drop(index=list(drop_set)).copy()
    if new_rows:
        final = gpd.GeoDataFrame(
            pd.concat([final, gpd.GeoDataFrame(new_rows, crs=roads.crs)], ignore_index=True),
            crs=roads.crs,
        )

    final["_len"] = final.geometry.length
    long_mask = (
        (final["_len"] > SPLIT_THRESHOLD)
        & (final.geom_type == "LineString")
        & (final["highway"] != "osm_grid")
    )
    long_roads = final[long_mask].copy()
    safe       = final[~long_mask].copy()

    if len(long_roads) > 0:
        all_endpoints: set = set()
        for g in final.geometry:
            if g and not g.is_empty and g.geom_type == "LineString":
                c = list(g.coords)
                all_endpoints.add(c[0])
                all_endpoints.add(c[-1])

        split_rows = []
        for _, row in long_roads.iterrows():
            line  = row.geometry
            total = line.length
            junc_dists = sorted(line.project(Point(c)) for c in set(line.coords) & all_endpoints)
            cur = 0.0
            while total - cur > MAX_STITCH_LEN:
                target = cur + MAX_STITCH_LEN
                valid_j = [d for d in junc_dists if cur + 500 < d <= target]
                split_at = max(valid_j) if valid_j else target
                frag = row.copy()
                frag["geometry"] = substring(line, cur, split_at)
                split_rows.append(frag)
                cur = split_at
            if cur < total:
                frag = row.copy()
                frag["geometry"] = substring(line, cur, total)
                split_rows.append(frag)

        log.info(f"  Split {len(long_roads):,} long roads → {len(split_rows):,} fragments")
        final = gpd.GeoDataFrame(
            pd.concat([safe, gpd.GeoDataFrame(split_rows, crs=roads.crs)], ignore_index=True),
            crs=roads.crs,
        )

    final = final.drop(columns=["_len"])
    log.info(f"  After stitch+split: {len(final):,} segments")
    return final


# ---------------------------------------------------------------------------
# Step 5 — Assign road IDs
# ---------------------------------------------------------------------------

def assign_road_ids(roads: gpd.GeoDataFrame, prefix: str = "R") -> gpd.GeoDataFrame:
    """Assign sequential road_id (R_000000 …) as the first column."""
    roads = roads.copy().reset_index(drop=True)
    roads.insert(0, "road_id", [f"{prefix}_{i:06d}" for i in range(len(roads))])
    return roads


# ---------------------------------------------------------------------------
# Step 6 — Build topology (edges + nodes)
# ---------------------------------------------------------------------------

def _heal_network(edges: gpd.GeoDataFrame, protected_coords: set) -> gpd.GeoDataFrame:
    """Zip collinear degree-2 segments that share a non-protected node."""
    rounded_protected = {(round(x, 3), round(y, 3)) for x, y in protected_coords}
    pt_to_roads: dict[tuple, list] = defaultdict(list)

    for idx, row in edges.iterrows():
        coords = list(row.geometry.coords)
        pt_to_roads[(round(coords[0][0], 3), round(coords[0][1], 3))].append(idx)
        pt_to_roads[(round(coords[-1][0], 3), round(coords[-1][1], 3))].append(idx)

    G = nx.Graph()
    G.add_nodes_from(edges.index)
    for pt, ids in pt_to_roads.items():
        if len(ids) != 2 or ids[0] == ids[1] or pt in rounded_protected:
            continue
        r1, r2 = ids
        if "highway" not in edges.columns or edges.at[r1, "highway"] == edges.at[r2, "highway"]:
            G.add_edge(r1, r2)

    rows = []
    for comp in nx.connected_components(G):
        comp = list(comp)
        if len(comp) == 1:
            rows.append(edges.loc[comp[0]])
        else:
            merged = linemerge(MultiLineString([edges.at[i, "geometry"] for i in comp]))
            base = edges.loc[comp[0]].copy()
            if "is_grid" in edges.columns:
                base["is_grid"] = "yes" if any(edges.at[i, "is_grid"] == "yes" for i in comp) else "no"
            if merged.geom_type == "MultiLineString":
                for sub in merged.geoms:
                    r = base.copy(); r["geometry"] = sub; rows.append(r)
            else:
                base["geometry"] = merged; rows.append(base)

    return gpd.GeoDataFrame(rows, crs=edges.crs).reset_index(drop=True)


def build_topology(
    roads: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Planarize roads at every intersection (with forced fractures at grid
    connection points), heal degree-2 segments, then assign edge_id/node_id.

    Returns (roads_out, edges, nodes) in the same local CRS as the input.
    *roads_out* is the full input (including grid) with road_id preserved.
    *edges* are the planarized segments with cross-referenced topology.
    """
    roads = roads.copy()
    roads["geometry"] = roads.geometry.make_valid()
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geometry.type == "LineString"].copy()

    # Separate grid lines — they are handled logically, not planarized
    if "highway" in roads.columns:
        grid_mask = roads["highway"] == "osm_grid"
    else:
        grid_mask = pd.Series(False, index=roads.index)

    grid_gdf = roads[grid_mask].copy()
    road_gdf  = roads[~grid_mask].copy()

    # Map grid endpoints to the nearest road-network points
    road_union = road_gdf.unary_union
    grid_edge_rows: list = []
    protected_coords: set = set()

    for _, row in grid_gdf.iterrows():
        line = row.geometry
        intersection = line.intersection(road_union)

        points: list = []
        geom_list = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
        for geom in geom_list:
            if geom.is_empty:
                continue
            if geom.geom_type == "Point":
                points.append(geom)
            elif geom.geom_type == "MultiPoint":
                points.extend(geom.geoms)
            elif geom.geom_type == "LineString" and len(geom.coords) >= 2:
                points += [Point(geom.coords[0]), Point(geom.coords[-1])]
            elif geom.geom_type == "MultiLineString":
                for sub in geom.geoms:
                    if len(sub.coords) >= 2:
                        points += [Point(sub.coords[0]), Point(sub.coords[-1])]

        sp = Point(line.coords[0])
        ep = Point(line.coords[-1])

        if not points:
            start_coord = (sp.x, sp.y)
            end_coord   = (ep.x, ep.y)
        else:
            dist_map = {line.project(pt): pt for pt in points}
            d_min, d_max = min(dist_map), max(dist_map)

            if len(dist_map) == 1 or abs(d_max - d_min) < 0.1:
                pt    = dist_map[d_min]
                coord = (pt.x, pt.y)
                if d_min < line.length / 2.0:
                    start_coord, end_coord = coord, (ep.x, ep.y)
                else:
                    start_coord, end_coord = (sp.x, sp.y), coord
                protected_coords.add(coord)
            else:
                start_coord = (dist_map[d_min].x, dist_map[d_min].y)
                end_coord   = (dist_map[d_max].x, dist_map[d_max].y)
                protected_coords.add(start_coord)
                protected_coords.add(end_coord)

        nr = row.copy()
        nr["logical_start"] = start_coord
        nr["logical_end"]   = end_coord
        grid_edge_rows.append(nr)

    grid_mapped = (
        gpd.GeoDataFrame(grid_edge_rows, crs=road_gdf.crs)
        if grid_edge_rows
        else gpd.GeoDataFrame(columns=list(road_gdf.columns) + ["logical_start", "logical_end"], crs=road_gdf.crs)
    )

    # Planarize with 2mm micro-cutters at grid connection points
    cutter_lines = []
    for px, py in protected_coords:
        cutter_lines.append(LineString([(px - 0.001, py), (px + 0.001, py)]))
        cutter_lines.append(LineString([(px, py - 0.001), (px, py + 0.001)]))

    noded = road_gdf.geometry.union_all() if hasattr(road_gdf.geometry, "union_all") else road_gdf.geometry.unary_union
    noded_series = gpd.GeoSeries([noded] + cutter_lines, crs=road_gdf.crs)
    noded = noded_series.union_all() if hasattr(noded_series, "union_all") else noded_series.unary_union

    edge_geoms = [g for g in (noded.geoms if hasattr(noded, "geoms") else [noded]) if g.geom_type == "LineString"]
    exploded = gpd.GeoDataFrame(geometry=edge_geoms, crs=road_gdf.crs)
    exploded = exploded[exploded.geometry.length > 0.005].copy()

    # Recover attributes via midpoint nearest join
    exploded["_mid"] = exploded.geometry.interpolate(0.5, normalized=True)
    mids = gpd.GeoDataFrame(geometry=exploded["_mid"], crs=road_gdf.crs)
    keep_cols = [c for c in ["highway", "name", "is_grid", "road_id"] if c in road_gdf.columns]
    joined = gpd.sjoin_nearest(mids, road_gdf[keep_cols + ["geometry"]], how="left")
    joined = joined[~joined.index.duplicated(keep="first")]
    for col in keep_cols:
        exploded[col] = joined[col]
    exploded = exploded.drop(columns=["_mid"])

    # Two healing passes
    edges = _heal_network(exploded, protected_coords)
    edges = _heal_network(edges, protected_coords)

    # Combine road edges + grid edges
    edges["logical_start"] = None
    edges["logical_end"]   = None
    all_edges = pd.concat([edges, grid_mapped], ignore_index=True)

    # ------------------------------------------------------------------
    # Coordinate-based node IDs — deterministic across raw and clean sets:
    #   same intersection point → same node_id regardless of which branch.
    # Edge IDs are derived from the sorted node-pair so the same physical
    # edge gets the same edge_id in both sets.
    # ------------------------------------------------------------------
    def _coord_node_id(pt: tuple) -> str:
        """1 m precision coordinate key → deterministic node ID."""
        return f"N_{int(round(pt[0]))}_{int(round(pt[1]))}"

    node_to_coord: dict[str, tuple] = {}
    node_to_edges: dict[str, list]  = defaultdict(list)
    node_to_nbrs:  dict[str, set]   = defaultdict(set)
    all_edges["start_node"] = None
    all_edges["end_node"]   = None

    pair_counter: dict[tuple, int] = defaultdict(int)
    edge_ids: list[str] = []

    for idx, row in all_edges.iterrows():
        if pd.notna(row.get("logical_start")):
            u_coord, v_coord = row["logical_start"], row["logical_end"]
        else:
            c = list(row.geometry.coords)
            u_coord, v_coord = c[0], c[-1]

        u_id = _coord_node_id(u_coord)
        v_id = _coord_node_id(v_coord)

        # Register node coordinates (first encounter wins)
        node_to_coord.setdefault(u_id, u_coord)
        node_to_coord.setdefault(v_id, v_coord)

        # Deterministic edge ID from sorted node pair; suffix _pN for parallels
        pair_key = tuple(sorted((u_id, v_id)))
        n = pair_counter[pair_key]
        pair_counter[pair_key] += 1
        eid = f"E_{pair_key[0]}_{pair_key[1]}" + (f"_p{n}" if n > 0 else "")
        edge_ids.append(eid)

        all_edges.at[idx, "start_node"] = u_id
        all_edges.at[idx, "end_node"]   = v_id
        node_to_edges[u_id].append(eid)
        node_to_edges[v_id].append(eid)
        if u_id != v_id:
            node_to_nbrs[u_id].add(v_id)
            node_to_nbrs[v_id].add(u_id)

    all_edges["edge_id"] = edge_ids

    nodes_data = [
        {
            "node_id":         nid,
            "connected_edges": ", ".join(node_to_edges[nid]),
            "connected_nodes": ", ".join(sorted(node_to_nbrs[nid])),
            "geometry":        Point(coord[0], coord[1]),
        }
        for nid, coord in node_to_coord.items()
    ]
    nodes = gpd.GeoDataFrame(nodes_data, crs=all_edges.crs)

    # Final column order for edges
    edge_cols = ["edge_id", "start_node", "end_node", "geometry"]
    for col in ["highway", "name", "is_grid", "road_id"]:
        if col in all_edges.columns:
            edge_cols.append(col)
    all_edges = all_edges[edge_cols]

    log.info(f"  Topology: {len(all_edges):,} edges, {len(nodes):,} nodes")
    return roads, all_edges, nodes


# ---------------------------------------------------------------------------
# Step 6b — Deduplicate parallel edges (clean set only)
# ---------------------------------------------------------------------------

# Highway class quality ranking — lower number = higher priority (keep this one)
_HIGHWAY_PRIORITY: dict[str, int] = {
    "primary":           0,
    "secondary":         1,
    "tertiary":          2,
    "residential":       3,
    "service":           4,
    "osm_grid":          5,
    "artificial":        6,
    "artificial_bridge": 7,
}


def deduplicate_parallel_edges(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    For every pair of edges that share the same unordered {start_node, end_node}
    (i.e. A→B and B→A count as the same pair), keep only the one with the
    highest highway class and drop the rest.  Node connectivity tables are
    rebuilt from the surviving edges.
    """
    edges = edges.copy()

    # Canonical (sorted) node pair for each edge
    edges["_pair"] = edges.apply(
        lambda r: tuple(sorted((r["start_node"], r["end_node"]))), axis=1
    )
    edges["_priority"] = edges["highway"].map(_HIGHWAY_PRIORITY).fillna(99).astype(int)

    # Within each pair keep the row with the lowest priority number (= best class)
    keep_mask = edges.groupby("_pair")["_priority"].transform("min") == edges["_priority"]
    # If still tied (same class), keep the first occurrence
    keep_idx = edges[keep_mask].groupby("_pair").head(1).index

    dropped = len(edges) - len(keep_idx)
    log.info(f"  Removed {dropped:,} parallel/duplicate edges")

    edges = edges.loc[keep_idx].drop(columns=["_pair", "_priority"]).reset_index(drop=True)

    # Rebuild node connectivity from the surviving edges
    node_to_edges: dict[str, list] = defaultdict(list)
    node_to_nbrs:  dict[str, set]  = defaultdict(set)
    for _, row in edges.iterrows():
        u, v, eid = row["start_node"], row["end_node"], row["edge_id"]
        node_to_edges[u].append(eid)
        node_to_edges[v].append(eid)
        if u != v:
            node_to_nbrs[u].add(v)
            node_to_nbrs[v].add(u)

    nodes = nodes.copy()
    nodes["connected_edges"] = nodes["node_id"].apply(
        lambda nid: ", ".join(sorted(node_to_edges.get(nid, [])))
    )
    nodes["connected_nodes"] = nodes["node_id"].apply(
        lambda nid: ", ".join(sorted(node_to_nbrs.get(nid, set())))
    )
    return edges, nodes


# ---------------------------------------------------------------------------
# Step 7 — Inject artificial paths through exclusion zones
# ---------------------------------------------------------------------------

def inject_artificial_paths(
    roads: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    exclusions: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Seed artificial nodes inside large exclusion polygons, connect them
    internally with a dense MST, and bridge them to the nearest real nodes
    on the boundary. Returns (roads_out, edges_out, nodes_out).
    """
    exc = exclusions.to_crs(roads.crs).copy()
    exc["geometry"] = exc.geometry.make_valid()
    exc = exc.explode(index_parts=False).reset_index(drop=True)
    exc = exc[exc.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    log.info(f"  {len(exc):,} valid exclusion polygons for seeding")

    art_nodes_data: list  = []
    art_edges_data: list  = []
    node_counter = 1
    edge_counter = 1
    poly_to_art:  dict    = defaultdict(list)
    art_geoms:    dict    = {}

    np.random.seed(42)

    for idx, row in tqdm(exc.iterrows(), total=len(exc), desc="  Seeding nodes", leave=False):
        poly = row.geometry
        if poly is None or poly.is_empty or poly.area < ART_NODE_AREA_MIN:
            continue
        n_nodes = max(1, int(poly.area / ART_NODE_DENSITY))
        minx, miny, maxx, maxy = poly.bounds
        pts: list = []
        attempts = 0
        while len(pts) < n_nodes and attempts < n_nodes * 100:
            p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if poly.contains(p):
                pts.append(p)
            attempts += 1
        for p in pts:
            nid = f"N_ART_{node_counter:06d}"
            art_nodes_data.append({"node_id": nid, "geometry": p})
            poly_to_art[idx].append(nid)
            art_geoms[nid] = p
            node_counter += 1

    if not art_nodes_data:
        log.info("  No exclusion zones large enough. Skipping artificial network.")
        return roads, edges, nodes

    log.info(f"  Generated {len(art_nodes_data):,} artificial nodes")
    art_nodes_gdf = gpd.GeoDataFrame(art_nodes_data, crs=nodes.crs)

    # Build internal networks (MST + nearest-4-neighbours density)
    for poly_idx, nids in poly_to_art.items():
        if len(nids) < 2:
            continue
        G = nx.Graph()
        for n1, n2 in itertools.combinations(nids, 2):
            G.add_edge(n1, n2, weight=art_geoms[n1].distance(art_geoms[n2]))

        dense: set = set(nx.minimum_spanning_tree(G).edges())
        for n in nids:
            for v, data in sorted(G[n].items(), key=lambda x: x[1]["weight"])[:4]:
                dense.add(tuple(sorted((n, v))))

        for u, v in dense:
            dist = art_geoms[u].distance(art_geoms[v])
            if dist > ART_EDGE_MAX_LEN:
                continue
            eid = f"E_ART_{edge_counter:06d}"
            art_edges_data.append({
                "edge_id":    eid,
                "start_node": u,
                "end_node":   v,
                "highway":    "artificial",
                "is_grid":    "no",
                "geometry":   LineString([art_geoms[u], art_geoms[v]]),
            })
            edge_counter += 1

    # Bridge real boundary nodes to the closest artificial node
    active_exc = exc.loc[list(poly_to_art.keys())].copy()
    active_exc["geometry"] = active_exc.geometry.buffer(50)
    nearby = gpd.sjoin(nodes, active_exc, how="inner", predicate="intersects")
    log.info(f"  {len(nearby):,} real nodes near exclusion boundaries")

    created_bridges: set = set()
    for _, row in nearby.iterrows():
        orig_nid   = row["node_id"]
        poly_idx   = row["index_right"]
        orig_geom  = row["geometry"]
        local_nids = poly_to_art.get(poly_idx, [])
        if not local_nids:
            continue
        best = min(local_nids, key=lambda n: orig_geom.distance(art_geoms[n]))
        dist = orig_geom.distance(art_geoms[best])
        if dist > ART_BRIDGE_MAX_LEN:
            continue
        sig = tuple(sorted((orig_nid, best)))
        if sig in created_bridges:
            continue
        created_bridges.add(sig)
        eid = f"E_ART_{edge_counter:06d}"
        art_edges_data.append({
            "edge_id":    eid,
            "start_node": orig_nid,
            "end_node":   best,
            "highway":    "artificial_bridge",
            "is_grid":    "no",
            "geometry":   LineString([orig_geom, art_geoms[best]]),
        })
        edge_counter += 1

    log.info(f"  Created {len(art_edges_data):,} artificial edges")

    # Update node connectivity tables
    node_to_edges: dict = {
        row["node_id"]: set(row["connected_edges"].split(", ")) if pd.notna(row.get("connected_edges")) and row["connected_edges"] else set()
        for _, row in nodes.iterrows()
    }
    node_to_nbrs: dict = {
        row["node_id"]: set(row["connected_nodes"].split(", ")) if pd.notna(row.get("connected_nodes")) and row["connected_nodes"] else set()
        for _, row in nodes.iterrows()
    }
    for nid in art_geoms:
        node_to_edges.setdefault(nid, set())
        node_to_nbrs.setdefault(nid, set())

    for edge in art_edges_data:
        u, v, eid = edge["start_node"], edge["end_node"], edge["edge_id"]
        node_to_edges[u].add(eid)
        node_to_edges[v].add(eid)
        node_to_nbrs[u].add(v)
        node_to_nbrs[v].add(u)

    nodes["connected_edges"] = nodes["node_id"].apply(lambda n: ", ".join(sorted(node_to_edges.get(n, set()))))
    nodes["connected_nodes"] = nodes["node_id"].apply(lambda n: ", ".join(sorted(node_to_nbrs.get(n, set()))))
    art_nodes_gdf["connected_edges"] = art_nodes_gdf["node_id"].apply(lambda n: ", ".join(sorted(node_to_edges.get(n, set()))))
    art_nodes_gdf["connected_nodes"] = art_nodes_gdf["node_id"].apply(lambda n: ", ".join(sorted(node_to_nbrs.get(n, set()))))

    art_edges_gdf = gpd.GeoDataFrame(art_edges_data, crs=edges.crs)

    # Add artificial roads to roads layer (tagged with R_ART_ ids)
    art_roads = art_edges_gdf[["highway", "is_grid", "geometry"]].copy().reset_index(drop=True)
    art_roads.insert(0, "road_id", [f"R_ART_{i:06d}" for i in range(len(art_roads))])

    final_roads = gpd.GeoDataFrame(pd.concat([roads, art_roads], ignore_index=True), crs=roads.crs)
    final_edges = gpd.GeoDataFrame(pd.concat([edges, art_edges_gdf], ignore_index=True), crs=edges.crs)
    final_nodes = gpd.GeoDataFrame(pd.concat([nodes, art_nodes_gdf], ignore_index=True), crs=nodes.crs)

    return final_roads, final_edges, final_nodes


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    roads: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    roads_path: Path,
    edges_path: Path,
    nodes_path: Path,
) -> None:
    """Project all three layers to WGS-84 and write GeoPackages."""
    roads.to_crs("EPSG:4326").to_file(roads_path, driver="GPKG")
    edges.to_crs("EPSG:4326").to_file(edges_path, driver="GPKG")
    nodes.to_crs("EPSG:4326").to_file(nodes_path, driver="GPKG")
    log.info(f"  → {roads_path.name}  ({len(roads):,} roads)")
    log.info(f"  → {edges_path.name}  ({len(edges):,} edges)")
    log.info(f"  → {nodes_path.name}  ({len(nodes):,} nodes)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_state(state_name: str) -> None:
    abbrev    = STATE_ABBREV[state_name]
    local_crs = STATE_CRS[state_name]
    paths     = _paths(abbrev)

    log.info(f"\n{'=' * 60}")
    log.info(f"State: {state_name} ({abbrev})   CRS: {local_crs}")
    log.info(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # SHARED — load, integrate grid, apply exclusions
    # ------------------------------------------------------------------
    log.info("\n[SHARED] Loading roads…")
    roads = load_roads(paths["roads_parquet"], local_crs)

    log.info("\n[SHARED] Integrating OSM grid…")
    roads = integrate_osm_grid(roads, paths["grid"])

    log.info("\n[SHARED] Loading exclusion zones…")
    exclusions_gdf = gpd.read_file(paths["exclusions"], engine="pyogrio").to_crs(local_crs)
    exclusion_union = _build_exclusion_union(exclusions_gdf)

    log.info("\n[SHARED] Applying exclusion zones…")
    roads_excl = apply_exclusion_zones(roads, exclusion_union)

    # Assign road_id ONCE on the shared base so that the same physical road
    # carries the same road_id in both the raw and clean outputs.
    log.info("\n[SHARED] Assigning road IDs…")
    roads_excl = assign_road_ids(roads_excl)

    # ------------------------------------------------------------------
    # SET 1 — raw (no cleaning or pruning)
    # ------------------------------------------------------------------
    log.info("\n" + "─" * 60)
    log.info("[SET 1 — RAW] Stitching and splitting…")
    roads_raw = stitch_and_split(roads_excl.copy())

    log.info("\n[SET 1 — RAW] Building topology…")
    roads_raw, edges_raw, nodes_raw = build_topology(roads_raw)

    log.info("\n[SET 1 — RAW] Injecting artificial paths…")
    roads_raw, edges_raw, nodes_raw = inject_artificial_paths(roads_raw, edges_raw, nodes_raw, exclusions_gdf)

    log.info("\n[SET 1 — RAW] Saving…")
    save_outputs(
        roads_raw, edges_raw, nodes_raw,
        paths["out_roads_raw"], paths["out_edges_raw"], paths["out_nodes_raw"],
    )

    # ------------------------------------------------------------------
    # SET 2 — clean (stump removal + parking/cemetery cuts + cluster pruning)
    # ------------------------------------------------------------------
    log.info("\n" + "─" * 60)
    log.info("[SET 2 — CLEAN] Removing short dead-end stumps (< %dm service/residential)…" % STUMP_LEN_PASS1)
    roads_clean = _remove_stumps(roads_excl.copy(), STUMP_LEN_PASS1, ["service", "residential"])

    log.info("\n[SET 2 — CLEAN] Loading parking/cemetery polygons…")
    parkings    = gpd.read_file(paths["parkings"],   engine="pyogrio").to_crs(local_crs)
    cemeteries  = gpd.read_file(paths["cemeteries"], engine="pyogrio").to_crs(local_crs)
    poly_feats  = gpd.GeoDataFrame(
        pd.concat([parkings[["geometry"]], cemeteries[["geometry"]]], ignore_index=True),
        crs=local_crs,
    )
    poly_feats["geometry"] = poly_feats.geometry.make_valid().buffer(0)
    poly_feats  = poly_feats[~poly_feats.geometry.is_empty]
    poly_union  = (
        poly_feats.geometry.union_all()
        if hasattr(poly_feats.geometry, "union_all")
        else poly_feats.geometry.unary_union
    )

    log.info("\n[SET 2 — CLEAN] Cutting roads inside parking/cemetery polygons…")
    roads_clean = _cut_polygon_features(roads_clean, poly_feats, poly_union, ["service", "residential"])

    log.info("\n[SET 2 — CLEAN] Pruning network (explode → stumps → cluster removal)…")
    roads_clean = prune_network(roads_clean)

    log.info("\n[SET 2 — CLEAN] Stitching and splitting…")
    roads_clean = stitch_and_split(roads_clean)

    log.info("\n[SET 2 — CLEAN] Building topology…")
    roads_clean, edges_clean, nodes_clean = build_topology(roads_clean)

    log.info("\n[SET 2 — CLEAN] Deduplicating parallel edges…")
    edges_clean, nodes_clean = deduplicate_parallel_edges(edges_clean, nodes_clean)

    log.info("\n[SET 2 — CLEAN] Injecting artificial paths…")
    roads_clean, edges_clean, nodes_clean = inject_artificial_paths(roads_clean, edges_clean, nodes_clean, exclusions_gdf)

    log.info("\n[SET 2 — CLEAN] Saving…")
    save_outputs(
        roads_clean, edges_clean, nodes_clean,
        paths["out_roads_clean"], paths["out_edges_clean"], paths["out_nodes_clean"],
    )

    log.info(f"\n{'=' * 60}")
    log.info(f"All 6 outputs written to: raw_data/{abbrev}/TRANSPORT/")
    log.info(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road network processing pipeline")
    parser.add_argument("state", help='Full US state name, e.g. "Massachusetts"')
    args = parser.parse_args()

    if args.state not in STATE_ABBREV:
        parser.error(
            f"Unknown state '{args.state}'.\n"
            f"Valid names: {', '.join(sorted(STATE_ABBREV))}"
        )

    process_state(args.state)