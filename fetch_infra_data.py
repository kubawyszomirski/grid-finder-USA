#!/usr/bin/env python3
"""
Infrastructure data fetching and processing pipeline.

Reads source datasets from the grid-research-raw-data S3 bucket (geoparquet),
clips to the requested state, and writes GeoJSON outputs to:
    raw_data/<STATE_ABBREV>/<CATEGORY>/

Usage:
    python fetch_infra_data.py Massachusetts
    python fetch_infra_data.py "New York"

Requirements:
    uv add geopandas pyarrow s3fs boto3 osmium scikit-learn
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote as url_quote

import requests

import osmium
import osmium.geom
import shapely.wkb as _shp_wkb
import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN

from fetch_utils import BaseConfig, S3Reader, GeoUtils, STATE_ABBREV, STATE_CRS

_wkb_factory = osmium.geom.WKBFactory()


def _tag_matches(tags, custom_filter: dict) -> bool:
    """Returns True if any tag in custom_filter matches the OSM object's tags."""
    for key, values in custom_filter.items():
        if key not in tags:
            continue
        if values is True:
            return True
        tag_val = tags[key]
        if isinstance(values, (list, tuple)) and tag_val in values:
            return True
        if isinstance(values, str) and tag_val == values:
            return True
    return False


class _OsmHandler(osmium.SimpleHandler):
    """Collects OSM nodes, ways, and areas matching a tag filter into GeoDataFrame rows."""

    def __init__(self, custom_filter: dict, keep_nodes: bool = True) -> None:
        super().__init__()
        self.custom_filter = custom_filter
        self.keep_nodes = keep_nodes
        self.rows: list[dict] = []

    def _add(self, obj, wkb: str) -> None:
        try:
            row: dict = {"geometry": _shp_wkb.loads(wkb, hex=True)}
            row.update(dict(obj.tags))
            self.rows.append(row)
        except Exception:
            pass

    def node(self, n) -> None:
        if self.keep_nodes and _tag_matches(n.tags, self.custom_filter):
            try:
                self._add(n, _wkb_factory.create_point(n))
            except Exception:
                pass

    def way(self, w) -> None:
        """Open ways only — closed ways are handled as areas by area()."""
        if w.is_closed():
            return
        if not _tag_matches(w.tags, self.custom_filter):
            return
        try:
            self._add(w, _wkb_factory.create_linestring(w))
        except Exception:
            pass

    def area(self, a) -> None:
        """Handles closed ways and multipolygon relations as proper polygons."""
        if not _tag_matches(a.tags, self.custom_filter):
            return
        try:
            self._add(a, _wkb_factory.create_multipolygon(a))
        except Exception:
            pass

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config(BaseConfig):
    """Infrastructure pipeline configuration."""

    # Analysis thresholds
    SOLAR_MIN_AREA_SQM     = 100
    SUBSTATION_ROAD_DIST_M = 200
    TUNNEL_MIN_LEN_M       = 500
    BRIDGE_MIN_LEN_M       = 50
    DAM_MIN_LEN_FT         = 150
    FARM_MIN_SQFT          = 3000
    FARM_ANCHOR_SQFT       = 9000
    FARM_CLUSTER_RADIUS_M  = 200
    FARM_MEGA_STRUCT_SQFT  = 20000

    S3_KEYS: dict[str, str] = {
        # --- Boundaries ---
        "us_states":              "USA/state borders/cb_2023_us_state_20m.gpkg",
        # --- Solar ---
        "solar":                  "USA/infrastructure/solar_all_2024q2_v1.parquet",
        "solar_uspvdb":           "USA/infrastructure/uspvdb_v3_0_20250430.parquet",
        # --- Wind ---
        "wind":                   "USA/infrastructure/wind_all_2024q2_v1.parquet",
        # --- EIA ---
        "eia_gas":                "USA/infrastructure/EIA_gas_generators_extracted.parquet",
        "eia_hydro":              "USA/infrastructure/EIA_Hydroelectric_generators_extracted.parquet",
        "eia_bess":               "USA/infrastructure/EIA_bess_generators_extracted.parquet",
        # --- Fuel ---
        "alt_fuel_stations":      "USA/infrastructure/alt_fuel_stations_CLEANED.parquet",
        # --- Substations & Grid ---
        "hifld_substations":      "USA/infrastructure/hifld_substationsv2.zip",
        "transmission_lines":     "USA/infrastructure/transmission-lines.parquet",
        # --- Industry ---
        "hifld_ethanol_gas":      "USA/infrastructure/hifld_ethanol_and_natural_gas_processing.parquet",
        "hifld_compressor":       "USA/infrastructure/hifld_natural_gas_compressor_stations.parquet",
        "intermodal_freight":     "USA/infrastructure/intermodal-freight-facilities-pipeline-terminals 2.parquet",
        "fracfocus":              "USA/infrastructure/fracfocus_active_grid_loads.parquet",
        # --- Mining ---
        "mines_active":           "USA/infrastructure/Mines_status_Active_NewMine_Intermittent.parquet",
        "sand_gravel":            "USA/infrastructure/sand-and-gravel-operations-1-geojson 2.parquet",
        "construction_minerals":  "USA/infrastructure/construction-minerals-operations-geojson 2.parquet",
        "refractory_minerals":    "USA/infrastructure/refractory-abrasive-industrial-mineral-operations-geojson 2.parquet",
        # --- Utilities ---
        "wastewater_plants":      "USA/infrastructure/icis-wastewater-treatment-plants-geojson 2.parquet",
        # --- Agriculture ---
        "mpi_establishments":     "USA/infrastructure/MPI_Establishments.parquet",
        # --- Transport ---
        "rail_network":           "USA/infrastructure/NTAD_North_American_Rail_Network_Lines_8140759925723186018.parquet",
        "dams":                   "USA/infrastructure/national-inventory-of-dams-nid-1-geojson.parquet",
        # --- Telecom ---
        "asr_towers":             "USA/infrastructure/ASR_towers_clean.parquet",
        "microwave_towers":       "USA/infrastructure/Microwave_Towers_Cleaned.parquet",
        "microwave_service":      "USA/infrastructure/microwave-service-towers-geojson.parquet",
        # --- Public ---
        "hospitals":              "USA/infrastructure/hospitals-3-geojson.parquet",
        "prisons":                "USA/infrastructure/prison-boundaries-1-geojson.parquet",
        "schools":                "USA/infrastructure/public-schools-geojson_CLEANED.parquet",
        # --- FRS folders ---
        "frs_primary":            "USA/infrastructure/frs_primary/",
        "frs_secondary":          "USA/infrastructure/frs_secondary/",
    }


# ==============================================================================
# OSM EXTRACTOR
# ==============================================================================

class OSMExtractor:
    """
    Downloads the state OSM PBF (S3 first, Geofabrik fallback) and extracts
    features via pyosmium. The temp file is cleaned up on garbage collection.
    """

    GEOFABRIK_BASE = "https://download.geofabrik.de/north-america/us"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pbf: Path | None = None

        geofabrik_slug = cfg.state_name.lower().replace(" ", "-")
        url = f"{self.GEOFABRIK_BASE}/{geofabrik_slug}-latest.osm.pbf"
        logger.info(f"  Downloading from Geofabrik: {url}")
        tmp = self._download_http(url)
        if tmp:
            self._pbf = tmp
        else:
            logger.warning("OSM PBF unavailable — OSM extractions will be skipped.")

    @staticmethod
    def _download_http(url: str) -> Path | None:
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False)
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
            tmp.flush()
            return Path(tmp.name)
        except Exception as e:
            logger.warning(f"  [OSM] HTTP download failed: {e}")
            return None

    def __del__(self) -> None:
        if self._pbf and self._pbf.exists():
            self._pbf.unlink(missing_ok=True)

    def extract(
        self,
        custom_filter: dict,
        name: str,
        keep_nodes: bool = True,
        point_only: bool = False,
    ) -> gpd.GeoDataFrame | None:
        if not self._pbf:
            return None
        logger.info(f"  [OSM] {name}")
        try:
            handler = _OsmHandler(custom_filter, keep_nodes=keep_nodes)
            # locations=True + flex_mem resolves node coordinates for way/relation geometry
            handler.apply_file(str(self._pbf), locations=True, idx="flex_mem")

            if not handler.rows:
                return None

            gdf = gpd.GeoDataFrame(handler.rows, geometry="geometry", crs="EPSG:4326")
            gdf = gdf[~gdf.geometry.isna()].copy()
            if point_only:
                gdf = gdf[gdf.geometry.type.isin(["Point", "MultiPoint"])]
            gdf["source"] = "OSM"
            gdf["category"] = name
            return gdf.to_crs(self.cfg.crs_metric)
        except Exception as e:
            logger.warning(f"  [OSM error] {name}: {e}")
            return None


# ==============================================================================
# FEMA DOWNLOADER
# ==============================================================================

class FEMADownloader:
    """
    Downloads FEMA USA_Structures for a given state directly from the public
    FEMA S3 bucket (no credentials required).

    Source: https://fema-femadata.s3.amazonaws.com/Partners/ORNL/USA_Structures/
    """

    FEMA_BUCKET = "https://fema-femadata.s3.amazonaws.com/"

    def __init__(self, state_name: str, state_abbrev: str) -> None:
        self.state_name   = state_name
        self.state_abbrev = state_abbrev

    def _head_ok(self, url: str) -> bool:
        try:
            return requests.head(url, timeout=10).status_code == 200
        except Exception:
            return False

    def _stream(self, url: str, dest: Path) -> Path:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return dest

    def _find_latest_key(self) -> str | None:
        prefix = f"Partners/ORNL/USA_Structures/{self.state_name}/"
        code   = self.state_abbrev
        params: dict = {"list-type": "2", "prefix": prefix}
        keys: list[str] = []

        while True:
            try:
                xml_resp = requests.get(self.FEMA_BUCKET, params=params, timeout=30)
                xml_resp.raise_for_status()
                root = ET.fromstring(xml_resp.content)
                keys += [
                    k.text for k in root.findall(".//Key")
                    if k.text and re.search(r"Deliverable\d{8}" + re.escape(code) + r"\.zip$", k.text)
                ]
                nxt = root.find(".//NextContinuationToken")
                if nxt is None or not nxt.text:
                    break
                params["continuation-token"] = nxt.text
            except Exception as e:
                logger.warning(f"  [FEMA] Listing failed: {e}")
                break

        # Fallback: probe recent dates if listing returned nothing
        if not keys:
            today = datetime.utcnow().date()
            limit = today - timedelta(days=36 * 30)
            date  = today
            while date >= limit:
                ymd      = date.strftime("%Y%m%d")
                test_key = f"{prefix}Deliverable{ymd}{code}.zip"
                if self._head_ok(self.FEMA_BUCKET + url_quote(test_key, safe="/")):
                    keys.append(test_key)
                    break
                date -= timedelta(days=1)

        if not keys:
            return None
        return max(keys, key=lambda k: re.search(r"Deliverable(\d{8})", k).group(1))  # type: ignore[union-attr]

    def fetch(self) -> gpd.GeoDataFrame | None:
        """Returns raw FEMA structures GDF (OCC_CLS, PRIM_OCC, HEIGHT, geometry)."""
        logger.info(f"  [FEMA] Locating latest ZIP for {self.state_name}...")
        key = self._find_latest_key()
        if key is None:
            logger.warning(f"  [FEMA] No ZIP found for {self.state_name} — skipping.")
            return None

        zip_url = self.FEMA_BUCKET + url_quote(key, safe="/")
        logger.info(f"  [FEMA] Downloading {Path(key).name} ...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            zip_path = self._stream(zip_url, tmp / Path(key).name)

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)

            gdb_path = next(tmp.rglob("*.gdb"), None)
            if gdb_path is None:
                logger.warning("  [FEMA] No .gdb found in ZIP.")
                return None

            layer = f"{self.state_abbrev}_Structures"
            try:
                gdf = gpd.read_file(gdb_path, layer=layer)
            except Exception as e:
                logger.warning(f"  [FEMA] Could not read layer {layer}: {e}")
                return None

            keep = [c for c in ("OCC_CLS", "PRIM_OCC", "HEIGHT", "geometry") if c in gdf.columns]
            return gdf[keep].copy()


# ==============================================================================
# PIPELINE
# ==============================================================================

class InfrastructurePipeline:

    def __init__(self, cfg: Config, force: bool = False) -> None:
        self.cfg   = cfg
        self.force = force
        self.s3    = S3Reader(cfg.S3_BUCKET)
        self.geo   = GeoUtils(cfg)
        self.mask  = self.geo.get_state_mask(self.s3)
        self.osm   = OSMExtractor(cfg)
        logger.info(
            f"Pipeline ready | {cfg.state_name} ({cfg.state_abbrev})"
            f" | CRS: {cfg.crs_metric}"
            f" | Output: {cfg.output_root}"
            + (" | force=True" if force else "")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, key: str) -> gpd.GeoDataFrame | None:
        """Load a national/state dataset from S3, clip to state."""
        return self.geo.load_and_clip(self.s3, self.mask, self.cfg.s3_uri(key))

    def _save(self, gdf: gpd.GeoDataFrame | None, category: str, stem: str) -> None:
        ab = self.cfg.state_abbrev
        GeoUtils.save_geojson(gdf, self.cfg.out(category, f"{ab}_{stem}.geojson"))

    @staticmethod
    def _to_points(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
        """Replace non-point geometries with their centroids."""
        if gdf is None or gdf.empty:
            return gdf
        gdf = gdf.copy()
        non_point = ~gdf.geometry.type.isin(["Point", "MultiPoint"])
        gdf.loc[non_point, "geometry"] = gdf.loc[non_point, "geometry"].centroid
        return gdf

    def _merge(self, gdfs: list) -> gpd.GeoDataFrame | None:
        parts = [g for g in gdfs if g is not None and not g.empty]
        if not parts:
            return None
        return gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs=self.cfg.crs_metric,
        )

    def _cached(self, *specs: tuple[str, str]) -> bool:
        """
        Returns True (and logs a skip) if all expected output files already exist
        and --force was not passed. Each spec is (category, stem).
        """
        if self.force:
            return False
        ab = self.cfg.state_abbrev
        missing = [
            self.cfg.out(cat, f"{ab}_{stem}.geojson")
            for cat, stem in specs
            if not self.cfg.out(cat, f"{ab}_{stem}.geojson").exists()
        ]
        if missing:
            return False
        names = ", ".join(f"{ab}_{stem}.geojson" for _, stem in specs)
        logger.info(f"  [Cached] {names} — skipping (use --force to rerun)")
        return True

    # ------------------------------------------------------------------
    # 1. ENERGY GENERATION
    # ------------------------------------------------------------------

    def process_solar(self) -> None:
        logger.info("\n--- Solar ---")
        if self._cached(("GENERATORS", "solar_merged")): return
        gdfs = [
            self.osm.extract(
                {
                    "power": ["generator", "plant", "solar_generator"],
                    "generator:source": ["solar"],
                    "plant:source": ["solar"],
                },
                "Solar_OSM",
            ),
            self._load("solar"),
            self._load("solar_uspvdb"),
        ]
        merged = self._merge(gdfs)
        if merged is not None:
            polys = merged[merged.geometry.type.isin(["Polygon", "MultiPolygon"])]
            polys = polys[polys.geometry.area >= self.cfg.SOLAR_MIN_AREA_SQM].copy()
            # Dissolve overlapping polygons into single non-overlapping geometries
            dissolved = polys.dissolve().explode(index_parts=False).reset_index(drop=True)
            merged = dissolved
        self._save(merged, "GENERATORS", "solar_merged")

    def process_wind(self) -> None:
        logger.info("\n--- Wind ---")
        if self._cached(("GENERATORS", "wind_merged")): return
        self._save(self._load("wind"), "GENERATORS", "wind_merged")

    def process_eia_generators(self) -> None:
        logger.info("\n--- EIA Generators ---")
        if self._cached(("GENERATORS", "eia_gas_hydro"), ("GENERATORS", "bess_charging_stations")): return
        # Files are pre-split by technology in S3 — no filtering needed
        self._save(
            self._merge([self._load("eia_gas"), self._load("eia_hydro")]),
            "GENERATORS", "eia_gas_hydro",
        )
        self._save(
            self._merge([self._load("eia_bess"), self._load("alt_fuel_stations")]),
            "GENERATORS", "bess_charging_stations",
        )

    # ------------------------------------------------------------------
    # 2. GRID INFRASTRUCTURE
    # ------------------------------------------------------------------

    def process_substations(self) -> None:
        logger.info("\n--- Substations ---")
        if self._cached(("SUBSTATIONS", "substations_final")): return
        gdfs = []

        osm_subs = self.osm.extract({"power": ["substation"]}, "Substation_OSM")
        if osm_subs is not None:
            osm_subs = osm_subs.copy()
            # Polygons → centroids for consistent point dataset
            osm_subs["geometry"] = osm_subs.geometry.centroid
            gdfs.append(osm_subs)

        hifld = self._load("hifld_substations")
        if hifld is not None:
            if "TYPE" in hifld.columns:
                hifld = hifld[hifld["TYPE"].str.lower() == "substation"]
            gdfs.append(hifld)

        merged = self._merge(gdfs)
        if merged is None:
            self._save(None, "SUBSTATIONS", "substations_final")
            return

        # Optional: filter by proximity to the road network (extracted from OSM)
        roads_gdf = self.osm.extract(
            {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                         "unclassified", "residential"]},
            "Roads_OSM",
            keep_nodes=False,
        )
        if roads_gdf is not None and not roads_gdf.empty:
            logger.info(f"  Road-proximity filter ({self.cfg.SUBSTATION_ROAD_DIST_M} m)...")
            roads_gdf = roads_gdf.to_crs(self.cfg.crs_metric)
            merged = gpd.sjoin_nearest(
                merged, roads_gdf,
                max_distance=self.cfg.SUBSTATION_ROAD_DIST_M,
                how="inner",
            )
            merged = merged[~merged.index.duplicated(keep="first")]
        else:
            logger.warning("  Road network not found — skipping proximity filter.")

        self._save(merged, "SUBSTATIONS", "substations_final")

    def process_transmission_lines(self) -> None:
        logger.info("\n--- Transmission & Distribution Lines ---")
        if self._cached(("GRID", "transmission_lines"), ("GRID", "distribution_lines")): return

        # High-voltage transmission lines from S3
        self._save(self._load("transmission_lines"), "GRID", "transmission_lines")

        # Distribution grid from OSM: power=line / power=minor_line, voltage < 100 kV or untagged
        osm_lines = self.osm.extract(
            {"power": ["line", "minor_line"]},
            "Grid_OSM",
            keep_nodes=False,
        )
        if osm_lines is not None:
            def _keep_voltage(v) -> bool:
                if not isinstance(v, str) or not v.strip():
                    return True
                try:
                    return float(v.split(";")[0].strip()) < 69_000
                except ValueError:
                    return True
            voltage_col = osm_lines.get("voltage", pd.Series(dtype=str, index=osm_lines.index))
            osm_lines = osm_lines[voltage_col.apply(_keep_voltage)].copy()
        self._save(osm_lines, "GRID", "osm_distribution_lines")

    # ------------------------------------------------------------------
    # 3. INDUSTRY & MINING
    # ------------------------------------------------------------------

    def process_oil_gas_chemical(self) -> None:
        logger.info("\n--- Oil / Gas / Chemical ---")
        if self._cached(("INDUSTRY", "oil_gas_chemical")): return
        gdfs = []

        gdfs.append(self.osm.extract(
            {"man_made": ["petroleum_well", "pumping_rig"],
             "industrial": ["fracking", "wellsite", "well_cluster"]},
            "Extraction_OSM",
        ))
        processing = self.osm.extract(
            {"industrial": ["oil", "gas"],
             "man_made": ["gasometer", "storage_tank"]},
            "Processing_OSM",
        )
        if processing is not None:
            # Drop water tanks (content=water) and untagged tanks (no content tag)
            is_tank = processing.get("man_made") == "storage_tank"
            content = processing.get("content", pd.Series("", index=processing.index)).fillna("")
            drop = is_tank & (content.str.lower().isin(["water", ""]))
            processing = processing[~drop].copy()
        gdfs.append(processing)

        transport = self.osm.extract(
            {"pipeline": ["valve", "marker", "vent", "station"],
             "man_made": ["pipeline"],
             "utility": ["gas"],
             "substance": ["gas", "natural_gas", "oil"]},
            "Transport_OSM",
        )
        if transport is not None:
            transport = transport[
                ~transport.geometry.type.isin(["LineString", "MultiLineString"])
            ].copy()
            gdfs.append(transport)

        for key in ("fracfocus", "hifld_ethanol_gas", "hifld_compressor"):
            gdfs.append(self._load(key))

        self._save(self._merge(gdfs), "INDUSTRY", "oil_gas_chemical")

    def process_industry(self) -> None:
        logger.info("\n--- Heavy Industry & Manufacturing ---")
        if self._cached(("INDUSTRY", "industry"), ("INDUSTRY", "works")): return
        gdfs = []

        gdfs.append(self.osm.extract(
            {"industrial": ["concrete_plant", "timber", "sawmill", "grinding_mill",
                            "brickyard", "brickworks", "scrap_yard", "auto_wrecker",
                            "waste_transfer_station"],
             "landuse": ["landfill"],
             "amenity": ["waste_transfer_station"]},
            "HeavyMaterials_OSM",
        ))
        gdfs.append(self.osm.extract(
            {"industrial": ["factory", "manufacturing", "machine_shop",
                            "metal_processing", "shipyard"],
             "man_made": ["compressor"]},
            "Manufacturing_OSM",
        ))
        gdfs.append(self._load("intermodal_freight"))

        self._save(self._merge(gdfs), "INDUSTRY", "industry")

        # General industrial works not covered by specific categories
        works = self.osm.extract(
            {"landuse": ["industrial"],
             "man_made": ["works"],
             "building": ["industrial"]},
            "Works_OSM",
        )
        self._save(self._to_points(works), "INDUSTRY", "works")

    def process_mining(self) -> None:
        logger.info("\n--- Mining ---")
        if self._cached(("MINING", "mining_final")): return
        gdfs = []

        osm_mines = self.osm.extract(
            {"landuse": ["quarry", "mine"],
             "man_made": ["quarry", "mine"],
             "industrial": ["quarry", "mine"]},
            "Mines_OSM",
        )
        if osm_mines is not None:
            osm_mines = osm_mines.copy()
            # Keep only polygon features (converted to centroid); ignore raw nodes
            osm_mines = osm_mines[
                osm_mines.geometry.type.isin(["Polygon", "MultiPolygon"])
            ]
            osm_mines["geometry"] = osm_mines.geometry.centroid
            gdfs.append(osm_mines)

        # mines_active is pre-filtered in S3 — no status column check needed
        for key in ("mines_active", "sand_gravel", "construction_minerals", "refractory_minerals"):
            gdfs.append(self._load(key))

        self._save(self._merge(gdfs), "MINING", "mining_final")

    # ------------------------------------------------------------------
    # 4. UTILITIES
    # ------------------------------------------------------------------

    def process_utilities(self) -> None:
        logger.info("\n--- Utilities ---")
        if self._cached(("UTILITIES", "utilities_merged")): return
        gdfs = [
            self.osm.extract(
                {"man_made": ["water_works", "wastewater_plant", "pumping_station"],
                 "industrial": ["cooling", "heating_station", "water"]},
                "Utilities_OSM",
            ),
            self._load("wastewater_plants"),
        ]
        self._save(self._to_points(self._merge(gdfs)), "UTILITIES", "utilities_merged")

    # ------------------------------------------------------------------
    # 5. FEMA STRUCTURES
    # ------------------------------------------------------------------

    def process_fema(self) -> None:
        logger.info("\n--- FEMA Structures ---")
        if self._cached(("BUILDINGS", "fema_buildings"), ("BUILDINGS", "fema_building_clusters")): return

        fema_raw = FEMADownloader(self.cfg.state_name, self.cfg.state_abbrev).fetch()
        fema = self.geo.clip(fema_raw, self.mask, "fema_structures") if fema_raw is not None else None

        if fema is None:
            self._save(None, "BUILDINGS", "fema_buildings")
            self._save(None, "BUILDINGS", "fema_building_clusters")
            return

        fema = fema.to_crs(self.cfg.crs_metric).copy()
        self._save(fema, "BUILDINGS", "fema_buildings")

        logger.info("  DBSCAN building cluster detection...")
        fema["sqft"] = fema.geometry.area * 10.764  # m² → ft²
        cands = fema[fema["sqft"] >= self.cfg.FARM_MIN_SQFT]
        cluster_rows: list[dict] = []

        if not cands.empty:
            coords = [(p.x, p.y) for p in cands.geometry.centroid]
            _db = DBSCAN(eps=self.cfg.FARM_CLUSTER_RADIUS_M, min_samples=3)
            _db.fit(coords)
            cands = cands.copy()
            cands["cluster"] = _db.labels_

            for cid, grp in cands[cands["cluster"] != -1].groupby("cluster"):
                if grp["sqft"].max() >= self.cfg.FARM_ANCHOR_SQFT:
                    anchor = grp.loc[grp["sqft"].idxmax()]
                    cluster_rows.append({
                        "geometry":  anchor.geometry.centroid,
                        "category":  "Building_Cluster",
                        "size_sqft": float(grp["sqft"].sum()),
                        "count":     len(grp),
                    })

            for _, row in cands[(cands["sqft"] >= self.cfg.FARM_MEGA_STRUCT_SQFT) & (cands["cluster"] == -1)].iterrows():
                cluster_rows.append({
                    "geometry":  row.geometry.centroid,
                    "category":  "Mega_Structure",
                    "size_sqft": float(row["sqft"]),
                    "count":     1,
                })

        clusters = gpd.GeoDataFrame(cluster_rows, crs=self.cfg.crs_metric) if cluster_rows else None
        self._save(clusters, "BUILDINGS", "fema_building_clusters")

    # ------------------------------------------------------------------
    # 6. AGRICULTURE
    # ------------------------------------------------------------------

    def process_agriculture(self) -> None:
        logger.info("\n--- Agriculture ---")
        if self._cached(("AGRICULTURE", "ag_farms_merged")): return
        gdfs = []

        gdfs.append(self.osm.extract(
            {"industrial": ["food", "grain_processing", "slaughterhouse", "meat", "bakery"],
             "craft": ["brewery", "winery", "distillery", "caterer", "confectionery"],
             "building": ["brewery"],
             "product": ["food", "meat", "dairy", "seafood", "bakery", "drinks"]},
            "Food_Proc_OSM",
        ))

        ag_struct = self.osm.extract(
            {"man_made": ["silo"],
             "building": ["greenhouse", "granary"]},
            "Ag_Struct_OSM",
        )
        if ag_struct is not None:
            ag_struct = ag_struct.copy()
            ag_struct["area_sqm"] = ag_struct.geometry.area
            # Drop tiny greenhouses
            tiny = (ag_struct.get("building") == "greenhouse") & (ag_struct["area_sqm"] < 500)
            ag_struct = ag_struct[~tiny]
            gdfs.append(ag_struct)

        gdfs.append(self._load("mpi_establishments"))

        self._save(self._to_points(self._merge(gdfs)), "AGRICULTURE", "ag_farms_merged")

    # ------------------------------------------------------------------
    # 6. TRANSPORT
    # ------------------------------------------------------------------

    def process_transport(self) -> None:
        logger.info("\n--- Transport ---")
        if self._cached(("TRANSPORT", "tunnels_major"), ("TRANSPORT", "bridges_lines"), ("TRANSPORT", "rail_network"), ("TRANSPORT", "dams_major")): return

        # Tunnels (linear, long enough to be meaningful)
        tunnels = self.osm.extract({"tunnel": ["yes"]}, "Tunnels_OSM")
        if tunnels is not None:
            tunnels = tunnels[tunnels.geometry.length >= self.cfg.TUNNEL_MIN_LEN_M].copy()
        self._save(tunnels, "TRANSPORT", "tunnels_major")

        # Bridges — lines only, with metric length attribute
        bridges = self.osm.extract({"bridge": ["yes"]}, "Bridges_OSM", keep_nodes=False)
        if bridges is not None:
            bridges = bridges[bridges.geometry.type.isin(["LineString", "MultiLineString"])].copy()
            bridges["length_m"] = bridges.geometry.length
            bridges = bridges[bridges["length_m"] >= self.cfg.BRIDGE_MIN_LEN_M]
        self._save(bridges, "TRANSPORT", "bridges_lines")

        # Rail network
        self._save(self._load("rail_network"), "TRANSPORT", "rail_network")

        # Dams (length threshold is in feet as per the NID schema)
        dams = self._load("dams")
        if dams is not None and "DAM_LENGTH" in dams.columns:
            dams = dams[dams["DAM_LENGTH"] >= self.cfg.DAM_MIN_LEN_FT]
        self._save(dams, "TRANSPORT", "dams_major")

    def process_roads(self) -> None:
        logger.info("\n--- Roads ---")
        ab = self.cfg.state_abbrev
        roads_path = self.cfg.out("TRANSPORT", f"{ab}_roads.parquet")
        if not self.force and roads_path.exists():
            logger.info(f"  [Cached] {ab}_roads.parquet — skipping (use --force to rerun)")
            return
        roads = self.osm.extract(
            {"highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                         "unclassified", "residential", "service",
                         "motorway_link", "trunk_link", "primary_link"]},
            "Roads_OSM",
            keep_nodes=False,
        )
        if roads is not None:
            def _link_cat(val: str | None) -> str | None:
                if not isinstance(val, str): return None
                if val == "motorway_link": return "highway_link"
                if val == "primary_link":  return "primary_link"
                if val == "trunk_link":    return "trunk_link"
                if val.endswith("_link"):  return "generic_link"
                return None
            roads = roads.copy()
            roads["link_category"] = roads.get("highway", pd.Series(dtype=str)).apply(_link_cat)
        GeoUtils.save_parquet(roads, roads_path)

    def process_airports(self) -> None:
        logger.info("\n--- Airports ---")
        if self._cached(("TRANSPORT", "airports")): return
        airports = self.osm.extract(
            {"aeroway": ["aerodrome", "airstrip", "helipad", "heliport"]},
            "Airports_OSM",
            keep_nodes=False,
        )
        if airports is not None:
            airports = airports[airports.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        self._save(airports, "TRANSPORT", "airports")

    def process_parkings(self) -> None:
        logger.info("\n--- Parkings ---")
        if self._cached(("TRANSPORT", "parkings")): return
        parkings = self.osm.extract(
            {"amenity": ["parking"]},
            "Parkings_OSM",
            keep_nodes=False,
        )
        if parkings is not None:
            parkings = parkings[parkings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        self._save(parkings, "TRANSPORT", "parkings")

    def process_cemeteries(self) -> None:
        logger.info("\n--- Cemeteries ---")
        if self._cached(("PUBLIC", "cemeteries")): return
        cemeteries = self.osm.extract(
            {"landuse": ["cemetery"], "amenity": ["grave_yard"]},
            "Cemeteries_OSM",
            keep_nodes=False,
        )
        if cemeteries is not None:
            cemeteries = cemeteries[cemeteries.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        self._save(cemeteries, "PUBLIC", "cemeteries")

    # ------------------------------------------------------------------
    # 7. TELECOM
    # ------------------------------------------------------------------

    def process_telecom(self) -> None:
        logger.info("\n--- Telecom ---")
        if self._cached(("TELECOM", "towers_major"), ("TELECOM", "towers_minor")): return

        major = [
            self._load("asr_towers"),
            self._load("microwave_towers"),
            self._load("microwave_service"),
        ]
        self._save(self._merge(major), "TELECOM", "towers_major")

        minor = [
            self.osm.extract(
                {"tower": ["communication"], "man_made": ["mast"]},
                "Towers_Minor_OSM",
            ),
        ]
        self._save(self._merge(minor), "TELECOM", "towers_minor")

    # ------------------------------------------------------------------
    # 8. PUBLIC INFRASTRUCTURE
    # ------------------------------------------------------------------

    def process_public_infra(self) -> None:
        logger.info("\n--- Public Infrastructure ---")
        if self._cached(("PUBLIC", "public_infra"), ("PUBLIC", "hospitality")): return
        gdfs: list[gpd.GeoDataFrame | None] = []

        # Aerial tramways / ski lifts
        gdfs.append(self.osm.extract(
            {"aerialway": ["chair_lift", "gondola", "cable_car", "drag_lift", "mixed_lift"]},
            "Ski_Aerial_OSM",
        ))

        # Institutional
        for key, cat in (("hospitals", "Hospital"), ("prisons", "Prison"), ("schools", "School")):
            res = self._load(key)
            if res is not None:
                res = res.copy()
                res["category"] = cat
                gdfs.append(res)

        # Hotels & short-stay accommodation
        hotels = self.osm.extract(
            {"tourism": ["hotel", "motel", "hostel", "guest_house"], "building": ["hotel"]},
            "Hotels_OSM",
        )
        if hotels is not None:
            hotels = hotels.copy()
            hotels["category"] = "Hotel"
            gdfs.append(hotels)

        # Monasteries / religious complexes
        monasteries = self.osm.extract(
            {"amenity": ["monastery"], "building": ["monastery", "convent"]},
            "Monasteries_OSM",
        )
        if monasteries is not None:
            monasteries = monasteries.copy()
            monasteries["category"] = "Monastery"
            gdfs.append(monasteries)

        # Hospitality sub-layer (hotels + monasteries only)
        hosp_categories = {"Hotel", "Monastery"}
        hosp = [
            g for g in gdfs
            if g is not None
            and "category" in g.columns
            and g["category"].isin(hosp_categories).any()
        ]
        self._save(self._to_points(self._merge(hosp)), "PUBLIC", "hospitality")

        # Save combined public infrastructure — hospitality excluded (separate file)
        non_hosp = [
            g for g in gdfs
            if g is not None
            and not ("category" in g.columns and g["category"].isin(hosp_categories).any())
        ]
        self._save(self._to_points(self._merge(non_hosp)), "PUBLIC", "public_infra")

    # ------------------------------------------------------------------
    # 9. FRS (Facility Registry)
    # ------------------------------------------------------------------

    def process_frs(self) -> None:
        logger.info("\n--- FRS ---")
        if self._cached(("FRS", "frs_primary_merged"), ("FRS", "frs_secondary_merged")): return
        for tier in ("frs_primary", "frs_secondary"):
            uris = self.s3.list_prefix(self.cfg.s3_uri(tier))
            gdfs = [
                self.geo.load_and_clip(self.s3, self.mask, uri)
                for uri in uris
                if uri.endswith(".parquet")
            ]
            self._save(self._merge(gdfs), "FRS", f"{tier}_merged")

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        logger.info(
            f"\n{'='*60}\n"
            f" Infrastructure Pipeline\n"
            f" State : {self.cfg.state_name} ({self.cfg.state_abbrev})\n"
            f" CRS   : {self.cfg.crs_metric}\n"
            f" Output: {self.cfg.output_root}\n"
            f"{'='*60}"
        )

        self.process_solar()
        self.process_wind()
        self.process_eia_generators()
        self.process_substations()
        self.process_transmission_lines()
        self.process_oil_gas_chemical()
        self.process_industry()
        self.process_mining()
        self.process_utilities()
        self.process_fema()
        self.process_agriculture()
        self.process_transport()
        self.process_roads()
        self.process_airports()
        self.process_parkings()
        self.process_telecom()
        self.process_public_infra()
        self.process_cemeteries()
        self.process_frs()

        logger.info("\n=== PIPELINE COMPLETE ===")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infrastructure data pipeline — reads from S3, writes to raw_data/<STATE>/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fetch_infra_data.py Massachusetts\n"
            "  python fetch_infra_data.py \"New York\"\n"
            "  python fetch_infra_data.py Texas\n"
        ),
    )
    parser.add_argument("state", help="Full US state name, e.g. 'Massachusetts' or 'New York'")
    parser.add_argument("--force", "-f", action="store_true", help="Rerun all steps even if outputs already exist")
    args = parser.parse_args()

    try:
        cfg = Config(args.state)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    InfrastructurePipeline(cfg, force=args.force).run_all()


if __name__ == "__main__":
    main()