#!/usr/bin/env python3
"""
Land data fetching and processing pipeline.

Reads source datasets from S3 or downloads them directly, clips to the
requested state, and writes outputs to:
    raw_data/<STATE_ABBREV>/LAND/
    raw_data/<STATE_ABBREV>/EXCLUSIONS/

Sources:
  S3 (grid-research-raw-data):
    lanid, vrm, scenic_byways, wetlands, pad_us, population
  Direct download:
    NLCD  — MRLC (mrlc.gov)
    CDL   — USDA NASS
    DEM   — Copernicus GLO-30 via Planetary Computer

Usage:
    python fetch_land_data.py Massachusetts
    python fetch_land_data.py "New York"

Requirements:
    uv add geopandas pyarrow s3fs boto3 rasterio shapely requests pystac-client planetary-computer
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box as shapely_box

from fetch_utils import BaseConfig, S3Reader, GeoUtils

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
    """Land pipeline configuration."""

    S3_KEYS: dict[str, str] = {
        # --- Boundaries ---
        "us_states":     "USA/state borders/cb_2023_us_state_20m.gpkg",
        # --- National rasters (S3) ---
        "lanid":         "USA/land/lanid2020.tif",
        # --- National vectors (S3) ---
        "scenic_byways": "USA/land/us_scenic_byways.parquet",
        "vrm":           "USA/land/US_strict_protection_gap12.parquet",
        # --- State-specific vectors (S3) — {abbrev} = uppercase ---
        "wetlands":      "USA/land/wetlands/{abbrev}_wetlands_simplified_10m.parquet",
        "population":    "USA/land/population/pop_{abbrev}.parquet",
        # --- State-specific parcels (S3) — zipped geoparquet ---
        "parcels":       "USA/land/parcels/{abbrev}_master_parcels.zip",
        # --- DSO service territory boundaries (S3) ---
        "dso_boundaries": "USA/land/dso-boundaries/{abbrev}_dso_boundaries.parquet",
    }

    # --- Direct download: NLCD 2023 (ScienceBase) ---
    NLCD_YEAR = "2024"
    NLCD_URL = (
        "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
        "Annual_NLCD_LndCov_{year}_CU_C1V1.zip"
    )

    # --- Direct download: CDL 2023 CONUS (USDA NASS) — clipped to state ---
    CDL_YEAR = "2023"
    CDL_URL  = (
        "https://www.nass.usda.gov/Research_and_Science/Cropland/"
        "Release/datasets/{year}_30m_cdls.zip"
    )


# ==============================================================================
# LAND PIPELINE
# ==============================================================================

class LandPipeline:

    def __init__(self, cfg: Config, force: bool = False) -> None:
        self.cfg   = cfg
        self.force = force
        self.s3    = S3Reader(cfg.S3_BUCKET)
        self.geo   = GeoUtils(cfg)
        self.mask  = self.geo.get_state_mask(self.s3)
        logger.info(
            f"Pipeline ready | {cfg.state_name} ({cfg.state_abbrev})"
            f" | CRS: {cfg.crs_metric}"
            f" | Output: {cfg.output_root}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cached(self, *paths: Path) -> bool:
        if self.force:
            return False
        missing = [p for p in paths if not p.exists()]
        if missing:
            return False
        names = ", ".join(p.name for p in paths)
        logger.info(f"  [Cached] {names} — skipping (use --force to rerun)")
        return True

    def _download_raster(self, key: str) -> Path | None:
        """Download a raster from S3 to a temp file, return path."""
        return self.s3.download_temp(self.cfg.s3_uri(key), suffix=".tif")

    def _load_vector(self, key: str) -> gpd.GeoDataFrame | None:
        """Load a parquet vector from S3, clip to state."""
        gdf = self.s3.read_parquet(self.cfg.s3_uri(key))
        return self.geo.clip(gdf, self.mask, key)

    def _stream_download(self, url: str, dest: Path) -> bool:
        """Stream-download a file over HTTP. Returns True on success."""
        logger.info(f"  Downloading {url} ...")
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
            return True
        except Exception as e:
            logger.warning(f"  [Download failed] {url}: {e}")
            return False

    def _extract_tif_from_zip(self, zip_path: Path, tmpdir: str) -> Path | None:
        """Extract a ZIP and return the first .tif/.img inside."""
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        for pattern in ("**/*.tif", "**/*.img"):
            matches = list(Path(tmpdir).rglob(pattern))
            if matches:
                return matches[0]
        logger.warning(f"  No raster found inside {zip_path.name}")
        return None

    # ------------------------------------------------------------------
    # 1. NLCD — National Land Cover Database (direct from MRLC)
    # ------------------------------------------------------------------

    def process_nlcd(self) -> None:
        logger.info("\n--- NLCD ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_nlcd.tif")
        if self._cached(out): return

        url = self.cfg.NLCD_URL.format(year=self.cfg.NLCD_YEAR)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "nlcd.zip"
            if not self._stream_download(url, zip_path):
                return

            tif = self._extract_tif_from_zip(zip_path, tmpdir)
            if tif is None:
                return

            result = self.geo.clip_raster_to_state(tif, out)
            if result:
                logger.info(f"  Saved {out.name}")

    # ------------------------------------------------------------------
    # 2. CDL — Cropland Data Layer (direct from USDA NASS)
    # ------------------------------------------------------------------

    def process_cdl(self) -> None:
        logger.info("\n--- CDL ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_cdl.tif")
        if self._cached(out): return

        url = self.cfg.CDL_URL.format(year=self.cfg.CDL_YEAR)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "cdl.zip"
            if not self._stream_download(url, zip_path):
                return

            tif = self._extract_tif_from_zip(zip_path, tmpdir)
            if tif is None:
                return

            result = self.geo.clip_raster_to_state(tif, out)
            if result:
                logger.info(f"  Saved {out.name}")

    # ------------------------------------------------------------------
    # 3. VRM — Strict protection areas (S3 parquet)
    # ------------------------------------------------------------------

    def process_vrm(self) -> None:
        logger.info("\n--- VRM (Strict Protection) ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_vrm.geojson")
        if self._cached(out): return

        gdf = self._load_vector("vrm")
        GeoUtils.save_geojson(gdf, out)

    # ------------------------------------------------------------------
    # 4. LANID — Irrigated land (S3 raster)
    # ------------------------------------------------------------------

    def process_lanid(self) -> None:
        logger.info("\n--- LANID ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_lanid.tif")
        if self._cached(out): return

        tmp = self._download_raster("lanid")
        if tmp is None: return

        result = self.geo.clip_raster_to_state(tmp, out)
        if result:
            logger.info(f"  Saved {out.name}")

    # ------------------------------------------------------------------
    # 5. Wetlands — NWI simplified (S3 parquet)
    # ------------------------------------------------------------------

    def process_wetlands(self) -> None:
        logger.info("\n--- Wetlands ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_wetlands.geojson")
        if self._cached(out): return

        gdf = self._load_vector("wetlands")
        GeoUtils.save_geojson(gdf, out)

    # ------------------------------------------------------------------
    # 6. Scenic Byways (S3 parquet)
    # ------------------------------------------------------------------

    def process_scenic_byways(self) -> None:
        logger.info("\n--- Scenic Byways ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_scenic_byways.geojson")
        if self._cached(out): return

        gdf = self._load_vector("scenic_byways")
        GeoUtils.save_geojson(gdf, out)

    # ------------------------------------------------------------------
    # 7. Population (S3 parquet)
    # ------------------------------------------------------------------

    def process_population(self) -> None:
        logger.info("\n--- Population ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_population.geojson")
        if self._cached(out): return

        gdf = self._load_vector("population")
        GeoUtils.save_geojson(gdf, out)

    # ------------------------------------------------------------------
    # 8. DEM — Copernicus GLO-30 (30 m, via Planetary Computer)
    # ------------------------------------------------------------------

    def process_dem(self) -> None:
        logger.info("\n--- DEM (Copernicus GLO-30) ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_dem.tif")
        if self._cached(out): return

        try:
            from pystac_client import Client
            import planetary_computer as pc
            import rasterio
            from rasterio.merge import merge as rio_merge
        except ImportError as e:
            logger.warning(f"  [Skip] Missing dependency for DEM download: {e}")
            return

        bounds = self.mask.to_crs("EPSG:4326").total_bounds
        bbox   = list(bounds)

        logger.info(f"  Searching Planetary Computer (bbox={[round(x, 3) for x in bbox]})...")
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
        if not items:
            logger.warning("  No DEM tiles found for this state.")
            return
        logger.info(f"  Found {len(items)} tiles — downloading...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tile_paths = []
            for item in items:
                url  = item.assets["data"].href
                dest = Path(tmpdir) / f"{item.id}.tif"
                if self._stream_download(url, dest):
                    tile_paths.append(dest)

            if not tile_paths:
                logger.warning("  All tile downloads failed.")
                return

            logger.info(f"  Merging {len(tile_paths)} tiles...")
            src_files = [rasterio.open(p) for p in tile_paths]
            mosaic, out_transform = rio_merge(src_files)
            meta = src_files[0].meta.copy()
            meta.update({
                "driver":    "GTiff",
                "height":    mosaic.shape[1],
                "width":     mosaic.shape[2],
                "transform": out_transform,
            })
            merged_path = Path(tmpdir) / "merged.tif"
            with rasterio.open(merged_path, "w", **meta) as dst:
                dst.write(mosaic)
            for src in src_files:
                src.close()

            result = self.geo.clip_raster_to_state(merged_path, out)
            if result:
                logger.info(f"  Saved {out.name}")

    # ------------------------------------------------------------------
    # 9. Parcels (S3 — zipped geoparquet)
    # ------------------------------------------------------------------

    def process_parcels(self) -> None:
        logger.info("\n--- Parcels ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_parcels.parquet")
        if self._cached(out): return

        zip_tmp = self.s3.download_temp(self.cfg.s3_uri("parcels"), suffix=".zip")
        if zip_tmp is None:
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_tmp) as zf:
                    zf.extractall(tmpdir)

                parquet_files = list(Path(tmpdir).rglob("*.parquet")) + \
                                list(Path(tmpdir).rglob("*.geoparquet"))
                if not parquet_files:
                    logger.warning("  No parquet file found inside parcels zip.")
                    return

                gdf = gpd.read_parquet(parquet_files[0])
                gdf = self.geo.clip(gdf, self.mask, "parcels")
                if gdf is None or gdf.empty:
                    logger.warning("  No parcel features within state boundary.")
                    return

                out.parent.mkdir(parents=True, exist_ok=True)
                gdf.to_parquet(out, index=False)
                logger.info(f"  Saved {out.name} ({len(gdf):,} features)")
        finally:
            zip_tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 10. DSO Boundaries (S3 parquet)
    # ------------------------------------------------------------------

    def process_dso_boundaries(self) -> None:
        logger.info("\n--- DSO Boundaries ---")
        out = self.cfg.out("LAND", f"{self.cfg.state_abbrev}_dso_boundaries.parquet")
        if self._cached(out): return

        gdf = self._load_vector("dso_boundaries")
        if gdf is None or gdf.empty:
            logger.warning("  No DSO boundaries found for this state.")
            return

        out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(out, index=False)
        logger.info(f"  Saved {out.name} ({len(gdf):,} features)")

    # ------------------------------------------------------------------
    # 11. Exclusion Zones (urban areas from building + population density)
    # ------------------------------------------------------------------

    def process_exclusions(self) -> None:
        """
        Builds urban exclusion polygons from:
          - FEMA building density: 1 km tile grid, filter > 350 bldg/km²,
            keep only clusters of ≥ 3 edge-adjacent tiles (rook adjacency)
          - Population density: census polygons with POP_SQMI > 2500
        Output is a dissolved GeoPackage of urban areas.
        """
        BLDG_DENSITY_THRESHOLD = 350    # buildings / km²
        POP_DENSITY_THRESHOLD  = 4000   # people / sq mile
        MIN_CLUSTER_SIZE       = 3
        TILE_SIZE_M            = 1000   # 1 km grid

        logger.info("\n--- Exclusion Zones (Urban Areas) ---")
        ab  = self.cfg.state_abbrev
        out = self.cfg.out("EXCLUSIONS", f"{ab}_exclusions.gpkg")
        if self._cached(out): return

        gdfs = []

        # --- 1. FEMA building density grid ---
        fema_path = self.cfg.out("BUILDINGS", f"{ab}_fema_buildings.geojson")
        if fema_path.exists():
            logger.info("  Building FEMA density grid...")
            buildings = gpd.read_file(fema_path).to_crs(self.cfg.crs_metric)

            if not buildings.empty:
                xmin, ymin, xmax, ymax = self.mask.to_crs(self.cfg.crs_metric).total_bounds
                tiles = [
                    shapely_box(x, y, x + TILE_SIZE_M, y + TILE_SIZE_M)
                    for x in np.arange(xmin, xmax, TILE_SIZE_M)
                    for y in np.arange(ymin, ymax, TILE_SIZE_M)
                ]
                grid = gpd.GeoDataFrame(geometry=tiles, crs=self.cfg.crs_metric)

                # Count buildings per tile
                joined = gpd.sjoin(buildings[["geometry"]], grid, how="left", predicate="within")
                counts = joined.groupby("index_right").size().rename("bldg_count")
                grid["bldg_count"] = counts.reindex(grid.index).fillna(0)
                grid["dens_bldg_km2"] = grid["bldg_count"] / (TILE_SIZE_M / 1000) ** 2

                high_dens = grid[grid["dens_bldg_km2"] > BLDG_DENSITY_THRESHOLD].copy()
                logger.info(f"  {len(high_dens):,} tiles above density threshold ({BLDG_DENSITY_THRESHOLD} bldg/km²).")

                if not high_dens.empty:
                    # Rook adjacency via spatial join (shared edges only, not diagonal)
                    neighbors = gpd.sjoin(high_dens, high_dens, how="inner", predicate="touches")
                    neighbors = neighbors.join(grid["geometry"], on="index_right", rsuffix="_right")
                    is_edge = neighbors["geometry"].intersection(neighbors["geometry_right"]).length > 0
                    edge_nbrs = neighbors[is_edge]

                    G = nx.Graph()
                    G.add_nodes_from(high_dens.index)
                    G.add_edges_from(zip(edge_nbrs.index, edge_nbrs["index_right"]))

                    kept = {
                        n
                        for comp in nx.connected_components(G)
                        if len(comp) >= MIN_CLUSTER_SIZE
                        for n in comp
                    }
                    urban_tiles = high_dens.loc[list(kept)]
                    logger.info(f"  {len(urban_tiles):,} tiles kept (clusters ≥ {MIN_CLUSTER_SIZE}).")
                    if not urban_tiles.empty:
                        gdfs.append(urban_tiles[["geometry"]])
        else:
            logger.warning(f"  FEMA buildings not found at {fema_path.name} — run fetch_infra_data.py first.")

        # --- 2. Population density ---
        pop_path = self.cfg.out("LAND", f"{ab}_population.geojson")
        if pop_path.exists():
            logger.info("  Loading population density...")
            pop = gpd.read_file(pop_path).to_crs(self.cfg.crs_metric)
            if "POP_SQMI" in pop.columns:
                pop["POP_SQMI"] = pd.to_numeric(pop["POP_SQMI"], errors="coerce").fillna(0)
                urban_pop = pop[pop["POP_SQMI"] > POP_DENSITY_THRESHOLD]
                logger.info(f"  {len(urban_pop):,} high-density population polygons (> {POP_DENSITY_THRESHOLD} ppl/mi²).")
                if not urban_pop.empty:
                    gdfs.append(urban_pop[["geometry"]])
            else:
                logger.warning(f"  No POP_SQMI column in population data (columns: {list(pop.columns)}).")
        else:
            logger.warning(f"  Population data not found at {pop_path.name} — run process_population first.")

        if not gdfs:
            logger.warning("  No exclusion inputs available — skipping.")
            return

        # --- 3. Union + dissolve ---
        merged = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            geometry="geometry",
            crs=self.cfg.crs_metric,
        )
        logger.info(f"  Dissolving {len(merged):,} features...")
        dissolved = merged.dissolve()
        GeoUtils.save_gpkg(dissolved, out, layer="exclusions")

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        logger.info(
            f"\n{'='*60}\n"
            f" Land Pipeline\n"
            f" State : {self.cfg.state_name} ({self.cfg.state_abbrev})\n"
            f" CRS   : {self.cfg.crs_metric}\n"
            f" Output: {self.cfg.output_root}\n"
            f"{'='*60}"
        )

        self.process_nlcd()
        self.process_cdl()
        self.process_vrm()
        self.process_lanid()
        self.process_wetlands()
        self.process_scenic_byways()
        self.process_population()
        self.process_dem()
        self.process_parcels()
        self.process_dso_boundaries()
        self.process_exclusions()

        logger.info("\n=== PIPELINE COMPLETE ===")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Land data pipeline — reads from S3 / direct download, writes to raw_data/<STATE>/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fetch_land_data.py Massachusetts\n"
            "  python fetch_land_data.py \"New York\"\n"
        ),
    )
    parser.add_argument("state", help="Full US state name, e.g. 'Massachusetts'")
    parser.add_argument("--force", "-f", action="store_true", help="Rerun all steps even if outputs exist")
    args = parser.parse_args()

    try:
        cfg = Config(args.state)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    LandPipeline(cfg, force=args.force).run_all()


if __name__ == "__main__":
    main()