"""
Shared utilities for fetch_infra_data.py and fetch_land_data.py.
"""
from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

import boto3
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


# ==============================================================================
# STATE METADATA
# ==============================================================================

STATE_ABBREV: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}

STATE_CRS: dict[str, str] = {
    "Alabama":        "EPSG:26929", "Alaska":         "EPSG:26903",
    "Arizona":        "EPSG:26912", "Arkansas":       "EPSG:26915",
    "California":     "EPSG:26910", "Colorado":       "EPSG:26913",
    "Connecticut":    "EPSG:26918", "Delaware":       "EPSG:26918",
    "Florida":        "EPSG:26917", "Georgia":        "EPSG:26917",
    "Hawaii":         "EPSG:26904", "Idaho":          "EPSG:26911",
    "Illinois":       "EPSG:26916", "Indiana":        "EPSG:26916",
    "Iowa":           "EPSG:26915", "Kansas":         "EPSG:26914",
    "Kentucky":       "EPSG:26916", "Louisiana":      "EPSG:26915",
    "Maine":          "EPSG:26919", "Maryland":       "EPSG:26918",
    "Massachusetts":  "EPSG:26986", "Michigan":       "EPSG:26917",
    "Minnesota":      "EPSG:26915", "Mississippi":    "EPSG:26916",
    "Missouri":       "EPSG:26915", "Montana":        "EPSG:26912",
    "Nebraska":       "EPSG:26914", "Nevada":         "EPSG:26911",
    "New Hampshire":  "EPSG:26919", "New Jersey":     "EPSG:26918",
    "New Mexico":     "EPSG:26913", "New York":       "EPSG:26918",
    "North Carolina": "EPSG:26917", "North Dakota":   "EPSG:26914",
    "Ohio":           "EPSG:26917", "Oklahoma":       "EPSG:26914",
    "Oregon":         "EPSG:26910", "Pennsylvania":   "EPSG:26918",
    "Rhode Island":   "EPSG:26919", "South Carolina": "EPSG:26917",
    "South Dakota":   "EPSG:26914", "Tennessee":      "EPSG:26916",
    "Texas":          "EPSG:26914", "Utah":           "EPSG:26912",
    "Vermont":        "EPSG:26919", "Virginia":       "EPSG:26917",
    "Washington":     "EPSG:26910", "West Virginia":  "EPSG:26917",
    "Wisconsin":      "EPSG:26916", "Wyoming":        "EPSG:26913",
}


# ==============================================================================
# BASE CONFIG
# ==============================================================================

class BaseConfig:
    """
    Base configuration for state-scoped pipelines.
    Subclasses define S3_KEYS and any domain-specific thresholds.
    """

    S3_BUCKET  = "grid-research-raw-data"
    CRS_LATLON = "EPSG:4326"

    S3_KEYS: dict[str, str] = {}

    def __init__(self, state_name: str) -> None:
        state_name = state_name.strip().title()
        if state_name not in STATE_ABBREV:
            raise ValueError(
                f"Unknown state: '{state_name}'.\n"
                f"Valid options: {sorted(STATE_ABBREV.keys())}"
            )
        self.state_name   = state_name
        self.state_abbrev = STATE_ABBREV[state_name]
        self.state_slug   = state_name.replace(" ", "_").lower()
        self.crs_metric   = STATE_CRS[state_name]

        script_dir       = Path(__file__).parent
        self.output_root = script_dir / "raw_data" / self.state_abbrev

    def s3_uri(self, key_name: str) -> str:
        key = self.S3_KEYS[key_name].format(
            slug=self.state_slug,
            abbrev=self.state_abbrev,
        )
        return f"s3://{self.S3_BUCKET}/{key}"

    def out(self, category: str, filename: str) -> Path:
        p = self.output_root / category / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# ==============================================================================
# S3 READER
# ==============================================================================

class S3Reader:
    """Thin wrapper for downloading/reading geodata from S3."""

    def __init__(self, bucket: str) -> None:
        self.bucket  = bucket
        self._client = None

    @property
    def client(self) -> boto3.client:
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    def read_parquet(self, s3_uri: str) -> gpd.GeoDataFrame | None:
        """Reads a geoparquet directly from S3 via s3fs."""
        try:
            logger.debug(f"  read_parquet: {s3_uri}")
            return gpd.read_parquet(s3_uri)
        except Exception as e:
            logger.warning(f"  [Skip] {s3_uri.rsplit('/', 1)[-1]}: {e}")
            return None

    def download_temp(self, s3_uri: str, suffix: str = "") -> Path | None:
        """Downloads an S3 object to a local temp file; caller must unlink."""
        _, key = s3_uri[5:].split("/", 1)
        bucket = s3_uri[5:].split("/", 1)[0]
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            logger.info(f"  Downloading {s3_uri} ...")
            self.client.download_file(bucket, key, tmp.name)
            return Path(tmp.name)
        except Exception as e:
            logger.warning(f"  [Skip] Could not download {s3_uri}: {e}")
            return None

    def list_prefix(self, s3_uri: str) -> list[str]:
        """Lists all object URIs under an S3 prefix."""
        prefix = s3_uri.replace(f"s3://{self.bucket}/", "")
        try:
            resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [
                f"s3://{self.bucket}/{obj['Key']}"
                for obj in resp.get("Contents", [])
            ]
        except Exception as e:
            logger.warning(f"  [Skip] list_prefix {s3_uri}: {e}")
            return []


# ==============================================================================
# GEO UTILITIES
# ==============================================================================

class GeoUtils:
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg   = cfg
        self._mask: gpd.GeoDataFrame | None = None

    def get_state_mask(self, s3: S3Reader) -> gpd.GeoDataFrame:
        if self._mask is not None:
            return self._mask

        logger.info(f"Loading state boundary: {self.cfg.state_name}")
        uri = self.cfg.s3_uri("us_states")
        ext = Path(uri).suffix
        tmp = s3.download_temp(uri, suffix=ext)
        if tmp is None:
            raise RuntimeError("Could not download US state boundaries from S3.")
        try:
            gdf = gpd.read_file(tmp)
        finally:
            tmp.unlink(missing_ok=True)

        if gdf.crs is None:
            gdf = gdf.set_crs(self.cfg.CRS_LATLON)

        col = next((c for c in ("NAME", "name", "State_Name") if c in gdf.columns), None)
        if col is None:
            raise RuntimeError("State boundary layer has no recognisable name column.")

        mask = gdf[gdf[col].str.lower() == self.cfg.state_name.lower()]
        if mask.empty:
            raise RuntimeError(f"'{self.cfg.state_name}' not found in boundaries layer.")

        self._mask = mask.to_crs(self.cfg.crs_metric)
        return self._mask

    def clip(
        self,
        gdf: gpd.GeoDataFrame | None,
        mask: gpd.GeoDataFrame,
        source: str = "",
    ) -> gpd.GeoDataFrame | None:
        if gdf is None or gdf.empty:
            return None
        try:
            if gdf.crs is None:
                gdf = gdf.set_crs(self.cfg.CRS_LATLON)
            gdf = gdf.to_crs(self.cfg.crs_metric)
            result = gpd.clip(gdf, mask)
            if result.empty:
                return None
            if source:
                result = result.copy()
                result["source_file"] = source
            return result
        except Exception as e:
            logger.warning(f"  [Clip error] {source}: {e}")
            return None

    def load_and_clip(
        self,
        s3: S3Reader,
        mask: gpd.GeoDataFrame,
        s3_uri: str,
    ) -> gpd.GeoDataFrame | None:
        """Load any S3 vector (parquet, zip, or direct file) and clip to state."""
        name = s3_uri.rsplit("/", 1)[-1]

        if s3_uri.endswith(".parquet"):
            gdf = s3.read_parquet(s3_uri)

        elif s3_uri.endswith(".zip"):
            tmp_zip = s3.download_temp(s3_uri, suffix=".zip")
            if tmp_zip is None:
                return None
            try:
                gdf = self._read_zip(tmp_zip, name)
            finally:
                tmp_zip.unlink(missing_ok=True)

        else:
            ext = Path(name).suffix
            tmp = s3.download_temp(s3_uri, suffix=ext)
            if tmp is None:
                return None
            try:
                gdf = gpd.read_file(tmp)
            except Exception as e:
                logger.warning(f"  [Read error] {name}: {e}")
                return None
            finally:
                tmp.unlink(missing_ok=True)

        return self.clip(gdf, mask, name)

    def clip_raster_to_state(self, src_path: Path, out_path: Path) -> Path | None:
        """Clips a raster to the state boundary and saves as GeoTIFF."""
        if out_path.exists():
            return out_path
        if self._mask is None:
            logger.warning("  No state mask available — cannot clip raster.")
            return None
        try:
            with rasterio.open(src_path) as src:
                mask_reproj = self._mask.to_crs(src.crs)
                shapes = [mapping(geom) for geom in mask_reproj.geometry]
                out_image, out_transform = rio_mask(src, shapes, crop=True, nodata=src.nodata)
                meta = src.meta.copy()
                meta.update({
                    "height":    out_image.shape[1],
                    "width":     out_image.shape[2],
                    "transform": out_transform,
                })
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(out_image)
            return out_path
        except Exception as e:
            logger.warning(f"  [Raster clip error] {e}")
            return None

    @staticmethod
    def _read_zip(zip_path: Path, name: str) -> gpd.GeoDataFrame | None:
        """Extracts a ZIP and reads the first recognised geo format inside."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmpdir)

            parts = []
            for pattern, driver in (
                ("**/*.geojsonseq", "GeoJSONSeq"),
                ("**/*.geojson",    None),
                ("**/*.shp",        None),
                ("**/*.gpkg",       None),
                ("**/*.gdb",        None),
            ):
                for f in Path(tmpdir).glob(pattern):
                    try:
                        parts.append(gpd.read_file(f, driver=driver) if driver else gpd.read_file(f))
                    except Exception as e:
                        logger.warning(f"  [Read error in zip] {f.name}: {e}")
                if parts:
                    break

            if not parts:
                logger.warning(f"  [Read error] No readable geo files in {name}")
                return None
            return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")

    @staticmethod
    def save_geojson(
        gdf: gpd.GeoDataFrame | None,
        output_path: Path,
        crs_out: str = "EPSG:4326",
    ) -> None:
        if gdf is None or gdf.empty:
            logger.info(f"  [Skip] No data for {output_path.name}")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_crs(crs_out).to_file(output_path, driver="GeoJSON")
        logger.info(f"  [Saved] {len(gdf):,} features → {output_path.relative_to(output_path.parents[3])}")

    @staticmethod
    def save_gpkg(
        gdf: gpd.GeoDataFrame | None,
        output_path: Path,
        layer: str = "data",
        crs_out: str = "EPSG:4326",
    ) -> None:
        if gdf is None or gdf.empty:
            logger.info(f"  [Skip] No data for {output_path.name}")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_crs(crs_out).to_file(output_path, driver="GPKG", layer=layer)
        logger.info(f"  [Saved] {len(gdf):,} features → {output_path.relative_to(output_path.parents[3])}")