"""Shared helpers: config loading, layer discovery, LV95 tile math, STAC paging."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import requests
import yaml

UA = {"User-Agent": "glacier-seg/0.1 (research; contact via GitHub maltsev-andrey)"}


def load_config() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("raw_glamos", "raw_image", "raw_dem", "manifest"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def http_get(url: str, *, params=None, stream=False, retries=4, timeout=60) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=UA, stream=stream, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt}/{retries} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------- GLAMOS layer discovery ----------

def list_vector_layers(folder: Path) -> list[tuple[Path, str | None]]:
    """Return (file, layer) pairs for every .shp and every layer inside .gpkg under folder."""
    import pyogrio

    out: list[tuple[Path, str | None]] = []
    for shp in sorted(folder.rglob("*.shp")):
        out.append((shp, None))
    for gpkg in sorted(folder.rglob("*.gpkg")):
        for name, _geom in pyogrio.list_layers(gpkg):
            out.append((gpkg, name))
    return out


def find_layer(folder: Path, hint: str, exclude=("divide", "point", "debris")) -> tuple[Path, str | None]:
    """Pick the first layer whose file/layer name contains hint and none of the excluded words."""
    hint = hint.lower()
    excl = [e for e in exclude if e != hint]
    for path, layer in list_vector_layers(folder):
        label = f"{path.name}:{layer or ''}".lower()
        if hint in label and not any(e in label for e in excl):
            return path, layer
    raise FileNotFoundError(f"No layer matching '{hint}' under {folder}. Check manifest/glamos_layers.txt")


def read_layer(path: Path, layer: str | None):
    import geopandas as gpd

    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS")
    return gdf.to_crs(2056)


# ---------- LV95 1 km tile math ----------

def tiles_for_geometry(geom, tile_m: int = 1000) -> set[str]:
    """swisstopo tile keys 'EEEE-NNNN' (km, lower-left corner) whose cell intersects geom (EPSG:2056)."""
    from shapely.geometry import box
    from shapely.prepared import prep

    minx, miny, maxx, maxy = geom.bounds
    pg = prep(geom)
    keys = set()
    for e in range(math.floor(minx / tile_m), math.floor(maxx / tile_m) + 1):
        for n in range(math.floor(miny / tile_m), math.floor(maxy / tile_m) + 1):
            if pg.intersects(box(e * tile_m, n * tile_m, (e + 1) * tile_m, (n + 1) * tile_m)):
                keys.add(f"{e}-{n}")
    return keys


def tile_key_from_item_id(item_id: str) -> tuple[int, str]:
    """'swissimage-dop10_2021_2600-1200' -> (2021, '2600-1200')"""
    parts = item_id.split("_")
    return int(parts[-2]), parts[-1]


# ---------- STAC ----------

def stac_items(api: str, collection: str, bbox_wgs84: tuple[float, float, float, float], limit: int = 100):
    """Yield all items of a collection intersecting bbox, following rel=next links."""
    url = f"{api}/collections/{collection}/items"
    params = {"bbox": ",".join(f"{v:.6f}" for v in bbox_wgs84), "limit": limit}
    while url:
        page = http_get(url, params=params).json()
        yield from page.get("features", [])
        nxt = [l for l in page.get("links", []) if l.get("rel") == "next"]
        url, params = (nxt[0]["href"], None) if nxt else (None, None)


def pick_asset(item: dict, gsd: float) -> str | None:
    for asset in item.get("assets", {}).values():
        if abs(float(asset.get("eo:gsd", -1)) - gsd) < 1e-6 and asset.get("href", "").endswith(".tif"):
            return asset["href"]
    return None
