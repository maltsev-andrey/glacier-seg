"""Stage 03: read SWISSIMAGE / swissALTI3D COGs over HTTP at reduced resolution and store local tiles.

COG overviews mean only the needed pyramid level is fetched (range requests), the 0.1 m
full-resolution file (~300 MB per km2) is never downloaded.

Output:
  raw/swissimage/<year>/<tile>.tif      RGB uint8, 0.5 m, EPSG:2056, tiled + LZW
  raw/swissalti3d/<tile>.tif            float32 elevation, 2 m
  manifest/download_log.csv             status per file
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from common import load_config

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
)


def fetch(href: str, out: Path, tile: str, res: float, tile_m: int, resampling: Resampling) -> str:
    if out.exists():
        return "skip"
    tmp = out.with_suffix(".part")
    e_km, n_km = (int(v) for v in tile.split("-"))
    size = int(round(tile_m / res))
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(f"/vsicurl/{href}") as src:
            data = src.read(out_shape=(src.count, size, size), resampling=resampling)
            profile = dict(
                driver="GTiff", width=size, height=size, count=src.count, dtype=data.dtype,
                crs="EPSG:2056", transform=from_origin(e_km * 1000, (n_km + 1) * 1000, res, res),
                tiled=True, blockxsize=512, blockysize=512, compress="LZW",
                predictor=2 if data.dtype.kind in "ui" else 3, nodata=src.nodata,
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
    os.replace(tmp, out)
    return "ok"


def run(jobs, workers):
    log = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            href, out = futs[f][0], futs[f][1]
            try:
                status = f.result()
            except Exception as e:
                status = f"error: {e}"
            log.append({"file": str(out), "href": href, "status": status})
            if i % 10 == 0 or status.startswith("error"):
                print(f"  [{i}/{len(jobs)}] {out.name}: {status}")
    return log


def main():
    cfg = load_config()
    p, r = cfg["paths"], cfg["raster"]
    man = Path(p["manifest"])

    img = pd.read_csv(man / "tiles_image.csv")
    dem = pd.read_csv(man / "tiles_dem.csv")

    jobs = [
        (row.href, Path(p["raw_image"]) / str(row.year) / f"{row.tile}.tif", row.tile,
         r["target_res_m"], r["tile_size_m"], Resampling.average)
        for row in img.itertuples()
    ]
    jobs += [
        (row.href, Path(p["raw_dem"]) / f"{row.tile}.tif", row.tile,
         r["dem_res_m"], r["tile_size_m"], Resampling.bilinear)
        for row in dem.itertuples()
    ]
    print(f"{len(jobs)} files to fetch ({len(img)} image, {len(dem)} DEM), workers={r['workers']}")
    log = run(jobs, r["workers"])

    df = pd.DataFrame(log)
    df.to_csv(man / "download_log.csv", index=False)
    print("\n" + df["status"].str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
