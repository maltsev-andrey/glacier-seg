"""Stage 04 (Phase 1): rasterize SGI2016 labels to 0.5 m tiles with an ignore band.

Every tile from the Phase 0 manifest gets one uint8 label raster on the same
LV95 grid as the imagery. The *whole* inventory is rasterized, not just the
pilot glaciers: neighbouring glaciers appear in the pilot tiles and must not
be labelled background.

Classes (config: classes):
  0 background, 1 clean ice/firn, 2 debris-covered ice, 255 ignore

Ignore band: +-ignore_band_m around every class boundary (outline and debris
edges). Labels are 0-4 years older than the imagery, so the boundary pixels
are exactly where the label is least trustworthy (README 5.4).

Output:
  labels/<tile>.tif            uint8, 0.5 m, EPSG:2056, tiled + LZW
  manifest/labels_report.csv   per-tile pixel counts per class
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from common import find_layer, load_config, read_layer


def tile_bounds(tile: str, tile_m: int) -> tuple[float, float, float, float]:
    e_km, n_km = (int(v) for v in tile.split("-"))
    return e_km * tile_m, n_km * tile_m, (e_km + 1) * tile_m, (n_km + 1) * tile_m


def burn(shapes, out_shape, transform, fill=0):
    if not shapes:
        return np.full(out_shape, fill, dtype=np.uint8)
    return rasterize(shapes, out_shape=out_shape, transform=transform, fill=fill, dtype="uint8")


def main():
    cfg = load_config()
    p, r, ph1 = cfg["paths"], cfg["raster"], cfg["phase1"]
    man = Path(p["manifest"])
    out_dir = Path(p["labels"])
    out_dir.mkdir(parents=True, exist_ok=True)

    res, tile_m = r["target_res_m"], r["tile_size_m"]
    size = int(round(tile_m / res))

    sgi_dir = Path(p["raw_glamos"]) / "sgi2016"
    o_path, o_layer = find_layer(sgi_dir, cfg["glamos"]["outline_hint"])
    d_path, d_layer = find_layer(sgi_dir, cfg["glamos"]["debris_hint"])
    print(f"outlines: {o_path.name}:{o_layer or ''}   debris: {d_path.name}:{d_layer or ''}")
    outlines = read_layer(o_path, o_layer)
    debris = read_layer(d_path, d_layer)

    # Ignore band source: every class boundary, buffered. Build once, index spatially.
    band_m = ph1["ignore_band_m"]
    bands = gpd.GeoSeries(
        list(outlines.geometry.boundary) + list(debris.geometry.boundary), crs=2056
    ).buffer(band_m)
    o_sidx, d_sidx, b_sidx = outlines.sindex, debris.sindex, bands.sindex

    tiles = sorted(pd.read_csv(man / "tiles_image.csv")["tile"].unique())
    print(f"{len(tiles)} tiles -> {out_dir}")

    rows = []
    for i, tile in enumerate(tiles, 1):
        minx, miny, maxx, maxy = tile_bounds(tile, tile_m)
        transform = from_origin(minx, maxy, res, res)
        bbox = (minx, miny, maxx, maxy)

        ice = outlines.geometry.iloc[o_sidx.intersection(bbox)]
        deb = debris.geometry.iloc[d_sidx.intersection(bbox)]
        ign = bands.iloc[b_sidx.intersection(bbox)]

        label = burn([(g, 1) for g in ice], (size, size), transform)
        if len(deb):
            deb_r = burn([(g, 1) for g in deb], (size, size), transform)
            label[(deb_r == 1) & (label == 1)] = 2  # debris only counts inside the outline
        if len(ign):
            ign_r = burn([(g, 1) for g in ign], (size, size), transform)
            label[ign_r == 1] = 255

        tmp = out_dir / f"{tile}.part"
        with rasterio.open(
            tmp, "w", driver="GTiff", width=size, height=size, count=1, dtype="uint8",
            crs="EPSG:2056", transform=transform, tiled=True, blockxsize=512, blockysize=512,
            compress="LZW", nodata=255,
        ) as dst:
            dst.write(label, 1)
        tmp.replace(out_dir / f"{tile}.tif")

        counts = {f"px_{c}": int((label == c).sum()) for c in (0, 1, 2, 255)}
        rows.append({"tile": tile, **counts})
        if i % 25 == 0:
            print(f"  [{i}/{len(tiles)}] {tile}")

    df = pd.DataFrame(rows)
    df.to_csv(man / "labels_report.csv", index=False)

    tot = df[[c for c in df.columns if c.startswith("px_")]].sum()
    valid = tot.sum() - tot["px_255"]
    print("\nPixel totals over all tiles:")
    for c, name in ((0, "background"), (1, "clean_ice_firn"), (2, "debris_covered_ice")):
        print(f"  {c} {name:<18} {tot[f'px_{c}']:>14,}  ({tot[f'px_{c}'] / valid:6.2%} of valid)")
    print(f"  255 ignore           {tot['px_255']:>14,}  ({tot['px_255'] / tot.sum():6.2%} of all)")


if __name__ == "__main__":
    main()
