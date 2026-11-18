"""Stage 08 (Phase 3): score predictions against the independent SGI2023 inventory.

The honest test: SGI2023 (acquired 2021-2024) was never used for training —
including Morteratsch, which has no usable SGI2016 pair at all. For every
pilot glacier and every predicted image year, this script rasterizes SGI2023
on the glacier's tile set and reports:

  iou_glacier   binary ice-or-debris vs the SGI2023 outline
  iou_ice/iou_debris  per class, when SGI2023 ships a debris layer
  area_pred/area_sgi2023  km2 of glacier pixels over the tile set

The same +-10 m ignore band as in training labels is applied around SGI2023
boundaries (phase3.sgi2023_ignore_band_m), so boundary digitization tolerance
is not scored. Areas are computed over the glacier's full tile set, so
neighbouring ice inside those tiles counts in both columns equally.

The year whose imagery matches the SGI2023 acquisition date should score
best; earlier years will show the retreat as area_pred > area_sgi2023.

Output:
  manifest/eval_sgi2023.csv   one row per (glacier, year)
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from common import find_layer, load_config, read_layer

N = 3
IGNORE = 255


def tile_transform(tile: str, tile_m: int, res: float):
    e_km, n_km = (int(v) for v in tile.split("-"))
    bbox = (e_km * tile_m, n_km * tile_m, (e_km + 1) * tile_m, (n_km + 1) * tile_m)
    return bbox, from_origin(bbox[0], bbox[3], res, res)


def main():
    cfg = load_config()
    p, r, ph2, ph3 = cfg["paths"], cfg["raster"], cfg["phase2"], cfg["phase3"]
    repo = Path(p["root"]).parent
    resolve = lambda v: Path(v) if Path(v).is_absolute() else repo / v
    pred_root = resolve(ph3["predictions_dir"]) / ph3["run"]
    res, tile_m = r["target_res_m"], r["tile_size_m"]
    size = int(round(tile_m / res))
    px_km2 = (res * res) / 1e6

    sgi_dir = Path(p["raw_glamos"]) / "sgi2023"
    o_path, o_layer = find_layer(sgi_dir, cfg["glamos"]["outline_hint"])
    outlines = read_layer(o_path, o_layer)
    try:
        d_path, d_layer = find_layer(sgi_dir, cfg["glamos"]["debris_hint"])
        debris = read_layer(d_path, d_layer)
        print(f"sgi2023: {o_path.name}:{o_layer or ''} + debris {d_path.name}:{d_layer or ''}")
    except Exception:
        debris = outlines.iloc[0:0]
        print(f"sgi2023: {o_path.name}:{o_layer or ''} (no debris layer -> ice/debris merged in scoring)")
    has_debris = len(debris) > 0

    band_m = ph3["sgi2023_ignore_band_m"]
    bands = gpd.GeoSeries(
        list(outlines.geometry.boundary) + list(debris.geometry.boundary), crs=2056
    ).buffer(band_m)
    o_sidx, d_sidx, b_sidx = outlines.sindex, debris.sindex, bands.sindex

    man = pd.read_csv(Path(p["manifest"]) / "tiles_image.csv")
    if ph3["years"]:
        man = man[man.year.isin(ph3["years"])]

    # rasterize sgi2023 once per tile, reuse across years
    label_cache: dict[str, np.ndarray] = {}

    def sgi_label(tile: str) -> np.ndarray:
        if tile in label_cache:
            return label_cache[tile]
        bbox, transform = tile_transform(tile, tile_m, res)
        def burn(geoms):
            geoms = list(geoms)
            if not geoms:
                return np.zeros((size, size), dtype=np.uint8)
            return rasterize([(g, 1) for g in geoms], out_shape=(size, size),
                             transform=transform, fill=0, dtype="uint8")
        lab = burn(outlines.geometry.iloc[o_sidx.intersection(bbox)]).copy()
        deb_r = burn(debris.geometry.iloc[d_sidx.intersection(bbox)])
        lab[(deb_r == 1) & (lab == 1)] = 2
        ign_r = burn(bands.iloc[b_sidx.intersection(bbox)])
        lab[ign_r == 1] = IGNORE
        label_cache[tile] = lab
        return lab

    rows = []
    for (glacier, year), grp in man.groupby(["glacier", "year"]):
        cm = np.zeros((N, N), dtype=np.int64)
        missing = 0
        for tile in sorted(grp.tile.unique()):
            pred_path = pred_root / str(year) / f"{tile}.tif"
            if not pred_path.exists():
                missing += 1
                continue
            with rasterio.open(pred_path) as src:
                pred = src.read(1)
            lab = sgi_label(tile)
            valid = lab != IGNORE
            np.add.at(cm, (lab[valid].astype(np.int64), pred[valid].astype(np.int64)), 1)
        if missing:
            print(f"[WARN] {glacier}/{year}: {missing} prediction tiles missing")
        if cm.sum() == 0:
            continue

        inter = np.diag(cm).astype(float)
        union = cm.sum(0) + cm.sum(1) - np.diag(cm)
        iou = inter / np.maximum(union, 1)
        # binary glacier (ice+debris merged)
        g_lab = cm[1:, :].sum()
        g_pred = cm[:, 1:].sum()
        g_inter = cm[1:, 1:].sum()
        iou_gl = g_inter / max(g_lab + g_pred - g_inter, 1)

        row = {"glacier": glacier, "year": int(year),
               "iou_glacier": round(float(iou_gl), 4),
               "area_pred_km2": round(float(g_pred) * px_km2, 3),
               "area_sgi2023_km2": round(float(g_lab) * px_km2, 3)}
        if has_debris:
            row |= {"iou_ice": round(float(iou[1]), 4), "iou_debris": round(float(iou[2]), 4)}
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["glacier", "year"])
    out_csv = Path(p["manifest"]) / "eval_sgi2023.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
