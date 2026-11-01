"""Phase 0 exit check: file integrity + one overlay PNG per pilot glacier (outline on imagery).

Output:
  data/manifest/phase0_check.csv     per image tile: year, mean RGB, % black (nodata), % near-white (snow/cloud)
  data/qc/<glacier>_<year>.png       overlay: SGI2016 outline (red) + debris (yellow) on the best-covered tile
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import find_layer, load_config, read_layer  # noqa: E402


def main():
    cfg = load_config()
    p = cfg["paths"]
    man, img_root = Path(p["manifest"]), Path(p["raw_image"])
    qc = Path(p["root"]) / "qc"
    qc.mkdir(exist_ok=True)

    rows = []
    for f in sorted(img_root.glob("*/*.tif")):
        with rasterio.open(f) as src:
            a = src.read(out_shape=(3, 500, 500))
        black = (a.sum(0) == 0).mean() * 100
        white = (a.min(0) > 230).mean() * 100
        rows.append({"year": int(f.parent.name), "tile": f.stem, "mean": a.mean().round(1),
                     "pct_black": round(black, 1), "pct_white": round(white, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(man / "phase0_check.csv", index=False)
    print(f"image tiles read OK: {len(df)}")
    print(df.groupby("year")[["pct_black", "pct_white"]].mean().round(1).to_string())
    print("\ntiles with >5% nodata:", int((df.pct_black > 5).sum()))

    sgi = Path(p["raw_glamos"]) / "sgi2016"
    outl = read_layer(*find_layer(sgi, cfg["glamos"]["outline_hint"]))
    debr = read_layer(*find_layer(sgi, cfg["glamos"]["debris_hint"], exclude=()))
    tiles = pd.read_csv(man / "tiles_image.csv")
    for gid in cfg["pilot"]["glaciers"]:
        g = outl[outl["sgi-id"] == gid]
        if g.empty:
            continue
        geom = g.union_all() if hasattr(g, "union_all") else g.unary_union
        cand = tiles[tiles.glacier == gid]
        year = int(cand.year.min())
        cand = cand[cand.year == year]

        def share(t):
            e, n = (int(v) for v in t.split("-"))
            return geom.intersection(box(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)).area

        tile = max(cand.tile, key=share)
        with rasterio.open(img_root / str(year) / f"{tile}.tif") as src:
            rgb = np.moveaxis(src.read(), 0, -1)
            l, b, r, t = src.bounds
        fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
        ax.imshow(rgb, extent=(l, r, b, t))
        clip = box(l, b, r, t)
        debr.clip(clip).boundary.plot(ax=ax, color="yellow", linewidth=0.8)
        g.clip(clip).boundary.plot(ax=ax, color="red", linewidth=1.2)
        name = g["name"].iloc[0]
        ax.set_title(f"{name} ({gid})  SWISSIMAGE {year}  tile {tile}\n"
                     f"red: SGI2016 outline (acq {int(g['year_acq'].iloc[0])}), yellow: debris")
        ax.set_xlim(l, r); ax.set_ylim(b, t); ax.set_axis_off()
        out = qc / f"{gid}_{year}_{tile}.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"overlay: {out}")


if __name__ == "__main__":
    main()
