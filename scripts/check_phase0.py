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

    # 1. integrity + radiometric stats on every image tile
    rows = []
    for f in sorted(img_root.glob("*/*.tif")):
        with rasterio.open(f) as src:
            a = src.read(out_shape=(3, 500, 500))  # 2 m preview is enough for stats
        black = (a.sum(0) == 0).mean() * 100
        white = (a.min(0) > 230).mean() * 100
        rows.append({"year": int(f.parent.name), "tile": f.stem, "mean": a.mean().round(1),
                     "pct_black": round(black, 1), "pct_white": round(white, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(man / "phase0_check.csv", index=False)
    print(f"image tiles read OK: {len(df)}")
    print(df.groupby("year")[["mean", "pct_black", "pct_white"]].mean().round(1).to_string())
    print("\ntiles with >5% nodata:", int((df.pct_black > 5).sum()))

    # 2. overlays per glacier: tile with most ice, tile with most debris
    sgi = Path(p["raw_glamos"]) / "sgi2016"
    outl = read_layer(*find_layer(sgi, cfg["glamos"]["outline_hint"]))
    debr = read_layer(*find_layer(sgi, cfg["glamos"]["debris_hint"], exclude=()))
    tiles = pd.read_csv(man / "tiles_image.csv")
    for gid in cfg["pilot"]["glaciers"]:
        g = outl[outl["sgi-id"] == gid]
        if g.empty:
            continue
        geom = g.union_all() if hasattr(g, "union_all") else g.unary_union
        d = debr[debr["sgi-id"] == gid]
        dgeom = (d.union_all() if hasattr(d, "union_all") else d.unary_union) if not d.empty else None
        cand = tiles[tiles.glacier == gid]

        def share(t, shape):
            if shape is None:
                return 0.0
            e, n = (int(v) for v in t.split("-"))
            return shape.intersection(box(e * 1000, n * 1000, (e + 1) * 1000, (n + 1) * 1000)).area

        # flight blocks do not follow glaciers: choose over ALL (tile, year), earliest year breaks ties
        picks = {"ice": max(cand.itertuples(), key=lambda r: (share(r.tile, geom), -r.year))}
        if dgeom is not None:
            picks["debris"] = max(cand.itertuples(), key=lambda r: (share(r.tile, dgeom), -r.year))
        for kind, best in picks.items():
            plot_tile(img_root, qc, g, debr, gid, kind, best.tile, int(best.year))


def plot_tile(img_root, qc, g, debr, gid, kind, tile, year):
    with rasterio.open(img_root / str(year) / f"{tile}.tif") as src:
        rgb = np.moveaxis(src.read(), 0, -1)
        l, b, r, t = src.bounds
    clip = box(l, b, r, t)
    g_c, d_c = g.clip(clip), debr.clip(clip)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    ax.imshow(rgb, extent=(l, r, b, t))
    if not d_c.empty:
        d_c.boundary.plot(ax=ax, color="yellow", linewidth=0.8)
    if not g_c.empty:
        g_c.boundary.plot(ax=ax, color="red", linewidth=1.2)
    ice_pct = g_c.area.sum() / clip.area * 100
    deb_pct = d_c.area.sum() / clip.area * 100
    ax.set_title(f"{g['name'].iloc[0]} ({gid})  SWISSIMAGE {year}  tile {tile}  [{kind}]\n"
                 f"red: SGI2016 outline (acq {int(g['year_acq'].iloc[0])}) {ice_pct:.0f}% of tile, "
                 f"yellow: debris {deb_pct:.0f}%")
    ax.set_xlim(l, r); ax.set_ylim(b, t); ax.set_axis_off()
    out = qc / f"{gid}_{kind}_{year}_{tile}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"overlay [{kind:6s}] {out.name}  ice {ice_pct:.0f}%  debris {deb_pct:.0f}%")


if __name__ == "__main__":
    main()
