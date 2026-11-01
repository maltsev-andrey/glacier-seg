"""Stage 02: pick pilot glaciers from SGI2016, find the 1 km tiles they cover, list STAC items per year.

Output:
  manifest/pilot_glaciers.gpkg     pilot outlines (EPSG:2056) + buffer
  manifest/tiles_image.csv         glacier, tile, year, href   (SWISSIMAGE 10 cm COG)
  manifest/tiles_dem.csv           tile, year, href            (swissALTI3D 2 m COG)
"""
import csv
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

from common import find_layer, load_config, pick_asset, read_layer, stac_items, tile_key_from_item_id, tiles_for_geometry


def match_names(gdf: gpd.GeoDataFrame, wanted: list[str]) -> gpd.GeoDataFrame:
    """Case-insensitive exact-or-prefix match of wanted names against every text column."""
    text_cols = [
        c for c in gdf.columns
        if c != "geometry" and (gdf[c].dtype == object or pd.api.types.is_string_dtype(gdf[c]))
    ]
    hits = []
    for w in wanted:
        wl = w.lower()
        found = None
        for c in text_cols:
            s = gdf[c].astype(str).str.strip().str.lower()
            m = gdf[s == wl]
            if m.empty:
                m = gdf[s.str.startswith(wl)]
            if not m.empty:
                found = m.assign(pilot_name=w, name_col=c)
                break
        if found is None:
            print(f"[WARN] '{w}' not found in any text column {text_cols}")
        else:
            hits.append(found)
    if not hits:
        raise SystemExit("No pilot glacier matched. Check names in config.yaml against glamos_layers.txt")
    return gpd.GeoDataFrame(pd.concat(hits, ignore_index=True), crs=2056)


def main():
    cfg = load_config()
    sgi_dir = Path(cfg["paths"]["raw_glamos"]) / "sgi2016"
    man = Path(cfg["paths"]["manifest"])

    path, layer = find_layer(sgi_dir, cfg["glamos"]["outline_hint"])
    print(f"SGI2016 outlines: {path.name}:{layer or ''}")
    outlines = read_layer(path, layer)

    pilot = match_names(outlines, cfg["pilot"]["glaciers"])
    # A glacier can be split into several polygons (ice divides); dissolve per pilot name
    pilot = pilot.dissolve(by="pilot_name", as_index=False)[["pilot_name", "geometry"]]
    pilot["area_km2"] = pilot.area / 1e6
    pilot["geom_buf"] = pilot.buffer(cfg["pilot"]["buffer_m"])
    print(pilot[["pilot_name", "area_km2"]].to_string(index=False))
    pilot.drop(columns="geom_buf").to_file(man / "pilot_glaciers.gpkg", layer="pilot", driver="GPKG")

    st = cfg["stac"]
    tile_m = cfg["raster"]["tile_size_m"]
    img_rows, dem_rows = [], {}
    tiles_per_year = defaultdict(set)

    for _, g in pilot.iterrows():
        wanted = tiles_for_geometry(g.geom_buf, tile_m)
        bbox = gpd.GeoSeries([g.geom_buf], crs=2056).to_crs(4326).total_bounds
        print(f"\n{g.pilot_name}: {len(wanted)} tiles")

        for item in stac_items(st["api"], st["image_collection"], tuple(bbox)):
            year, key = tile_key_from_item_id(item["id"])
            if key not in wanted:
                continue
            href = pick_asset(item, st["image_gsd_source"])
            if href:
                img_rows.append({"glacier": g.pilot_name, "tile": key, "year": year, "href": href})
                tiles_per_year[year].add(key)

        for item in stac_items(st["api"], st["dem_collection"], tuple(bbox)):
            year, key = tile_key_from_item_id(item["id"])
            if key not in wanted:
                continue
            href = pick_asset(item, st["dem_gsd_source"])
            # keep the newest DEM per tile; glacier surface lowering is a known bias, noted in README
            if href and (key not in dem_rows or year > dem_rows[key]["year"]):
                dem_rows[key] = {"tile": key, "year": year, "href": href}

    img = pd.DataFrame(img_rows).drop_duplicates(["tile", "year"]).sort_values(["glacier", "year", "tile"])
    img.to_csv(man / "tiles_image.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(dem_rows.values()).sort_values("tile").to_csv(man / "tiles_dem.csv", index=False)

    print("\nImagery epochs available (year: tiles):")
    for y in sorted(tiles_per_year):
        print(f"  {y}: {len(tiles_per_year[y])}")
    n = len(img)
    # 1 km at 0.5 m, 3 bands uint8, LZW ~ 50-60% -> ~7 MB per tile
    print(f"\nImage tiles: {n}  (~{n * 7 / 1024:.1f} GB at 0.5 m)   DEM tiles: {len(dem_rows)}")


if __name__ == "__main__":
    main()
