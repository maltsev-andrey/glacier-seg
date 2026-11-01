# glacier-seg

Semantic segmentation of Swiss glaciers (clean ice / debris-covered ice / background) from swisstopo
SWISSIMAGE orthophotos, trained on GLAMOS Swiss Glacier Inventory outlines, run on the Tesla P100.

> Working README for the build phase. The public, visual README (gpu-nbody-simulation style) is written at publish time.

## 1. Question

How much glacier area did Switzerland lose between the SWISSIMAGE epochs 2017-2025, per glacier,
and can a U-Net reproduce the official GLAMOS outlines from RGB imagery alone (plus DEM)?

## 2. Data (verified 01.11.2026)

| Source | What | Resolution / period | License | Role |
|---|---|---|---|---|
| SWISSIMAGE 10 cm (`ch.swisstopo.swissimage-dop10`, STAC) | RGB orthophoto, 1 km COG tiles, EPSG:2056 | 0.1 m delivered, ~0.25 m effective in the Alps; 2017-2025, 3-year cycle | swisstopo OGD, free, cite source | Model input |
| swissALTI3D (`ch.swisstopo.swissalti3d`, STAC) | DEM, 1 km COG tiles | 0.5 / 2 m | swisstopo OGD | Extra input channels (elevation, slope) |
| GLAMOS SGI2016 | Outlines + debris cover + ice divides | 2013-2018 | CC BY 4.0 | Training / validation labels |
| GLAMOS SGI2023 (release 2026) | Same structure | 2021-2024 | CC BY 4.0 | **Independent test set** |
| GLAMOS SGI1850 / 1973 / 2010 | Outlines only | historical | CC BY 4.0 | Long-term context on the final maps |

Constraints found during the data check:

- **No free imagery before 2017.** SWISSIMAGE HIST (B/W, pre-1998, 0.5 m) is paid, ordered via geodata@swisstopo.ch.
  SWISSIMAGE HIST 1946 is viewable as WMS/WMTS only. The "decades" story therefore comes from the GLAMOS
  vectors (1850 -> 2023), the ML part covers 2017-2025 at 3 epochs per location.
- **No snow label.** SGI polygons give glacier and debris only, so the model has 3 classes, not 4.
  Seasonal snow outside the glacier is a known false-positive risk, handled by the DEM channels and by review.
- **DEM date mismatch.** Only the newest swissALTI3D per tile is used; ice surface lowering between epochs
  makes elevation slightly inconsistent for older imagery. Acceptable for slope, documented as a bias.

## 3. Pipeline

```
01_fetch_glamos.py   GLAMOS zips -> raw/glamos/<inventory>/, layer report
02_select_tiles.py   pilot glaciers from SGI2016 -> 1 km tile set -> STAC items per year -> manifests
03_download_tiles.py COG range reads at 0.5 m (image) / 2 m (DEM) -> local GeoTIFF tiles
04_make_labels.py    (Phase 1) rasterize SGI2016 / SGI2023 to label tiles, 0/1/2/255
05_chips.py          (Phase 1) 512x512 chips, spatial train/val split by glacier (no leakage)
train.py             (Phase 2) U-Net, ResNet34 encoder, AMP fp16, Kueue Job on queue p100
infer.py / area.py   (Phase 3) full-epoch inference, per-glacier area 2017 -> 2025, maps
```

COG overviews matter here: a 1 km tile at 0.1 m is ~300 MB, but reading it at 0.5 m pulls only the
0.4 m overview through HTTP range requests. Pilot (6 glaciers, 3 epochs) is roughly 1 GB on disk.
All Swiss glaciers at 3 epochs is estimated at 40-60 GB.

## 4. Phases and exit criteria

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 Data | Stages 01-03 on 6 pilot glaciers | Tiles open in QGIS, SGI2016 outlines line up with imagery |
| 1 Labels | Rasterize, chip, split by glacier | Class balance report, visual check of 20 random chips |
| 2 Baseline | U-Net RGB only, then RGB+DEM | IoU per class on val; debris IoU is the hard metric |
| 3 Test | Predict epoch 2021-2024, compare with SGI2023 | Area error per glacier vs GLAMOS, in % |
| 4 Scale | All Swiss glaciers, 3 epochs | Area-change table + maps |
| 5 Publish | Wiki chapters, public repo, LinkedIn | README with figures |

## 5. Hardware

- Data stages: CPU + network only.
- Training: P100 16 GB via Kubernetes + Kueue (ClusterQueue `p100`). U-Net/ResNet34, 512 px chips,
  batch ~16 with AMP fits comfortably. P100 has no tensor cores, so fp16 gives memory headroom more than speed.

## 6. Run Phase 0

srv-tesla-bme (IP: fill in)
```bash
mkdir -p /nfs/shared/projects/glacier-seg/data && cd ~/glacier-seg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# edit config.yaml -> paths.root if /data is not the right volume
python scripts/01_fetch_glamos.py
cat /nfs/shared/projects/glacier-seg/data/manifest/glamos_layers.txt      # check layer names + name column
python scripts/02_select_tiles.py
python scripts/03_download_tiles.py
```

## 7. Attribution

- Swiss Glacier Inventory: GLAMOS - Glacier Monitoring Switzerland, CC BY 4.0
  (SGI2016: Linsbauer et al. 2021, doi:10.3389/feart.2021.704189; SGI2023: doi:10.18750/inventory.sgi2023.r2026)
- SWISSIMAGE, swissALTI3D: Federal Office of Topography swisstopo
