"""Stage 07 (Phase 3): sliding-window inference over full tiles.

Loads models_dir/<phase3.run>/best.pt and predicts every (tile, year) pair in
the Phase 0 manifest (optionally restricted by phase3.years). The checkpoint
carries its own phase2 config, so input channels and encoder are reconstructed
from it — no risk of a config/weights mismatch.

512 px windows, 50 % overlap, softmax averaged across overlapping windows.
Normalization is identical to training: ImageNet stats on RGB, per-window
DEM centering, slope/45-1.

Output:
  predictions/<run>/<year>/<tile>.tif   uint8 class raster, 0.5 m, EPSG:2056
    (0 bg, 1 clean ice, 2 debris; georeferenced -> drop into QGIS over the
     swisstopo basemap for visual QC)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from common import load_config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
N_CLASSES = 3


def window_offsets(size: int, win: int, stride: int) -> list[int]:
    offs = list(range(0, size - win + 1, stride))
    if offs[-1] != size - win:
        offs.append(size - win)
    return offs


def slope_deg(dem: np.ndarray, res_m: float) -> np.ndarray:
    gy, gx = np.gradient(dem.astype(np.float32), res_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def main():
    cfg = load_config()
    p, r, ph1, ph2, ph3 = cfg["paths"], cfg["raster"], cfg["phase1"], cfg["phase2"], cfg["phase3"]
    repo = Path(p["root"]).parent
    resolve = lambda v: Path(v) if Path(v).is_absolute() else repo / v

    import segmentation_models_pytorch as smp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(resolve(ph2["models_dir"]) / ph3["run"] / "best.pt",
                      map_location=device, weights_only=False)
    tcfg = ckpt["config"]
    in_ch = 3 if tcfg["input_set"] == "rgb" else 5
    model = smp.Unet(tcfg["encoder"], encoder_weights=None,
                     in_channels=in_ch, classes=N_CLASSES).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"run {ph3['run']}: epoch {ckpt['epoch']}, val mIoU {ckpt['miou']:.4f}, "
          f"{tcfg['input_set']} ({in_ch} ch), device {device}")

    res, tile_m = r["target_res_m"], r["tile_size_m"]
    size = int(round(tile_m / res))
    win, stride, bs = ph1["chip_px"], ph3["window_stride_px"], ph3["batch_size"]
    offs = window_offsets(size, win, stride)
    windows = [(y, x) for y in offs for x in offs]

    man = pd.read_csv(Path(p["manifest"]) / "tiles_image.csv")
    if ph3["years"]:
        man = man[man.year.isin(ph3["years"])]
    pairs = man[["tile", "year"]].drop_duplicates().itertuples(index=False)
    pairs = sorted(pairs)
    out_root = resolve(ph3["predictions_dir"]) / ph3["run"]
    print(f"{len(windows)} windows/tile, {len(man[['tile','year']].drop_duplicates())} tile-years -> {out_root}")

    amp = device == "cuda"
    done = 0
    for tile, year in pairs:
        img_path = Path(p["raw_image"]) / str(year) / f"{tile}.tif"
        dem_path = Path(p["raw_dem"]) / f"{tile}.tif"
        if not (img_path.exists() and dem_path.exists()):
            print(f"[WARN] missing input for {tile}/{year}, skipping")
            continue

        with rasterio.open(img_path) as src:
            img = src.read(out_shape=(src.count, size, size), resampling=Resampling.nearest)[:3]
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        terrain = None
        if in_ch == 5:
            with rasterio.open(dem_path) as src:
                dem_n = src.read(1).astype(np.float32)
            slope_n = slope_deg(dem_n, r["dem_res_m"])
            f = size // dem_n.shape[0]
            dem = np.kron(dem_n, np.ones((f, f), dtype=np.float32)) if f > 1 else dem_n
            slope = np.kron(slope_n, np.ones((f, f), dtype=np.float32)) if f > 1 else slope_n
            terrain = np.stack([dem, slope / 45.0 - 1.0])

        prob = np.zeros((N_CLASSES, size, size), dtype=np.float32)
        hits = np.zeros((size, size), dtype=np.float32)
        for i in range(0, len(windows), bs):
            batch, coords = [], windows[i:i + bs]
            for y0, x0 in coords:
                x = img[:, y0:y0 + win, x0:x0 + win]
                if terrain is not None:
                    t = terrain[:, y0:y0 + win, x0:x0 + win].copy()
                    t[0] -= t[0].mean()                      # per-window DEM centering, as in training
                    t[0] /= 300.0
                    x = np.concatenate([x, t])
                batch.append(x)
            xb = torch.from_numpy(np.stack(batch)).to(device)
            with torch.no_grad(), torch.amp.autocast(device, enabled=amp):
                pb = model(xb).softmax(1).float().cpu().numpy()
            for (y0, x0), pw in zip(coords, pb):
                prob[:, y0:y0 + win, x0:x0 + win] += pw
                hits[y0:y0 + win, x0:x0 + win] += 1

        pred = (prob / hits).argmax(0).astype(np.uint8)

        e_km, n_km = (int(v) for v in tile.split("-"))
        out_dir = out_root / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f"{tile}.part"
        with rasterio.open(
            tmp, "w", driver="GTiff", width=size, height=size, count=1, dtype="uint8",
            crs="EPSG:2056", transform=from_origin(e_km * tile_m, (n_km + 1) * tile_m, res, res),
            tiled=True, blockxsize=512, blockysize=512, compress="LZW",
        ) as dst:
            dst.write(pred, 1)
        tmp.replace(out_dir / f"{tile}.tif")

        done += 1
        if done % 50 == 0:
            print(f"  [{done}] {tile}/{year}")
    print(f"done: {done} tile-years")


if __name__ == "__main__":
    main()
