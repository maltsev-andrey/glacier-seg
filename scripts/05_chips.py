"""Stage 05 (Phase 1): cut 512 px training chips, split by glacier, report class balance.

Each chip bundles everything the model sees for one 256 m square:
  img    uint8  (3, 512, 512)   RGB at 0.5 m
  dem    float32 (512, 512)     elevation, bilinear 2 m -> 0.5 m
  slope  float32 (512, 512)     degrees, Horn method on the 2 m DEM
  label  uint8  (512, 512)      0 bg / 1 ice / 2 debris / 255 ignore

Chip keep rule: a chip stays if >= min_glacier_frac of its valid pixels are
ice or debris; pure-background chips are kept at bg_keep_frac (deterministic
hash of the chip id, so reruns are identical).

Gap rule (README 5.4): gap = |image year - inventory acquisition year| per
glacier. Chips with gap > max_label_gap_years get gap_ok=False and are meant
to be excluded from training; they stay in the index for later use.

Output:
  chips/<split>/<glacier>_<year>_<tile>_<row><col>.npz
  manifest/chips_index.csv     one row per chip: split, glacier, tile, year, gap, fractions
  data/qc_chips/<glacier>_*.png  overlay previews for visual check
  manifest/class_balance.csv   per split x gap_ok: chip counts, pixel fractions
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

from common import load_config

LABEL_COLORS = {1: (0.2, 0.6, 1.0, 0.45), 2: (1.0, 0.8, 0.0, 0.45), 255: (1.0, 0.0, 0.0, 0.35)}


def read_resampled(path: Path, size: int, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(out_shape=(src.count, size, size), resampling=resampling)


def slope_deg(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Horn slope in degrees on the native-resolution DEM."""
    gy, gx = np.gradient(dem.astype(np.float32), res_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def keep_background(chip_id: str, frac: float) -> bool:
    h = int(hashlib.md5(chip_id.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < frac * 10_000


def save_qc(png: Path, img: np.ndarray, label: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    rgb = np.transpose(img, (1, 2, 0))
    for a in ax:
        a.imshow(rgb)
        a.set_xticks([]), a.set_yticks([])
    overlay = np.zeros((*label.shape, 4), dtype=np.float32)
    for cls, rgba in LABEL_COLORS.items():
        overlay[label == cls] = rgba
    ax[1].imshow(overlay)
    ax[0].set_title("RGB"), ax[1].set_title("label overlay")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def main():
    cfg = load_config()
    p, r, ph1 = cfg["paths"], cfg["raster"], cfg["phase1"]
    man = Path(p["manifest"])
    chips_dir = Path(p["chips"])
    qc_dir = Path(p["root"]) / "qc_chips"
    qc_dir.mkdir(parents=True, exist_ok=True)

    res, tile_m = r["target_res_m"], r["tile_size_m"]
    size = int(round(tile_m / res))
    cp, stride = ph1["chip_px"], ph1["chip_stride_px"]
    offsets = list(range(0, size - cp + 1, stride))
    split_of = {g: s for s, gs in ph1["split"].items() for g in gs}

    img_man = pd.read_csv(man / "tiles_image.csv")
    rows, qc_left = [], {}

    for n, t in enumerate(img_man.itertuples(), 1):
        glacier, split = t.glacier, split_of.get(t.glacier)
        if split is None:
            print(f"[WARN] {t.glacier} not in phase1.split, skipping")
            continue
        gap = abs(int(t.year) - int(ph1["acq_year"][glacier]))
        gap_ok = gap <= ph1["max_label_gap_years"]

        img_path = Path(p["raw_image"]) / str(t.year) / f"{t.tile}.tif"
        lab_path = Path(p["labels"]) / f"{t.tile}.tif"
        dem_path = Path(p["raw_dem"]) / f"{t.tile}.tif"
        if not (img_path.exists() and lab_path.exists() and dem_path.exists()):
            print(f"[WARN] missing input for {t.tile}/{t.year}, skipping")
            continue

        img = read_resampled(img_path, size, Resampling.nearest)[:3]
        label = read_resampled(lab_path, size, Resampling.nearest)[0]
        with rasterio.open(dem_path) as src:
            dem_native = src.read(1).astype(np.float32)
        slope_native = slope_deg(dem_native, r["dem_res_m"])
        f = size // dem_native.shape[0]
        dem = np.kron(dem_native, np.ones((f, f), dtype=np.float32)) if f > 1 else dem_native
        slope = np.kron(slope_native, np.ones((f, f), dtype=np.float32)) if f > 1 else slope_native

        out_dir = chips_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for yi, y0 in enumerate(offsets):
            for xi, x0 in enumerate(offsets):
                lab_c = label[y0:y0 + cp, x0:x0 + cp]
                valid = lab_c != 255
                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                frac_ice = float(((lab_c == 1) & valid).sum()) / n_valid
                frac_deb = float(((lab_c == 2) & valid).sum()) / n_valid
                chip_id = f"{glacier}_{t.year}_{t.tile}_{yi}{xi}"
                glacier_frac = frac_ice + frac_deb
                if glacier_frac < ph1["min_glacier_frac"] and not keep_background(chip_id, ph1["bg_keep_frac"]):
                    continue

                np.savez_compressed(
                    out_dir / f"{chip_id}.npz",
                    img=img[:, y0:y0 + cp, x0:x0 + cp],
                    dem=dem[y0:y0 + cp, x0:x0 + cp],
                    slope=slope[y0:y0 + cp, x0:x0 + cp],
                    label=lab_c,
                )
                rows.append({
                    "chip": chip_id, "split": split, "glacier": glacier, "tile": t.tile,
                    "year": int(t.year), "gap": gap, "gap_ok": gap_ok,
                    "frac_ice": round(frac_ice, 4), "frac_debris": round(frac_deb, 4),
                    "frac_bg": round(1 - frac_ice - frac_deb, 4),
                    "frac_ignore": round(1 - n_valid / lab_c.size, 4),
                })
                if glacier_frac > 0.2 and qc_left.setdefault(glacier, ph1["qc_per_glacier"]) > 0:
                    qc_left[glacier] -= 1
                    save_qc(qc_dir / f"{chip_id}.png", img[:, y0:y0 + cp, x0:x0 + cp], lab_c,
                            f"{chip_id}  split={split} gap={gap}")
        if n % 50 == 0:
            print(f"  [{n}/{len(img_man)}] tiles done, {len(rows)} chips")

    idx = pd.DataFrame(rows)
    idx.to_csv(man / "chips_index.csv", index=False)

    bal = (idx.groupby(["split", "gap_ok"])
              .agg(chips=("chip", "count"), ice=("frac_ice", "mean"),
                   debris=("frac_debris", "mean"), bg=("frac_bg", "mean"),
                   ignore=("frac_ignore", "mean"))
              .round(4))
    bal.to_csv(man / "class_balance.csv")
    print(f"\n{len(idx)} chips -> {chips_dir}")
    print("\nClass balance (mean pixel fraction per chip):")
    print(bal.to_string())
    train_usable = len(idx[(idx.split == "train") & idx.gap_ok])
    print(f"\nTrainable chips (split=train, gap_ok): {train_usable}")


if __name__ == "__main__":
    main()
