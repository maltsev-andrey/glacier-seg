"""Stage 06 (Phase 2): U-Net baseline on the chips, IoU per class on the validation glacier.

Design (README 5.4 / 6.2):
- train = split 'train' AND gap_ok; val = split 'val' AND gap_ok (Zmutt, debris-heavy)
- class weights: median-frequency balancing computed from the train chips index,
  capped at max_class_weight; loss = weighted CE + Dice, both ignoring class 255
- brightness varies ~14% between flight years -> ImageNet normalization plus
  brightness/contrast jitter so the model cannot learn "dark = 2017"
- input_set 'rgb' (3 ch, ImageNet encoder) or 'rgb_terrain' (5 ch: + elevation
  standardized per chip, + slope/45); smp replicates conv1 weights for extra channels

Output (models_dir/<run>/):
  best.pt, last.pt          model + optimizer + epoch + val metrics
  log.csv                   per epoch: losses, per-class IoU, mIoU, lr
  preds/*.png               RGB | truth | prediction triptychs for 8 val chips
"""
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import load_config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_CLASSES, IGNORE = 3, 255


# ---------- data ----------

class ChipDataset(Dataset):
    def __init__(self, chips_dir: Path, index: pd.DataFrame, input_set: str, augment: bool):
        self.files = [chips_dir / r.split / f"{r.chip}.npz" for r in index.itertuples()]
        self.input_set = input_set
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        img = d["img"].astype(np.float32) / 255.0          # (3,H,W)
        label = d["label"].astype(np.int64)                # (H,W)
        extra = None
        if self.input_set == "rgb_terrain":
            dem = (d["dem"] - d["dem"].mean()) / 300.0     # per-chip center, fixed scale
            slope = d["slope"] / 45.0 - 1.0
            extra = np.stack([dem, slope]).astype(np.float32)   # (2,H,W)

        if self.augment:
            # geometry applied jointly to image, terrain and label; photometry to RGB only
            k = random.randrange(4)
            if k:
                img, label = np.rot90(img, k, (1, 2)), np.rot90(label, k)
                extra = np.rot90(extra, k, (1, 2)) if extra is not None else None
            if random.random() < 0.5:
                img, label = img[:, :, ::-1], label[:, ::-1]
                extra = extra[:, :, ::-1] if extra is not None else None
            if random.random() < 0.5:
                img, label = img[:, ::-1, :], label[::-1, :]
                extra = extra[:, ::-1, :] if extra is not None else None
            img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)

        x = (torch.from_numpy(np.ascontiguousarray(img)) - IMAGENET_MEAN) / IMAGENET_STD
        if extra is not None:
            x = torch.cat([x, torch.from_numpy(np.ascontiguousarray(extra))])
        return x, torch.from_numpy(np.ascontiguousarray(label))


# ---------- loss / metrics ----------

def dice_loss(logits, target, eps=1e-6):
    valid = target != IGNORE
    t = torch.where(valid, target, torch.zeros_like(target))
    one_hot = F.one_hot(t, N_CLASSES).permute(0, 3, 1, 2).float() * valid[:, None]
    prob = logits.softmax(1) * valid[:, None]
    inter = (prob * one_hot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def confusion(logits, target, cm):
    pred = logits.argmax(1)
    valid = target != IGNORE
    idx = target[valid] * N_CLASSES + pred[valid]
    cm += torch.bincount(idx, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def iou_from_cm(cm):
    inter = cm.diag().float()
    union = cm.sum(0).float() + cm.sum(1).float() - inter
    return (inter / union.clamp(min=1)).tolist()


# ---------- main ----------

def main():
    cfg = load_config()
    p, ph1, ph2 = cfg["paths"], cfg["phase1"], cfg["phase2"]
    torch.manual_seed(ph2["seed"]), random.seed(ph2["seed"]), np.random.seed(ph2["seed"])

    import segmentation_models_pytorch as smp

    chips_dir = Path(p["chips"])
    repo = Path(p["root"]).parent
    models_dir = Path(ph2["models_dir"]) if Path(ph2["models_dir"]).is_absolute() else repo / ph2["models_dir"]
    run = f"unet_{ph2['encoder']}_{ph2['input_set']}"
    out = models_dir / run
    (out / "preds").mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(Path(p["manifest"]) / "chips_index.csv")
    tr_idx = idx[(idx.split == "train") & idx.gap_ok]
    va_idx = idx[(idx.split == "val") & idx.gap_ok]
    if tr_idx.empty or va_idx.empty:
        raise SystemExit("empty train or val set — check phase1.split and gap_ok in chips_index.csv")

    # median-frequency class weights from the train index
    freq = np.array([tr_idx.frac_bg.mean(), tr_idx.frac_ice.mean(), tr_idx.frac_debris.mean()])
    weights = np.minimum(np.median(freq) / freq, ph2["max_class_weight"])
    print(f"train {len(tr_idx)} chips, val {len(va_idx)} chips ({sorted(va_idx.glacier.unique())})")
    print(f"class freq bg/ice/debris: {freq.round(3)} -> CE weights {weights.round(2)}")

    in_ch = 3 if ph2["input_set"] == "rgb" else 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(ph2["encoder"], encoder_weights=ph2.get("encoder_weights", "imagenet"),
                     in_channels=in_ch, classes=N_CLASSES).to(device)

    dl = dict(batch_size=ph2["batch_size"], num_workers=ph2["num_workers"],
              pin_memory=(device == "cuda"), persistent_workers=ph2["num_workers"] > 0)
    tr_dl = DataLoader(ChipDataset(chips_dir, tr_idx, ph2["input_set"], ph2["augment"]),
                       shuffle=True, drop_last=True, **dl)
    va_dl = DataLoader(ChipDataset(chips_dir, va_idx, ph2["input_set"], False), **dl)

    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                             ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=ph2["lr"], weight_decay=ph2["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ph2["epochs"], eta_min=ph2["lr"] / 50)
    scaler = torch.amp.GradScaler(enabled=ph2["amp"] and device == "cuda")

    best_miou, since_best = -1.0, 0
    with open(out / "log.csv", "w", newline="") as flog:
        log = csv.writer(flog)
        log.writerow(["epoch", "train_loss", "val_loss", "iou_bg", "iou_ice", "iou_debris", "miou", "lr", "sec"])

        for epoch in range(1, ph2["epochs"] + 1):
            t0 = time.time()
            model.train()
            tr_loss, nb = 0.0, 0
            for x, y in tr_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device, enabled=scaler.is_enabled()):
                    logits = model(x)
                    loss = ce(logits, y) + ph2["dice_weight"] * dice_loss(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt), scaler.update()
                tr_loss += loss.item()
                nb += 1
                if ph2["limit_train_batches"] and nb >= ph2["limit_train_batches"]:
                    break
            sched.step()

            model.eval()
            va_loss, nvb = 0.0, 0
            cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)
            with torch.no_grad():
                for x, y in va_dl:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with torch.amp.autocast(device, enabled=scaler.is_enabled()):
                        logits = model(x)
                        va_loss += (ce(logits, y) + ph2["dice_weight"] * dice_loss(logits, y)).item()
                    confusion(logits.float(), y, cm)
                    nvb += 1
            ious = iou_from_cm(cm.cpu())
            miou = sum(ious) / len(ious)
            sec = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            log.writerow([epoch, round(tr_loss / max(nb, 1), 4), round(va_loss / max(nvb, 1), 4),
                          *[round(v, 4) for v in ious], round(miou, 4), f"{lr_now:.2e}", round(sec, 1)])
            flog.flush()
            print(f"[{epoch:3d}/{ph2['epochs']}] loss {tr_loss/max(nb,1):.3f}/{va_loss/max(nvb,1):.3f}"
                  f"  IoU bg {ious[0]:.3f} ice {ious[1]:.3f} debris {ious[2]:.3f}"
                  f"  mIoU {miou:.3f}  {sec:.0f}s")

            state = dict(epoch=epoch, model=model.state_dict(), opt=opt.state_dict(),
                         ious=ious, miou=miou, config=ph2)
            torch.save(state, out / "last.pt")
            if miou > best_miou:
                best_miou, since_best = miou, 0
                torch.save(state, out / "best.pt")
            else:
                since_best += 1
                if since_best >= ph2["early_stop_patience"]:
                    print(f"early stop: no val mIoU gain for {since_best} epochs")
                    break

    # prediction triptychs from the best checkpoint
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    model.eval()
    save_preds(model, va_dl, device, out / "preds", n=8)
    print(f"\nbest mIoU {best_miou:.4f} -> {out}")


def save_preds(model, loader, device, preds_dir, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([(0.3, 0.3, 0.3), (0.2, 0.6, 1.0), (1.0, 0.8, 0.0)])
    done = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for b in range(x.shape[0]):
                if done >= n:
                    return
                rgb = (x[b, :3] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
                truth = y[b].numpy().astype(float)
                truth[truth == IGNORE] = np.nan
                fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
                ax[0].imshow(rgb), ax[0].set_title("RGB")
                ax[1].imshow(truth, cmap=cmap, vmin=0, vmax=2), ax[1].set_title("truth (nan=ignore)")
                ax[2].imshow(pred[b].numpy(), cmap=cmap, vmin=0, vmax=2), ax[2].set_title("prediction")
                for a in ax:
                    a.set_xticks([]), a.set_yticks([])
                fig.tight_layout()
                fig.savefig(preds_dir / f"val_{done:02d}.png", dpi=110)
                plt.close(fig)
                done += 1


if __name__ == "__main__":
    main()
