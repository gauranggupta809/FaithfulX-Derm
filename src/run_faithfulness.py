"""RQ3: deletion / insertion faithfulness, with a RANDOM-ORDER control.

Deletion : progressively replace the most-attributed cells with a blurred version of the image;
           a faithful map makes the probability of the original predicted class fall FAST (low AUC).
Insertion: start from the blurred image and progressively add the most-attributed cells back;
           a faithful map makes the probability rise FAST (high AUC).
Control  : the same procedure with a random cell order. Faithful = clearly better than random.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import ImageFilter

from common import (get_device, load_checkpoint, load_meta, load_pil, preprocess,
                    seed_everything, select_images)
from xai import GradCAM, integrated_gradients

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--meta", default=None)
ap.add_argument("--ckpt", default="models/resnet18_v2.pt")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--n-images", type=int, default=200)
ap.add_argument("--ig-steps", type=int, default=32)
ap.add_argument("--n-random", type=int, default=5)
ap.add_argument("--out", default="results/faithfulness.csv")
ap.add_argument("--curves", default="results/curves.csv")
a = ap.parse_args()

GRID, CELL = 28, 8  # 28x28 cells of 8x8 pixels
FRACS = np.linspace(0, 1, 21)

seed_everything(42)
dev = get_device()
model, classes = load_checkpoint(a.ckpt, dev)
cam = GradCAM(model)
df = select_images(load_meta(a.data_root, a.meta, a.split), a.n_images)
rng = np.random.default_rng(0)


def auc(y):
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(FRACS)))


def masks_from_order(order):
    """order: cell indices, most important first -> (len(FRACS),1,224,224) masks (1 = cell selected)."""
    m = np.zeros((len(FRACS), GRID * GRID), np.float32)
    for i, f in enumerate(FRACS):
        m[i, order[: int(round(f * GRID * GRID))]] = 1
    m = torch.tensor(m).view(-1, 1, GRID, GRID)
    return F.interpolate(m, scale_factor=CELL, mode="nearest").to(dev)


@torch.no_grad()
def curves(x, x_fill, order, target):
    m = masks_from_order(order)
    dele = x * (1 - m) + x_fill * m
    ins = x_fill * (1 - m) + x * m
    pd_ = torch.softmax(model(dele), 1)[:, target].cpu().numpy()
    pi_ = torch.softmax(model(ins), 1)[:, target].cpu().numpy()
    return pd_, pi_


rows, crv = [], []
for k, (_, r) in enumerate(df.iterrows(), 1):
    img = load_pil(a.data_root, r["image_path"])
    x = preprocess(img).unsqueeze(0).to(dev)
    x_fill = preprocess(img.filter(ImageFilter.GaussianBlur(20))).unsqueeze(0).to(dev)
    with torch.no_grad():
        target = int(model(x).argmax())
    # random control (shared by both methods)
    rd, ri = [], []
    for _ in range(a.n_random):
        d, i = curves(x, x_fill, rng.permutation(GRID * GRID), target)
        rd.append(d), ri.append(i)
    rd, ri = np.mean(rd, 0), np.mean(ri, 0)
    maps = {"gradcam": cam(x, target), "ig": integrated_gradients(model, x, target, a.ig_steps)}
    for method, m in maps.items():
        cell = F.adaptive_avg_pool2d(m[None, None], GRID).flatten().cpu().numpy()
        order = np.argsort(-cell)
        d, i = curves(x, x_fill, order, target)
        rows.append({"image_id": r["isic_id"], "lesion_id": r["lesion_id"], "method": method,
                     "del_auc": auc(d), "ins_auc": auc(i), "rand_del_auc": auc(rd), "rand_ins_auc": auc(ri),
                     "del_gap": auc(rd) - auc(d), "ins_gap": auc(i) - auc(ri)})
        for kind, y in (("del", d), ("ins", i)):
            crv += [{"image_id": r["isic_id"], "method": method, "kind": kind, "frac": f, "prob": p}
                    for f, p in zip(FRACS, y)]
    for kind, y in (("del", rd), ("ins", ri)):
        crv += [{"image_id": r["isic_id"], "method": "random", "kind": kind, "frac": f, "prob": p}
                for f, p in zip(FRACS, y)]
    if k % 10 == 0:
        print(f"[{k}/{len(df)}]", flush=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
pd.DataFrame(crv).to_csv(a.curves, index=False)
print("Saved", a.out, "and", a.curves)
