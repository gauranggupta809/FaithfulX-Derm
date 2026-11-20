"""RQ1 + RQ2: prediction stability and explanation stability under perturbations.

Protocol: explanations ALWAYS target the ORIGINAL predicted class (fixed target),
for both the original and the perturbed image.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

import perturb
from common import (get_device, load_checkpoint, load_meta, load_pil, preprocess,
                    seed_everything, select_images)
from xai import GradCAM, integrated_gradients, similarities

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--meta", default=None)
ap.add_argument("--ckpt", default="models/resnet18_v2.pt")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--n-images", type=int, default=200)
ap.add_argument("--perturbs", default=",".join(perturb.DEFAULT))
ap.add_argument("--ig-steps", type=int, default=32)
ap.add_argument("--out", default="results/stability.csv")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

seed_everything(42)
dev = get_device()
model, classes = load_checkpoint(a.ckpt, dev)
cam = GradCAM(model)
df = select_images(load_meta(a.data_root, a.meta, a.split), a.n_images)
conds = perturb.conditions(a.perturbs.split(","))
print(f"{len(df)} images x {len(conds)} conditions x 2 methods | split={a.split}")

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set(pd.read_csv(out)["image_id"]) if (a.resume and out.exists()) else set()
if not a.resume and out.exists():
    out.unlink()


def probs_of(x):
    with torch.no_grad():
        return torch.softmax(model(x), 1)[0].cpu()


t0, k = time.time(), 0
for _, r in df.iterrows():
    if r["isic_id"] in done:
        continue
    img = load_pil(a.data_root, r["image_path"])
    x0 = preprocess(img).unsqueeze(0).to(dev)
    p0 = probs_of(x0)
    pred0 = int(p0.argmax())
    maps0 = {"gradcam": cam(x0, pred0), "ig": integrated_gradients(model, x0, pred0, a.ig_steps)}
    rows = []
    for name, level in conds:
        xp = preprocess(perturb.apply(img, name, level, seed=0)).unsqueeze(0).to(dev)
        pp = probs_of(xp)
        predp = int(pp.argmax())
        mp = {"gradcam": cam(xp, pred0), "ig": integrated_gradients(model, xp, pred0, a.ig_steps)}
        for method in ("gradcam", "ig"):
            rows.append({
                "image_id": r["isic_id"], "lesion_id": r["lesion_id"], "true_class": r["class"],
                "orig_pred": classes[pred0], "orig_correct": int(classes[pred0] == r["class"]),
                "orig_conf": float(p0[pred0]), "perturb": name, "level": level, "method": method,
                "pert_pred": classes[predp], "flipped": int(predp != pred0),
                "conf_orig_class_after": float(pp[pred0]),
                "conf_change": float(pp[pred0] - p0[pred0]),
                **similarities(maps0[method], mp[method])})
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    k += 1
    el = time.time() - t0
    print(f"[{k}/{len(df)-len(done)}] {r['isic_id']}  elapsed {el/60:.1f} min, "
          f"est. total {el/k*(len(df)-len(done))/60:.0f} min", flush=True)
print("Saved", out)
