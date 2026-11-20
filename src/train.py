"""Fine-tune ResNet-18 (layer4 + new head). Best epoch is chosen by VALIDATION macro-F1."""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import (LesionDataset, build_model, get_classes, get_device, load_meta,
                    seed_everything, set_train_mode)
from evaluate import predict, report

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--meta", default=None)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--lr-backbone", type=float, default=1e-4)
ap.add_argument("--freeze-backbone", action="store_true", help="train only the head")
ap.add_argument("--no-pretrained", action="store_true", help="offline smoke tests only")
ap.add_argument("--train-limit", type=int, default=0, help="debug: use fewer images")
ap.add_argument("--num-workers", type=int, default=0)
ap.add_argument("--out", default="models/resnet18_v2.pt")
ap.add_argument("--results", default="results")
a = ap.parse_args()

seed_everything(42)
dev = get_device()
print("Device:", dev)
classes = get_classes(a.data_root, a.meta)
tr = load_meta(a.data_root, a.meta, "train")
va = load_meta(a.data_root, a.meta, "val")
if a.train_limit:
    tr = tr.sample(min(len(tr), a.train_limit), random_state=42)
    va = va.sample(min(len(va), max(100, a.train_limit // 4)), random_state=42)

unfreeze = not a.freeze_backbone
model = build_model(len(classes), pretrained=not a.no_pretrained, unfreeze_layer4=unfreeze).to(dev)

# class weights (square root of inverse frequency) soften the Nevus imbalance
counts = tr["class"].value_counts().reindex(classes).fillna(1).values
w = (counts.sum() / (len(classes) * counts)) ** 0.5
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

groups = [{"params": model.fc.parameters(), "lr": a.lr_head}]
if unfreeze:
    groups.append({"params": model.layer4.parameters(), "lr": a.lr_backbone})
opt = torch.optim.AdamW(groups, weight_decay=1e-4)

dl_tr = DataLoader(LesionDataset(tr, a.data_root, classes, train=True), batch_size=a.batch_size,
                   shuffle=True, num_workers=a.num_workers)
dl_va = DataLoader(LesionDataset(va, a.data_root, classes), batch_size=64, num_workers=a.num_workers)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.results).mkdir(parents=True, exist_ok=True)
hist, best = [], -1
for ep in range(1, a.epochs + 1):
    t0 = time.time()
    set_train_mode(model, unfreeze)
    tot, n = 0.0, 0
    for x, y in dl_tr:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y)
        n += len(y)
    model.eval()
    y_true, y_pred = predict(model, dl_va, dev)
    m = report(y_true, y_pred, classes, f"{a.results}/tmp_val")
    hist.append({"epoch": ep, "train_loss": tot / n, **m})
    print(f"epoch {ep}/{a.epochs} loss {tot/n:.3f} val macroF1 {m['macro_f1']:.3f} "
          f"bal.acc {m['balanced_accuracy']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if m["macro_f1"] > best:
        best = m["macro_f1"]
        torch.save({"state_dict": model.state_dict(), "classes": classes,
                    "unfreeze_layer4": unfreeze, "epoch": ep, "val_macro_f1": best}, a.out)
    pd.DataFrame(hist).to_csv(f"{a.results}/training_history.csv", index=False)

ck = torch.load(a.out, map_location=dev, weights_only=False)
model.load_state_dict(ck["state_dict"])
model.eval()
y_true, y_pred = predict(model, dl_va, dev)
print("BEST (validation):", report(y_true, y_pred, classes, f"{a.results}/classification_val"))
