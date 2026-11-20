"""Classification metrics. Used by train.py and runnable alone."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from common import LesionDataset, get_device, load_checkpoint, load_meta


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def report(y, p, classes, out_prefix):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    m = {"accuracy": accuracy_score(y, p),
         "balanced_accuracy": balanced_accuracy_score(y, p),
         "macro_f1": f1_score(y, p, average="macro", labels=labels, zero_division=0),
         "weighted_f1": f1_score(y, p, average="weighted", labels=labels, zero_division=0)}
    json.dump(m, open(f"{out_prefix}_metrics.json", "w"), indent=2)
    pd.DataFrame(confusion_matrix(y, p, labels=labels), index=classes, columns=classes
                 ).to_csv(f"{out_prefix}_confusion.csv")
    rep = classification_report(y, p, labels=labels, target_names=classes,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(f"{out_prefix}_classwise.csv")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--ckpt", default="models/resnet18_v2.pt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="results/classification")
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()
    if a.split == "test":
        print("WARNING: test split. Only do this ONCE, after all choices are final.")
    dev = get_device()
    model, classes = load_checkpoint(a.ckpt, dev)
    df = load_meta(a.data_root, a.meta, a.split)
    dl = DataLoader(LesionDataset(df, a.data_root, classes), batch_size=64, num_workers=a.num_workers)
    y, p = predict(model, dl, dev)
    print(report(y, p, classes, f"{a.out}_{a.split}"))
