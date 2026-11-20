"""Shared helpers: data loading, model building, image selection."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG = 224

# turns an already-224x224 PIL image into a normalized tensor
preprocess = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta(data_root, meta=None, split=None):
    """Read master_metadata.csv. Windows backslashes in image_path are fixed."""
    meta_path = Path(meta) if meta else Path(data_root) / "master_metadata.csv"
    df = pd.read_csv(meta_path)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    if split:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def get_classes(data_root, meta=None):
    """Class order is fixed (alphabetical) so it is identical in every script."""
    return sorted(load_meta(data_root, meta)["class"].unique())


def load_pil(data_root, rel_path):
    img = Image.open(Path(data_root) / rel_path).convert("RGB")
    return img.resize((IMG, IMG), Image.BILINEAR)


class LesionDataset(Dataset):
    def __init__(self, df, data_root, classes, train=False):
        self.df = df.reset_index(drop=True)
        self.root = data_root
        self.c2i = {c: i for i, c in enumerate(classes)}
        aug = []
        if train:  # geometry-only augmentation (no colour jitter: it would hide brightness sensitivity)
            aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20)]
        self.tf = T.Compose([T.Resize((IMG, IMG))] + aug +
                            [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(Path(self.root) / r["image_path"]).convert("RGB")
        return self.tf(img), self.c2i[r["class"]]


def build_model(n_classes, pretrained=True, unfreeze_layer4=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True
    if unfreeze_layer4:
        for p in m.layer4.parameters():
            p.requires_grad = True
    return m


def set_train_mode(m, unfreeze_layer4=True):
    """Train mode, but keep frozen BatchNorm layers fixed."""
    m.train()
    frozen = [m.bn1, m.layer1, m.layer2, m.layer3] + ([] if unfreeze_layer4 else [m.layer4])
    for mod in frozen:
        mod.eval()


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck["classes"]


def select_images(df, n_images, seed=42):
    """Same sample for every experiment: one image per lesion, balanced across classes."""
    df = df.sample(frac=1, random_state=seed).drop_duplicates("lesion_id")
    per = max(1, n_images // df["class"].nunique())
    parts = [g.head(per) for _, g in df.groupby("class")]
    return pd.concat(parts).reset_index(drop=True)
