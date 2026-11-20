"""Grad-CAM and Integrated Gradients (own implementation, no extra libraries).

Both return a 224x224 map scaled to [0, 1]. The target class is passed in explicitly
so you can keep it FIXED (the original predicted class) for original and perturbed images.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchvision.transforms.functional import gaussian_blur


def _norm(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.act, self.grad, self.on = None, None, False
        model.layer4.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        if self.on and out.requires_grad:
            self.act = out
            out.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, target):
        self.on = True
        x = x.clone().requires_grad_(True)  # guarantees a graph even if layer4 is frozen
        self.model.zero_grad()
        self.model(x)[0, target].backward()
        self.on = False
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return _norm(cam[0, 0].detach())


def integrated_gradients(model, x, target, steps=32):
    """x: 1x3xHxW normalized tensor. Baseline = zeros (the dataset-mean image)."""
    base = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)[1:].view(-1, 1, 1, 1)
    xs = (base + alphas * (x - base)).requires_grad_(True)
    out = model(xs)[:, target].sum()
    g = torch.autograd.grad(out, xs)[0].mean(0, keepdim=True)
    attr = ((x - base) * g).abs().sum(1, keepdim=True)  # |attribution|, summed over RGB
    attr = gaussian_blur(attr, kernel_size=[11, 11], sigma=[3.0, 3.0])  # tame pixel noise
    return _norm(attr[0, 0].detach())


def _pool(m, size=56):
    return F.adaptive_avg_pool2d(m[None, None].cpu(), size)[0, 0].numpy().ravel()


def similarities(a, b, top=0.2):
    """Cosine, Spearman rank correlation, and IoU of the top-20% regions."""
    a, b = _pool(a), _pool(b)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    rho = spearmanr(a, b).correlation
    k = int(top * len(a))
    ta, tb = np.zeros(len(a), bool), np.zeros(len(b), bool)
    ta[np.argpartition(-a, k)[:k]] = True
    tb[np.argpartition(-b, k)[:k]] = True
    iou = float((ta & tb).sum() / (ta | tb).sum())
    return {"cosine": cos, "spearman": float(rho) if rho == rho else 0.0, "iou_top20": iou}
