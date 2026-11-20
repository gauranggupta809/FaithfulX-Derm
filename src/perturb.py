"""Controlled image perturbations (applied to the 224x224 PIL image, BEFORE normalization).

Two NULL conditions (almost-invisible changes) are included. They tell you how much the
explanation changes even for a harmless edit, so real instability can be judged against them.
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _brightness(img, f, seed):
    return ImageEnhance.Brightness(img).enhance(f)


def _contrast(img, f, seed):
    return ImageEnhance.Contrast(img).enhance(f)


def _blur(img, r, seed):
    return img.filter(ImageFilter.GaussianBlur(r))


def _noise(img, std, seed):  # std on the 0-255 pixel scale
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32) + rng.normal(0, std, (img.height, img.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _jpeg(img, q, seed):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _colorshift(img, k, seed):  # white-balance style: more red, less blue
    a = np.asarray(img).astype(np.float32)
    a[..., 0] *= 1 + k
    a[..., 2] *= 1 - k
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


LEVELS = {
    "brightness": (_brightness, {"s1": 1.25, "s2": 1.5}),
    "contrast": (_contrast, {"s1": 0.75, "s2": 0.5}),
    "blur": (_blur, {"s1": 0.7, "s2": 1.4}),
    "noise": (_noise, {"s1": 8, "s2": 16}),
    "jpeg": (_jpeg, {"s1": 60, "s2": 30}),
    "colorshift": (_colorshift, {"s1": 0.10, "s2": 0.20}),
    "null_noise": (_noise, {"s0": 1.0}),
    "null_jpeg": (_jpeg, {"s0": 95}),
}
DEFAULT = ["brightness", "contrast", "blur", "noise", "jpeg", "colorshift"]
NULLS = ["null_noise", "null_jpeg"]


def apply(img, name, level, seed=0):
    fn, params = LEVELS[name]
    return fn(img, params[level], seed)


def conditions(names):
    """List of (name, level) incl. the null conditions."""
    out = []
    for n in names + NULLS:
        out += [(n, lv) for lv in LEVELS[n][1]]
    return out
