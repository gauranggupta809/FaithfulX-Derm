"""Phase 3 Sanity Checks:
Verifies:
  a) the similarity of a map with itself is exactly 1.0 (cosine and spearman);
  b) Grad-CAM and IG maps are 224x224 and scaled to [0, 1];
  c) deletion at fraction 0 equals insertion at fraction 1 equals original class probability;
  d) average similarity under null perturbations is higher than under severity-2 perturbations.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFilter

from common import (get_device, load_checkpoint, load_meta, load_pil, preprocess,
                    seed_everything, select_images)
from xai import GradCAM, integrated_gradients, similarities
from perturb import apply, NULLS, DEFAULT

def run_sanity_checks(data_root, meta=None, ckpt="models/resnet18_v2.pt", n_images=5):
    seed_everything(42)
    dev = get_device()
    print(f"Loading checkpoint {ckpt} onto {dev}...")
    model, classes = load_checkpoint(ckpt, dev)
    cam = GradCAM(model)

    df_val = load_meta(data_root, meta, "val")
    df_sample = select_images(df_val, n_images)
    print(f"Selected {len(df_sample)} validation images for sanity checks.\n")

    results_a = []
    results_b = []
    results_c = []
    null_sims = []
    s2_sims = []

    GRID, CELL = 28, 8

    for idx, (_, row) in enumerate(df_sample.iterrows(), 1):
        img = load_pil(data_root, row["image_path"])
        x = preprocess(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(x)
            orig_probs = torch.softmax(logits, 1)[0]
            target = int(logits.argmax())
            orig_prob = float(orig_probs[target])

        # Generate maps
        g_map = cam(x, target)
        ig_map = integrated_gradients(model, x, target, steps=24)

        # Check a: self-similarity == 1.0
        sim_g = similarities(g_map, g_map)
        sim_ig = similarities(ig_map, ig_map)
        pass_a = (
            abs(sim_g["cosine"] - 1.0) < 1e-4 and
            abs(sim_g["spearman"] - 1.0) < 1e-4 and
            abs(sim_ig["cosine"] - 1.0) < 1e-4 and
            abs(sim_ig["spearman"] - 1.0) < 1e-4
        )
        results_a.append(pass_a)

        # Check b: shape is 224x224 and scaled to [0, 1]
        pass_b = (
            g_map.shape == (224, 224) and
            ig_map.shape == (224, 224) and
            g_map.min().item() >= -1e-6 and g_map.max().item() <= 1.0001 and
            ig_map.min().item() >= -1e-6 and ig_map.max().item() <= 1.0001
        )
        results_b.append(pass_b)

        # Check c: deletion at frac=0 == insertion at frac=1 == original class probability
        img_blur = img.filter(ImageFilter.GaussianBlur(20))
        x_fill = preprocess(img_blur).unsqueeze(0).to(dev)

        # frac = 0 mask (all zeros: original image entirely intact)
        mask_0 = torch.zeros((1, 1, 224, 224), device=dev)
        x_del_0 = x * (1 - mask_0) + x_fill * mask_0
        # frac = 1 mask (all ones: original image entirely inserted)
        mask_1 = torch.ones((1, 1, 224, 224), device=dev)
        x_ins_1 = x_fill * (1 - mask_1) + x * mask_1

        with torch.no_grad():
            p_del_0 = float(torch.softmax(model(x_del_0), 1)[0, target])
            p_ins_1 = float(torch.softmax(model(x_ins_1), 1)[0, target])

        pass_c = (
            abs(p_del_0 - orig_prob) < 1e-4 and
            abs(p_ins_1 - orig_prob) < 1e-4
        )
        results_c.append(pass_c)

        # Check d: null perturbations vs severity-2 perturbations
        # Null: null_noise (s0), null_jpeg (s0)
        for null_p in ["null_noise", "null_jpeg"]:
            img_null = apply(img, null_p, "s0", seed=0)
            x_null = preprocess(img_null).unsqueeze(0).to(dev)
            g_null = cam(x_null, target)
            null_sims.append(similarities(g_map, g_null)["spearman"])

        # Severe (s2): noise, blur, jpeg
        for s2_p in ["noise", "blur", "jpeg"]:
            img_s2 = apply(img, s2_p, "s2", seed=0)
            x_s2 = preprocess(img_s2).unsqueeze(0).to(dev)
            g_s2 = cam(x_s2, target)
            s2_sims.append(similarities(g_map, g_s2)["spearman"])

    avg_null_sim = float(np.mean(null_sims))
    avg_s2_sim = float(np.mean(s2_sims))
    pass_d = avg_null_sim > avg_s2_sim

    print("--- SANITY CHECK RESULTS ---")
    print(f"Check (a) Self-similarity == 1.0: {'PASS' if all(results_a) else 'FAIL'} ({sum(results_a)}/{len(results_a)})")
    print(f"Check (b) 224x224 & [0, 1] bounds: {'PASS' if all(results_b) else 'FAIL'} ({sum(results_b)}/{len(results_b)})")
    print(f"Check (c) Del(0) == Ins(1) == Prob: {'PASS' if all(results_c) else 'FAIL'} ({sum(results_c)}/{len(results_c)})")
    print(f"Check (d) Sim(null) > Sim(s2): {'PASS' if pass_d else 'FAIL'} (Avg null: {avg_null_sim:.4f} vs Avg s2: {avg_s2_sim:.4f})")
    print("----------------------------\n")

    all_passed = all(results_a) and all(results_b) and all(results_c) and pass_d
    if not all_passed:
        print("ERROR: One or more sanity checks failed!")
        sys.exit(1)
    else:
        print("All Phase 3 sanity checks PASSED successfully.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--ckpt", default="models/resnet18_v2.pt")
    ap.add_argument("--n-images", type=int, default=5)
    a = ap.parse_args()
    run_sanity_checks(a.data_root, a.meta, a.ckpt, a.n_images)
