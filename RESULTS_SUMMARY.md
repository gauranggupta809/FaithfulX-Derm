# Results Summary — FaithfulX-Derm

> Plain-English summary of all findings. Every number comes from a CSV file in `results/`.  
> Written for: undergraduate audience, no prior XAI knowledge required.

---

## What We Did (In One Paragraph)

We trained a ResNet-18 neural network to classify skin lesion images from the HAM10000 dataset into 7 categories. We then applied two popular explanation methods — **Grad-CAM** and **Integrated Gradients (IG)** — and asked four questions: (1) Does the *prediction* change when we distort the image slightly? (2) Does the *explanation heatmap* change? (3) Are the explanations actually pointing at meaningful pixels? (4) Can explanation instability tell us when the model is making a mistake?

---

## Model Performance

| Metric | Value |
|---|---|
| Validation accuracy | 79.31% |
| Macro-F1 | 0.585 |
| Balanced accuracy | 0.583 |

> Source: `results/classification_val_metrics.json`

The model is reasonably good at the dominant class (Nevus, F1 = 0.903) but weaker on rare classes like Dermatofibroma (F1 = 0.270). This matters because explanation quality may vary by class.

---

## RQ1 — How Often Does the Prediction Flip Under Distortion?

**Question**: If we apply image distortions (noise, blur, JPEG compression, brightness change, colour shift), how often does the model give a *different* answer than it did on the clean image?

**Key Numbers** (source: `results/summary_stability.csv`):

| Distortion | Mild (s1) flip rate | Strong (s2) flip rate |
|---|---|---|
| Gaussian noise | **47.6%** | **63.3%** |
| JPEG compression | 33.7% | 52.4% |
| Brightness change | 24.7% | 41.6% |
| Blur | 14.5% | 35.5% |
| Colour shift | 14.5% | 25.3% |
| Null (control) | 3.6% | — |

**What this means**: Even mild noise causes the model to change its answer almost half the time. Notably, the `null_jpeg` (q=95) control is not inert — it flips 16.3% of validation predictions (10.3% on test), compared to 3.6% (2.4% on test) for `null_noise` (σ=1). This unexpected sensitivity to q=95 re-encoding was not investigated further and may reflect low-confidence predictions near decision boundaries.

**What it does NOT mean**: The model is broken. 79% accuracy on a 7-class problem is competitive. But clinicians should be aware that confidence alone does not guarantee robustness.

---

## RQ2 — Do the Explanation Heatmaps Change When the Image Changes?

**Question**: Even when the prediction stays the same, does the highlighted region (the "explanation") shift around under distortion?

We use **Spearman rank correlation** between the original and perturbed heatmaps. A score of 1.0 = perfectly identical; 0.0 = no relationship. We also compared against a **null baseline** (σ=1 noise control) to know what "natural fluctuation" looks like.

**Key Numbers** (source: `results/summary_stability.csv`):

| Method | Null baseline (`null_noise`) | Noise s1 Spearman | Noise s2 Spearman |
|---|---|---|---|
| Grad-CAM | 0.995 | 0.696 | **0.551** |
| IG | 0.976 | 0.816 | **0.739** |

**What this means**:
- Both methods show real instability beyond the null baseline under strong noise.
- **H2 was partly supported**: Both methods changed far more under severity-2 perturbations than under the null controls, but IG was not less stable than Grad-CAM. Under severe noise (noise, severity 2), the Grad-CAM map was all zero for 28.3% of validation images (17.6% on test), and only for images whose prediction had flipped. The model stopped supporting the original class, so Grad-CAM, which shows only positive evidence, had nothing to display and the map became all zero. It happened only in images whose prediction had flipped. IG has no ReLU, so it does not collapse. If these cases are scored as zero similarity, Integrated Gradients appears more stable on Spearman correlation. If they are excluded, the two methods are statistically indistinguishable (Grad-CAM 0.769 versus IG 0.739 on validation, with overlapping 95% confidence intervals). IG is higher on cosine similarity (0.925 vs. 0.619), and Grad-CAM is higher on top-20% region overlap (`iou_top20` 0.494 vs. 0.428 under noise s2; 12-condition mean 0.664 vs. 0.559). The ranking therefore depends on the metric and on how zero maps are handled, and we do not claim that either method is more stable. The two methods also differ in map resolution and smoothing, so the comparison is not like for like.
- **Zero-map flip status and confidence** (source: `results/stability.csv`, `results_test/stability.csv`): Across the 12 real perturbations, **100.0% of Grad-CAM zero-map rows have `flipped == 1`** (60/60 on validation; 37/37 on test), compared with 31.83% (615/1,932) of non-zero rows on validation and 31.55% (613/1,943) on test. Mean `conf_orig_class_after` is **0.000524** for zero-map rows versus **0.636448** for non-zero rows on validation (**0.000288** vs. **0.644600** on test). Among non-zero-map Grad-CAM rows at `noise_s2`, mean Spearman is **0.8480** (n=61) when `flipped == 0` and **0.6857** (n=58) when `flipped == 1` on validation (**0.8640** [n=58] when `flipped == 0` and **0.6532** [n=78] when `flipped == 1` on test).
- Colour shift causes the least disruption to explanations (Grad-CAM s2 Spearman = 0.889; IG = 0.898).

**H2 verdict**: *Partly supported.* Both methods changed far more under severity-2 perturbations than under the null controls, but IG was not less stable than Grad-CAM. Under severe noise (noise, severity 2), the Grad-CAM map was all zero for 28.3% of validation images (17.6% on test), and only for images whose prediction had flipped. If these cases are scored as zero similarity, Integrated Gradients appears more stable on Spearman correlation. If they are excluded, the two methods are statistically indistinguishable (Grad-CAM 0.769 versus IG 0.739 on validation, with overlapping 95% confidence intervals). IG is higher on cosine similarity, and Grad-CAM is higher on top-20% region overlap. The ranking therefore depends on the metric and on how zero maps are handled, and we do not claim that either method is more stable. The two methods also differ in map resolution and smoothing, so the comparison is not like for like.

---

## RQ3 — Do the Explanations Actually Point at Important Pixels?

**Question**: Are the pixels the explanation highlights actually the ones the model relies on? This is called *faithfulness*.

**Method**: We pool attributions to a 28×28 grid of 8×8-pixel cells and use two tests across 21 steps (0%, 5%, ..., 100%) with a Gaussian-blurred (σ=20) baseline fill:
- **Deletion test** — replace the top-highlighted cells with the blurred fill and see how much the confidence drops. A big drop = the explanation found real signal.
- **Insertion test** — start from the blurred fill, restore only the top-highlighted cells, and see how much confidence rises. A big rise = the explanation found real signal.

We compare each method's score to a **random deletion baseline** (masking random cells). A larger gap = more faithful.

**Key Numbers** (source: `results/faithfulness_summary.csv`):

| Method | Deletion gap (↑ better) | 95% CI | Insertion gap (↑ better) | 95% CI | % images beating random |
|---|---|---|---|---|---|
| Grad-CAM | **0.079** | [0.044, 0.114] | **0.157** | [0.121, 0.190] | 64.5% |
| IG | 0.056 | [0.031, 0.080] | 0.100 | [0.072, 0.130] | 60.8% |

**What this means**: Both methods are better than random, with 95% confidence intervals excluding zero — they are genuinely pointing at regions the model uses. Grad-CAM's gap is larger. However, the absolute gaps are modest, and roughly 35 to 42% of images show no advantage over random deletion (validation: Grad-CAM 35.5%, IG 39.2%; test: Grad-CAM 37.6%, IG 41.8%). We found no detectable association between per-image stability and faithfulness (Spearman about 0.04, n=166).

**H3 verdict**: *Supported.* Both methods beat the random baseline. Grad-CAM's faithfulness gap is larger.

---

## RQ4 — Can Explanation Instability Warn Us When the Model Is Wrong?

**Question**: When the explanation heatmap is highly unstable across perturbations, is the model more likely to have made a wrong prediction? If yes, instability could act as a "red flag" detector.

We measure this with **AUROC** (Area Under the ROC Curve). An AUROC of 1.0 = perfect detector; 0.5 = random guess.

**Key Numbers** (source: `results/instability_vs_error.csv`, `results/added_value_over_confidence.csv`):

| Signal | AUROC | 95% CI |
|---|---|---|
| Grad-CAM instability | 0.616 | [0.523, 0.690] |
| IG instability | 0.473 | [0.375, 0.568] |
| **Low confidence (reference)** | **0.737** | **[0.662, 0.810]** |

| Method | Confidence-only AUROC | Conf + Instability AUROC | Added gain |
|---|---|---|---|
| Grad-CAM | 0.719 | 0.709 | **−0.010** |
| IG | 0.719 | 0.719 | **+0.0002** |

**What this means**:
- Grad-CAM instability is a *weak* error signal (AUROC 0.616 [0.523, 0.690]).
- IG instability (AUROC 0.473 [0.375, 0.568] on validation; 0.460 [0.359, 0.544] on test) is **indistinguishable from chance** because its 95% confidence intervals include 0.5.
- Simply looking at the model's own confidence (AUROC 0.737 [0.662, 0.810]) is a stronger error detector than either explanation-based signal.
- Adding explanation instability to confidence brings *no detectable improvement* (gain ≈ 0 for both methods).

**H4 verdict**: *Not supported.* In this setting (one model, one dataset, 60 validation errors) we found no evidence that instability adds information beyond confidence.

---

## Extra — Which Disease Classes Are Most Unstable?

(Source: `results/per_class_stability.csv` — Grad-CAM rows)

| Class | Sample size (n) | Flip rate | Grad-CAM Spearman | IG Spearman | Clinical / methodological note |
|---|---|---|---|---|---|
| `akiec` — Actinic keratosis | 28 | 44.0% | 0.693 (0.744 excl. zero maps) | 0.836 | Precancerous; lowest under Grad-CAM |
| `bcc` — Basal cell carcinoma | 28 | 39.0% | 0.724 (0.769 excl. zero maps) | **0.876** | Malignant; second lowest under Grad-CAM, highest under IG |
| `bkl` — Benign keratosis | 28 | 44.3% | 0.815 | 0.848 | Highest flip rate on validation |
| `mel` — Melanoma | 28 | 28.9% | **0.926** | 0.825 | Malignant; highly stable under Grad-CAM |
| `nv` — Nevus | 28 | 12.5% | 0.921 | 0.811 | Most common; highly stable under Grad-CAM, lowest under IG |
| `vasc` — Vascular lesion | 15 ⚠️ | 33.3% | 0.875 | 0.868 | Rare class (n < 20, unreliable) |
| `df` — Dermatofibroma | 11 ⚠️ | 36.4% | 0.787 | 0.838 | Rare class (n < 20, unreliable) |

**What this means**: Under Grad-CAM, `akiec` and `bcc` have the lowest explanation rank stability (Spearman 0.693, 0.724). However, four qualifications must be stated:
1. **Does not hold for IG**: Under Integrated Gradients, `bcc` has the highest validation Spearman (0.876) and `nv`/`mel` have the lowest (0.811, 0.825).
2. **Melanoma is highly stable under Grad-CAM**: `mel` has the highest validation Grad-CAM Spearman (0.926; test 0.880).
3. **Partly affected by Grad-CAM zero maps**: 43 of the 60 validation Grad-CAM zero-map rows occur in `akiec` (23) and `bcc` (20); excluding zero maps raises their Grad-CAM Spearman to 0.744 and 0.769, though they remain the two lowest classes under Grad-CAM.
4. **Rare classes are underpowered**: `df` (n=11) and `vasc` (n=15 val, n=14 test) are below n=20 and unreliable.

Also note that on raw prediction flip rate `bkl` is highest (44.3%), above `akiec` (44.0%) and `bcc` (39.0%). Flip rate measures classifier boundary sensitivity rather than explanation stability.

---

## Summary Table

| Research Question | Hypothesis | Verdict | Directionally consistent on test? |
|---|---|---|---|
| RQ1: Prediction stability | Noise most destabilising | ✅ Supported for predictions (for explanations it holds only when all-zero Grad-CAM maps are scored as zero similarity) | ✅ Yes (noise s2 flip 0.648) |
| RQ2: Explanation stability | IG worse than Grad-CAM | ⚠️ Partly supported — both changed far more under s2 than null, but IG was not less stable than Grad-CAM; ranking depends on metric and on how Grad-CAM zero maps are handled | ✅ Yes (excl. zero maps, noise s2 Spearman is 0.743 GC vs 0.736 IG; IG higher on cosine, GC higher on top-20% IoU) |
| RQ3: Faithfulness | Both methods beat random | ✅ Supported | ✅ Yes (all gaps positive, 95% CI > 0) |
| RQ4: Error detection value | Instability adds value over confidence | ❌ Not supported (no evidence of added value over confidence) | ✅ Yes (confidence 0.701 > 0.629 > 0.460 [indistinguishable from chance]) |

---

## Held-Out Test Split Confirmation

Everything above was measured on the validation split. Phases 4–6 were repeated once on the held-out test split (165 images, all 7 classes) after all design choices were frozen.

| Quantity | Validation | Test |
|---|---|---|
| Accuracy | 0.7931 | 0.7843 |
| Macro-F1 | 0.5849 | 0.6468 |
| Balanced accuracy | 0.5825 | 0.6560 |
| Grad-CAM Spearman (noise s2, all / excl. zero maps) | 0.551 / 0.769 | 0.612 / 0.743 |
| IG Spearman (noise s2) | 0.739 | 0.736 |
| Grad-CAM deletion gap | 0.0793 | 0.0669 |
| IG deletion gap | 0.0557 | 0.0327 |
| Confidence AUROC | 0.7374 | 0.7012 |
| Grad-CAM instability AUROC | 0.6159 | 0.6288 |
| IG instability AUROC | 0.4731 | 0.4598 |

**The main findings were directionally consistent on the held-out test split.** For RQ2: under severe noise (noise, severity 2), the Grad-CAM map was all zero for 28.3% of validation images (17.6% on test), and only for images whose prediction had flipped. If these cases are scored as zero similarity, Integrated Gradients appears more stable on Spearman correlation (0.736 vs. 0.612 on test). If they are excluded, the two methods are statistically indistinguishable (Grad-CAM 0.769 versus IG 0.739 on validation, and 0.743 versus 0.736 on test, with overlapping 95% confidence intervals), and averaged over all 12 real conditions the two methods are near-tied on Spearman even before exclusion (Grad-CAM 0.837 vs. IG 0.830 on test). IG is higher on cosine similarity, and Grad-CAM is higher on top-20% region overlap (Grad-CAM 0.675 vs. IG 0.548 on test). The ranking therefore depends on the metric and on how zero maps are handled, and we do not claim that either method is more stable. The two methods also differ in map resolution and smoothing, so the comparison is not like for like.

The notable difference is higher test balanced accuracy, driven by better rare-class performance (`df` F1 0.579 vs 0.270; `vasc` 0.737 vs 0.581) at supports of only 20–21 images — sampling variation, not evidence of better generalisation; because the checkpoint was selected on validation, the test split gives the unbiased estimate for the metrics used for selection.

---

*All numbers from `results/` and `results_test/` CSVs. No numbers invented or estimated.*
