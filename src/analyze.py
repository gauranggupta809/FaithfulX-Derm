"""Turn the raw CSVs into the tables and figures for the report.

Produces (in results/):
  summary_stability.csv       flip rate + similarity (with 95% CI) per method/perturbation/level
  threshold_sensitivity.csv   A/B/C/D counts at thresholds 0.70/0.80/0.85/0.90
  instability_vs_error.csv    can instability / confidence flag WRONG predictions? (AUROC)
  added_value_over_confidence.csv   does instability add anything beyond confidence?
  per_class_stability.csv     stability per true class
  faithfulness_summary.csv    attribution vs random deletion/insertion
  stability_vs_faithfulness.csv
  fig_similarity.png, fig_curves.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--metric", default="spearman", choices=["cosine", "spearman", "iou_top20"])
a = ap.parse_args()
R = Path(a.results)
rng = np.random.default_rng(0)


def boot(df, col, n=1000):
    """Mean with 95% CI, resampling LESIONS (not rows) to respect repeated measurements."""
    g = df.groupby("lesion_id")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (n, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), *np.percentile(means, [2.5, 97.5])


st = pd.read_csv(R / "stability.csv")
st["cond"] = st["perturb"] + "_" + st["level"]
real = st[~st["perturb"].str.startswith("null")]

# 1) summary table -------------------------------------------------------------
rows = []
for (m, p, lv), g in st.groupby(["method", "perturb", "level"]):
    row = {"method": m, "perturb": p, "level": lv, "n_images": g["image_id"].nunique(),
           "flip_rate": g["flipped"].mean()}
    for col in ("cosine", "spearman", "iou_top20"):
        mean, lo, hi = boot(g, col)
        row.update({f"{col}_mean": mean, f"{col}_lo": lo, f"{col}_hi": hi})
    rows.append(row)
summ = pd.DataFrame(rows)
null_ref = summ[summ["perturb"].str.startswith("null")].groupby("method")[f"{a.metric}_mean"].mean()
summ["drop_vs_null"] = summ.apply(lambda r: null_ref[r["method"]] - r[f"{a.metric}_mean"], axis=1)
summ.to_csv(R / "summary_stability.csv", index=False)

# 2) threshold sensitivity -----------------------------------------------------
tr = []
for thr in (0.70, 0.80, 0.85, 0.90):
    for (m, p), g in st.groupby(["method", "perturb"]):
        ps = g["flipped"] == 0
        es = g[a.metric] >= thr
        tr.append({"metric": a.metric, "threshold": thr, "method": m, "perturb": p, "n": len(g),
                   "A_stable_stable": int((ps & es).sum()), "B_stable_unstable": int((ps & ~es).sum()),
                   "C_changed_stable": int((~ps & es).sum()), "D_changed_unstable": int((~ps & ~es).sum())})
pd.DataFrame(tr).to_csv(R / "threshold_sensitivity.csv", index=False)

# 3) does explanation instability flag wrong predictions? -----------------------
per_img = (real.groupby(["image_id", "method"])
           .agg(sim=(a.metric, "mean"), flip=("flipped", "mean"), conf=("orig_conf", "first"),
                correct=("orig_correct", "first"), lesion_id=("lesion_id", "first")).reset_index())
per_img["wrong"] = 1 - per_img["correct"]
ev = []
for m, g in per_img.groupby("method"):
    if g["wrong"].nunique() < 2:
        continue
    for name, score in (("explanation instability (1 - similarity)", 1 - g["sim"]),
                        ("prediction flip rate", g["flip"]),
                        ("low confidence (1 - confidence) [reference]", 1 - g["conf"])):
        bs = []
        for _ in range(500):
            i = rng.integers(0, len(g), len(g))
            if g["wrong"].values[i].min() == g["wrong"].values[i].max():
                continue
            bs.append(roc_auc_score(g["wrong"].values[i], score.values[i]))
        ev.append({"method": m, "signal": name, "auroc": roc_auc_score(g["wrong"], score),
                   "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                   "n_images": len(g), "n_wrong": int(g["wrong"].sum())})
pd.DataFrame(ev).to_csv(R / "instability_vs_error.csv", index=False)

# 3b) does instability add information BEYOND confidence? (5-fold cross-validated) ----
added = []
for m, g in per_img.groupby("method"):
    if g["wrong"].sum() < 10 or (len(g) - g["wrong"].sum()) < 10:
        continue
    y = g["wrong"].values
    cvs = StratifiedKFold(5, shuffle=True, random_state=0)
    res = {}
    for name, cols in (("conf_only", ["conf"]), ("conf_plus_instability", ["conf", "sim"])):
        p = cross_val_predict(LogisticRegression(max_iter=1000), g[cols].values, y,
                              cv=cvs, method="predict_proba")[:, 1]
        res[name] = roc_auc_score(y, p)
    added.append({"method": m, "auroc_conf_only": res["conf_only"],
                  "auroc_conf_plus_instability": res["conf_plus_instability"],
                  "gain": res["conf_plus_instability"] - res["conf_only"], "n_images": len(g)})
pd.DataFrame(added).to_csv(R / "added_value_over_confidence.csv", index=False)

# 3c) per-class stability (are clinically important classes less stable?) -------------
(real.groupby(["method", "true_class"])
     .agg(n_images=("image_id", "nunique"), flip_rate=("flipped", "mean"),
          mean_cosine=("cosine", "mean"), mean_spearman=("spearman", "mean"),
          mean_iou_top20=("iou_top20", "mean")).reset_index()
     .to_csv(R / "per_class_stability.csv", index=False))

# 4) faithfulness --------------------------------------------------------------
if (R / "faithfulness.csv").exists():
    fa = pd.read_csv(R / "faithfulness.csv")
    fr = []
    for m, g in fa.groupby("method"):
        for col in ("del_auc", "rand_del_auc", "ins_auc", "rand_ins_auc", "del_gap", "ins_gap"):
            mean, lo, hi = boot(g, col)
            fr.append({"method": m, "measure": col, "mean": mean, "ci_lo": lo, "ci_hi": hi})
        fr.append({"method": m, "measure": "share_of_images_beating_random_deletion",
                   "mean": float((g["del_gap"] > 0).mean())})
    pd.DataFrame(fr).to_csv(R / "faithfulness_summary.csv", index=False)
    j = per_img.merge(fa[["image_id", "method", "del_gap", "ins_gap"]], on=["image_id", "method"])
    sv = []
    for m, g in j.groupby("method"):
        for col in ("del_gap", "ins_gap"):
            rho, p = spearmanr(g["sim"], g[col])
            sv.append({"method": m, "faithfulness": col, "spearman_rho": rho, "p_value": p, "n": len(g)})
    pd.DataFrame(sv).to_csv(R / "stability_vs_faithfulness.csv", index=False)

    cv = pd.read_csv(R / "curves.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, kind in enumerate(("del", "ins")):
        for m, g in cv[cv["kind"] == kind].groupby("method"):
            c = g.groupby("frac")["prob"].mean()
            ax[k].plot(c.index, c.values, label=m, ls="--" if m == "random" else "-")
        ax[k].set_title("Deletion (lower = more faithful)" if kind == "del"
                        else "Insertion (higher = more faithful)")
        ax[k].set_xlabel("fraction of cells deleted/inserted")
        ax[k].set_ylabel("probability of original predicted class")
        ax[k].legend()
    plt.tight_layout()
    plt.savefig(R / "fig_curves.png", dpi=150)
    plt.close()

# 5) similarity figure ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
order = list(summ["perturb"].unique())
for ax, m in zip(axs, ("gradcam", "ig")):
    g = summ[summ["method"] == m]
    for lv, off in (("s1", -0.2), ("s2", 0.2), ("s0", 0.0)):
        h = g[g["level"] == lv]
        if h.empty:
            continue
        x = [order.index(p) + off for p in h["perturb"]]
        y = h[f"{a.metric}_mean"].values
        ax.bar(x, y, width=0.35, label={"s1": "severity 1", "s2": "severity 2", "s0": "null"}[lv],
               yerr=[y - h[f"{a.metric}_lo"].values, h[f"{a.metric}_hi"].values - y], capsize=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_title(m)
    ax.set_ylabel(f"mean {a.metric} (original vs perturbed map)")
    ax.legend()
plt.tight_layout()
plt.savefig(R / "fig_similarity.png", dpi=150)
print("Done. Tables and figures are in", R)
print(pd.read_csv(R / "instability_vs_error.csv").to_string(index=False)
      if (R / "instability_vs_error.csv").stat().st_size > 5 else "Not enough misclassified images for AUROC.")
