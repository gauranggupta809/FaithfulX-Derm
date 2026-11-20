"""Independent audit recomputation script for FaithfulX-Derm.
Recomputes all statistics from raw CSVs (results/ and results_test/) using pure NumPy + Pandas
(zero external dependencies beyond numpy/pandas) and without modifying existing project files.
Outputs audit/audit_numbers.csv and prints detailed diagnostic tables for the audit report.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

audit_rows = []


def add_num(check_id, split, method, metric, value, ci_lo=np.nan, ci_hi=np.nan, ref_value=np.nan, diff=np.nan, source_file="", note=""):
    audit_rows.append({
        "check_id": check_id,
        "split": split,
        "method": method,
        "metric": metric,
        "value": value,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ref_value": ref_value,
        "diff": diff,
        "source_file": source_file,
        "note": note,
    })


def boot_lesion(df, col, rng, n=1000):
    g = df.groupby("lesion_id")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (n, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def roc_auc_np(y_true, y_score):
    """Exact ROC AUC via Mann-Whitney U statistic with tie handling, matching sklearn.metrics.roc_auc_score."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Average ranks for ties (1-based)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    n = len(y_score)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # average rank for indices i..j-1 (1-based: (i+1 + j)/2)
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos_ranks = ranks[pos].sum()
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def spearman_rho_np(x, y):
    rx = pd.Series(x).rank(method="average").values
    ry = pd.Series(y).rank(method="average").values
    return float(np.corrcoef(rx, ry)[0, 1])


def stratified_kfold_indices(y, n_splits=5, seed=0):
    """Stratified K-Fold matching standard class-wise random partitioning."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        for i, idx in enumerate(cls_idx):
            folds[i % n_splits].append(int(idx))
    out = []
    all_idx = np.arange(len(y))
    for k in range(n_splits):
        test_idx = np.array(sorted(folds[k]), dtype=np.int64)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]
        out.append((train_idx, test_idx))
    return out


def fit_predict_logreg_l2(X_train, y_train, X_test, C=1.0, max_iter=500):
    """L2-regularized binary logistic regression matching sklearn LogisticRegression(C=1.0) via Newton-Raphson (IRLS)."""
    n, d = X_train.shape
    # Augment with intercept (last column, unpenalized)
    Xa = np.hstack([X_train, np.ones((n, 1), dtype=np.float64)])
    Xta = np.hstack([X_test, np.ones((len(X_test), 1), dtype=np.float64)])
    w = np.zeros(d + 1, dtype=np.float64)
    reg = np.diag([1.0 / C] * d + [0.0])
    for _ in range(max_iter):
        z = np.clip(Xa @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xa.T @ (p - y_train) + reg @ w
        s = np.maximum(p * (1.0 - p), 1e-8)
        H = (Xa.T * s) @ Xa + reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w -= step
        if np.linalg.norm(step) < 1e-8:
            break
    zt = np.clip(Xta @ w, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zt))


def cv_predict_proba(X, y, n_splits=5, seed=0, C=1.0):
    preds = np.zeros(len(y), dtype=np.float64)
    for tr_idx, te_idx in stratified_kfold_indices(y, n_splits=n_splits, seed=seed):
        preds[te_idx] = fit_predict_logreg_l2(X[tr_idx], y[tr_idx], X[te_idx], C=C)
    return preds


print("=================================================================")
print("CHECK A1: SPLIT & FILES")
print("=================================================================")
meta_candidates = [
    ROOT / "master_metadata.csv",
    ROOT / "data" / "master_metadata.csv",
    Path("C:/skin_lesion_dataset/master_metadata.csv"),
]
meta_found = [p for p in meta_candidates if p.exists()]
ckpt_found = (ROOT / "models" / "resnet18_v2.pt").exists()
print(f"master_metadata.csv present in workspace? {len(meta_found) > 0} ({meta_found})")
print(f"models/resnet18_v2.pt present in workspace? {ckpt_found}")

st_val = pd.read_csv(ROOT / "results" / "stability.csv")
st_test = pd.read_csv(ROOT / "results_test" / "stability.csv")
fa_val = pd.read_csv(ROOT / "results" / "faithfulness.csv")
fa_test = pd.read_csv(ROOT / "results_test" / "faithfulness.csv")
crv_val = pd.read_csv(ROOT / "results" / "curves.csv")
crv_test = pd.read_csv(ROOT / "results_test" / "curves.csv")

val_lesions = set(st_val["lesion_id"].unique())
test_lesions = set(st_test["lesion_id"].unique())
val_imgs = set(st_val["image_id"].unique())
test_imgs = set(st_test["image_id"].unique())
overlap_lesions_xai = len(val_lesions & test_lesions)
overlap_imgs_xai = len(val_imgs & test_imgs)
print(f"XAI Val unique lesions: {len(val_lesions)}, Test unique lesions: {len(test_lesions)}, Overlap: {overlap_lesions_xai}")
print(f"XAI Val unique images: {len(val_imgs)}, Test unique images: {len(test_imgs)}, Overlap: {overlap_imgs_xai}")
add_num("A1", "val_vs_test", "all", "xai_lesion_overlap", overlap_lesions_xai, source_file="results/stability.csv & results_test/stability.csv")
add_num("A1", "val_vs_test", "all", "xai_image_overlap", overlap_imgs_xai, source_file="results/stability.csv & results_test/stability.csv")

print("\n=================================================================")
print("CHECK A2: MODEL QUALITY & MAJORITY BASELINE")
print("=================================================================")
for split_name, folder, prefix in [("val", "results", "classification_val"), ("test", "results_test", "classification_test")]:
    m_path = ROOT / folder / f"{prefix}_metrics.json"
    c_path = ROOT / folder / f"{prefix}_classwise.csv"
    cm_path = ROOT / folder / f"{prefix}_confusion.csv"
    metrics = json.loads(m_path.read_text())
    cw = pd.read_csv(c_path, index_col=0)
    cm = pd.read_csv(cm_path, index_col=0)
    total_n = int(cm.values.sum())
    nv_n = int(cm.loc["nv"].sum())
    maj_acc = nv_n / total_n
    maj_bal_acc = 1.0 / 7.0
    maj_macro_f1 = (2.0 * maj_acc / (1.0 + maj_acc)) / 7.0

    # Recompute from confusion matrix directly to verify saved JSON/CSV
    cm_mat = cm.values
    re_acc = float(np.trace(cm_mat) / total_n)
    re_recalls = np.diag(cm_mat) / cm_mat.sum(axis=1)
    re_precs = np.divide(np.diag(cm_mat), cm_mat.sum(axis=0), out=np.zeros(7), where=cm_mat.sum(axis=0) > 0)
    re_f1s = np.divide(2 * re_precs * re_recalls, re_precs + re_recalls, out=np.zeros(7), where=(re_precs + re_recalls) > 0)
    re_bal_acc = float(re_recalls.mean())
    re_macro_f1 = float(re_f1s.mean())

    print(f"--- {split_name.upper()} (n={total_n}) ---")
    print(f"Accuracy: {metrics['accuracy']:.4f} (Recomputed from CM: {re_acc:.4f} | Majority baseline: {maj_acc:.4f})")
    print(f"Balanced Acc: {metrics['balanced_accuracy']:.4f} (Recomputed from CM: {re_bal_acc:.4f} | Majority baseline: {maj_bal_acc:.4f})")
    print(f"Macro-F1: {metrics['macro_f1']:.4f} (Recomputed from CM: {re_macro_f1:.4f} | Majority baseline: {maj_macro_f1:.4f})")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print("Per-class recall & support:")
    for idx_c, cls in enumerate(cm.index):
        rec = cw.loc[cls, "recall"]
        f1 = cw.loc[cls, "f1-score"]
        sup = int(cw.loc[cls, "support"])
        tp = int(cm.loc[cls, cls])
        print(f"  {cls:6s}: recall={rec:.4f} ({tp}/{sup}), f1={f1:.4f}, support={sup}")
        add_num("A2", split_name, "resnet18", f"recall_{cls}", rec, ref_value=re_recalls[idx_c], diff=rec - re_recalls[idx_c], source_file=f"{folder}/{prefix}_classwise.csv", note=f"support={sup}, tp={tp}")
    add_num("A2", split_name, "resnet18", "accuracy", metrics["accuracy"], ref_value=maj_acc, diff=metrics["accuracy"] - maj_acc, source_file=f"{folder}/{prefix}_metrics.json", note="ref is majority-class baseline")
    add_num("A2", split_name, "resnet18", "balanced_accuracy", metrics["balanced_accuracy"], ref_value=maj_bal_acc, diff=metrics["balanced_accuracy"] - maj_bal_acc, source_file=f"{folder}/{prefix}_metrics.json")
    add_num("A2", split_name, "resnet18", "macro_f1", metrics["macro_f1"], ref_value=maj_macro_f1, diff=metrics["macro_f1"] - maj_macro_f1, source_file=f"{folder}/{prefix}_metrics.json")

tmp_m = json.loads((ROOT / "results" / "tmp_val_metrics.json").read_text())
hist = pd.read_csv(ROOT / "results" / "training_history.csv")
print(f"Training history best epoch: {hist.loc[hist['macro_f1'].idxmax(), 'epoch']} (macro_f1={hist['macro_f1'].max():.4f}), tmp_val macro_f1 (epoch 8)={tmp_m['macro_f1']:.4f}")

print("\n=================================================================")
print("CHECK A3: SAMPLE & MISCLASSIFIED IMAGES")
print("=================================================================")
for split_name, st_df, fa_df, folder in [("val", st_val, fa_val, "results"), ("test", st_test, fa_test, "results_test")]:
    img_df = st_df.drop_duplicates("image_id")
    n_imgs = len(img_df)
    n_lesions = img_df["lesion_id"].nunique()
    n_wrong = int((img_df["orig_correct"] == 0).sum())
    fa_imgs = fa_df["image_id"].nunique()
    fa_lesions = fa_df["lesion_id"].nunique()
    print(f"--- {split_name.upper()} ---")
    print(f"Stability unique images: {n_imgs}, unique lesions: {n_lesions}, total rows: {len(st_df)}")
    print(f"Faithfulness unique images: {fa_imgs}, unique lesions: {fa_lesions}, total rows: {len(fa_df)}, same image set: {set(img_df['image_id']) == set(fa_df['image_id'])}")
    print(f"Misclassified clean images: {n_wrong} / {n_imgs} ({n_wrong/n_imgs*100:.2f}%)")
    cls_counts = img_df.groupby("true_class").agg(
        n_images=("image_id", "count"),
        n_wrong=("orig_correct", lambda s: int((s == 0).sum()))
    )
    print(cls_counts)
    add_num("A3", split_name, "all", "unique_images", n_imgs, source_file=f"{folder}/stability.csv")
    add_num("A3", split_name, "all", "unique_lesions", n_lesions, source_file=f"{folder}/stability.csv")
    add_num("A3", split_name, "all", "misclassified_images", n_wrong, source_file=f"{folder}/stability.csv")
    for c, row in cls_counts.iterrows():
        add_num("A3", split_name, c, "class_images_and_errors", int(row["n_images"]), ref_value=int(row["n_wrong"]), source_file=f"{folder}/stability.csv", note="ref_value is n_wrong in class")

print("\n=================================================================")
print("CHECK A4 & B2: RECOMPUTE SUMMARY_STABILITY & NULL BASELINE & INSTABILITY_VS_ERROR")
print("=================================================================")
for split_name, st_df, folder in [("val", st_val, "results"), ("test", st_test, "results_test")]:
    rng = np.random.default_rng(0)
    summ_saved = pd.read_csv(ROOT / folder / "summary_stability.csv")
    rows = []
    for (m, p, lv), g in st_df.groupby(["method", "perturb", "level"]):
        row = {
            "method": m, "perturb": p, "level": lv,
            "n_images": g["image_id"].nunique(),
            "flip_rate": g["flipped"].mean()
        }
        for col in ("cosine", "spearman", "iou_top20"):
            mean, lo, hi = boot_lesion(g, col, rng)
            row.update({f"{col}_mean": mean, f"{col}_lo": lo, f"{col}_hi": hi})
        rows.append(row)
    summ_re = pd.DataFrame(rows)
    merged = summ_re.merge(summ_saved, on=["method", "perturb", "level"], suffixes=("_re", "_saved"))
    max_diffs = {}
    for col in ["flip_rate", "cosine_mean", "spearman_mean", "iou_top20_mean", "cosine_lo", "cosine_hi", "spearman_lo", "spearman_hi", "iou_top20_lo", "iou_top20_hi"]:
        d = (merged[f"{col}_re"] - merged[f"{col}_saved"]).abs().max()
        max_diffs[col] = d
    print(f"[{split_name.upper()}] Max absolute difference between recomputed and saved summary_stability.csv:")
    for k, v in max_diffs.items():
        print(f"  {k:18s}: {v:.2e}")
    add_num("A4", split_name, "all", "max_diff_summary_stability_means", max(max_diffs["flip_rate"], max_diffs["cosine_mean"], max_diffs["spearman_mean"], max_diffs["iou_top20_mean"]), source_file=f"{folder}/summary_stability.csv")

    print(f"\n[{split_name.upper()}] B2 Null vs Real Perturbations (Spearman & Cosine with 95% lesion-bootstrap CI):")
    for _, r in summ_re.iterrows():
        print(f"  {r['method']:7s} | {r['perturb']:10s} | {r['level']:2s} | flip={r['flip_rate']:.4f} | spearman={r['spearman_mean']:.4f} [{r['spearman_lo']:.4f}, {r['spearman_hi']:.4f}] | cosine={r['cosine_mean']:.4f} [{r['cosine_lo']:.4f}, {r['cosine_hi']:.4f}] | iou={r['iou_top20_mean']:.4f} [{r['iou_top20_lo']:.4f}, {r['iou_top20_hi']:.4f}]")
        add_num("B2", split_name, r["method"], f"spearman_{r['perturb']}_{r['level']}", r["spearman_mean"], r["spearman_lo"], r["spearman_hi"], source_file=f"{folder}/stability.csv")
        add_num("B2", split_name, r["method"], f"cosine_{r['perturb']}_{r['level']}", r["cosine_mean"], r["cosine_lo"], r["cosine_hi"], source_file=f"{folder}/stability.csv")
        add_num("B2", split_name, r["method"], f"iou_{r['perturb']}_{r['level']}", r["iou_top20_mean"], r["iou_top20_lo"], r["iou_top20_hi"], source_file=f"{folder}/stability.csv")
        add_num("B2", split_name, r["method"], f"flip_{r['perturb']}_{r['level']}", r["flip_rate"], source_file=f"{folder}/stability.csv")

    # Recompute instability_vs_error.csv for cosine, spearman, iou_top20
    real = st_df[~st_df["perturb"].str.startswith("null")]
    ive_saved = pd.read_csv(ROOT / folder / "instability_vs_error.csv")
    for metric_col in ("cosine", "spearman", "iou_top20"):
        per_img = (real.groupby(["image_id", "method"])
                   .agg(sim=(metric_col, "mean"), flip=("flipped", "mean"), conf=("orig_conf", "first"),
                        correct=("orig_correct", "first"), lesion_id=("lesion_id", "first")).reset_index())
        per_img["wrong"] = 1 - per_img["correct"]
        rng_ive = np.random.default_rng(0)
        for _ in range(28 * 3):
            _ = rng_ive.integers(0, len(per_img) // 2, (1000, len(per_img) // 2))
        print(f"\n[{split_name.upper()}] Recomputed instability_vs_error (metric={metric_col}):")
        for m, g in per_img.groupby("method"):
            for name, score in (("explanation instability (1 - similarity)", 1 - g["sim"]),
                                ("prediction flip rate", g["flip"]),
                                ("low confidence (1 - confidence) [reference]", 1 - g["conf"])):
                bs = []
                for _ in range(500):
                    i = rng_ive.integers(0, len(g), len(g))
                    if g["wrong"].values[i].min() == g["wrong"].values[i].max():
                        continue
                    bs.append(roc_auc_np(g["wrong"].values[i], score.values[i]))
                auroc = roc_auc_np(g["wrong"].values, score.values)
                lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                saved_row = ive_saved[(ive_saved["method"] == m) & (ive_saved["signal"] == name)]
                saved_val = float(saved_row["auroc"].values[0]) if len(saved_row) else np.nan
                diff = auroc - saved_val
                flag = " <-- MISMATCH > 0.01 vs saved cosine CSV" if abs(diff) > 0.01 else ""
                print(f"  {m:7s} | {name:45s} | AUROC={auroc:.4f} [{lo:.4f}, {hi:.4f}] | saved={saved_val:.4f} (diff={diff:+.6f}){flag}")
                add_num("A4", split_name, m, f"auroc_{metric_col}_{name[:15].strip()}", auroc, lo, hi, ref_value=saved_val, diff=diff, source_file=f"{folder}/instability_vs_error.csv")

print("\n=================================================================")
print("CHECK B3: SANITY CHECKS FROM RAW DATA")
print("=================================================================")
for split_name, crv_df, st_df, folder in [("val", crv_val, st_val, "results"), ("test", crv_test, st_test, "results_test")]:
    orig_probs = st_df.groupby("image_id")["orig_conf"].first()
    for m in ("gradcam", "ig", "random"):
        del0 = crv_df[(crv_df["method"] == m) & (crv_df["kind"] == "del") & (crv_df["frac"] == 0.0)].set_index("image_id")["prob"]
        ins1 = crv_df[(crv_df["method"] == m) & (crv_df["kind"] == "ins") & (crv_df["frac"] == 1.0)].set_index("image_id")["prob"]
        diff_del0_ins1 = (del0 - ins1).abs().max()
        diff_del0_orig = (del0 - orig_probs.reindex(del0.index)).abs().max()
        print(f"[{split_name.upper()}] {m:7s}: max |del(0) - ins(1)| = {diff_del0_ins1:.2e}, max |del(0) - orig_conf| = {diff_del0_orig:.2e}")
        add_num("B3", split_name, m, "max_diff_del0_ins1_origprob", max(diff_del0_ins1, diff_del0_orig), source_file=f"{folder}/curves.csv & {folder}/stability.csv")

for split_name, st_df, folder in [("val", st_val, "results"), ("test", st_test, "results_test")]:
    zero_sp = st_df[st_df["spearman"] == 0.0]
    print(f"[{split_name.upper()}] Rows in stability.csv where spearman == 0.0 (ConstantInputWarning fallback): {len(zero_sp)} / {len(st_df)} ({len(zero_sp)/len(st_df)*100:.2f}%)")
    if len(zero_sp) > 0:
        print("  Breakdown by method & perturb:", zero_sp.groupby(["method", "perturb", "level"]).size().to_dict())
        print("  Breakdown by true_class:", zero_sp.groupby(["method", "true_class"]).size().to_dict())
    add_num("B3", split_name, "gradcam", "constant_map_zero_spearman_rows", len(zero_sp), source_file=f"{folder}/stability.csv", note="Grad-CAM zero map after ReLU gives spearman=0.0, cosine=0.0, iou=1.0")

print("\n=================================================================")
print("CHECK B4: RANDOM-DELETION CONTROL (FAITHFULNESS)")
print("=================================================================")
for split_name, fa_df, folder in [("val", fa_val, "results"), ("test", fa_test, "results_test")]:
    rng_f = np.random.default_rng(0)
    fa_saved = pd.read_csv(ROOT / folder / "faithfulness_summary.csv")
    print(f"--- {split_name.upper()} ---")
    for m, g in fa_df.groupby("method"):
        for col in ("del_auc", "rand_del_auc", "ins_auc", "rand_ins_auc", "del_gap", "ins_gap"):
            mean, lo, hi = boot_lesion(g, col, rng_f)
            sv = fa_saved[(fa_saved["method"] == m) & (fa_saved["measure"] == col)]
            sv_mean = float(sv["mean"].values[0])
            print(f"  {m:7s} | {col:12s} | mean={mean:.4f} [{lo:.4f}, {hi:.4f}] | saved={sv_mean:.4f} (diff={mean-sv_mean:+.2e})")
            add_num("B4", split_name, m, col, mean, lo, hi, ref_value=sv_mean, diff=mean - sv_mean, source_file=f"{folder}/faithfulness.csv")
        share_del = float((g["del_gap"] > 0).mean())
        share_ins = float((g["ins_gap"] > 0).mean())
        print(f"  {m:7s} | share beating random deletion : {share_del:.4f} ({int((g['del_gap']>0).sum())}/{len(g)})")
        print(f"  {m:7s} | share beating random insertion: {share_ins:.4f} ({int((g['ins_gap']>0).sum())}/{len(g)})")
        add_num("B4", split_name, m, "share_beating_random_deletion", share_del, source_file=f"{folder}/faithfulness.csv")
        add_num("B4", split_name, m, "share_beating_random_insertion", share_ins, source_file=f"{folder}/faithfulness.csv")

print("\n=================================================================")
print("CHECK B5: THRESHOLD SENSITIVITY (0.70 vs 0.90, COSINE & SPEARMAN)")
print("=================================================================")
for split_name, st_df, folder in [("val", st_val, "results"), ("test", st_test, "results_test")]:
    print(f"--- {split_name.upper()} ---")
    for metric_col in ("cosine", "spearman"):
        for thr in (0.70, 0.90):
            real = st_df[~st_df["perturb"].str.startswith("null")]
            for subset_name, sub_df in [("REAL (12 conds)", real), ("null_noise", st_df[st_df["perturb"] == "null_noise"]), ("null_jpeg", st_df[st_df["perturb"] == "null_jpeg"])]:
                for m, g in sub_df.groupby("method"):
                    ps = g["flipped"] == 0
                    es = g[metric_col] >= thr
                    A = int((ps & es).sum())
                    B = int((ps & ~es).sum())
                    C = int((~ps & es).sum())
                    D = int((~ps & ~es).sum())
                    n = len(g)
                    print(f"  {metric_col:8s} | thr={thr:.2f} | {m:7s} | {subset_name:15s} (n={n:4d}): A={A:4d} ({A/n*100:5.1f}%), B={B:4d} ({B/n*100:5.1f}%), C={C:4d} ({C/n*100:5.1f}%), D={D:4d} ({D/n*100:5.1f}%)")
                    key_prefix = "real" if subset_name.startswith("REAL") else subset_name
                    add_num("B5", split_name, m, f"{key_prefix}_{metric_col}_thr{thr}_A_pct", A / n, source_file=f"{folder}/stability.csv", note=f"A={A}, B={B}, C={C}, D={D}")

print("\n=================================================================")
print("CHECK C1 & C2: ERROR DETECTION, 10x5-FOLD CV GAIN, AND CLASS CONFOUND CHECK")
print("=================================================================")
for split_name, st_df, folder in [("val", st_val, "results"), ("test", st_test, "results_test")]:
    real = st_df[~st_df["perturb"].str.startswith("null")]
    print(f"--- {split_name.upper()} ---")
    for metric_col in ("cosine", "spearman"):
        per_img = (real.groupby(["image_id", "method"])
                   .agg(sim=(metric_col, "mean"), flip=("flipped", "mean"), conf=("orig_conf", "first"),
                        correct=("orig_correct", "first"), lesion_id=("lesion_id", "first"),
                        orig_pred=("orig_pred", "first"), true_class=("true_class", "first")).reset_index())
        per_img["wrong"] = 1 - per_img["correct"]
        per_img["instab"] = 1.0 - per_img["sim"]

        for m, g in per_img.groupby("method"):
            g = g.sort_values("image_id").reset_index(drop=True)
            y = g["wrong"].values
            X_conf = g[["conf"]].values
            X_conf_instab = g[["conf", "sim"]].values

            # One-hot encode predicted class (drop first alphabetical class)
            pred_dummies = pd.get_dummies(g["orig_pred"], drop_first=True, dtype=float).values
            X_conf_cls = np.hstack([X_conf, pred_dummies])
            X_conf_cls_instab = np.hstack([X_conf_instab, pred_dummies])

            # 10-times repeated 5-fold CV with different seeds (seeds 0..9)
            aurocs_base, aurocs_plus, gains = [], [], []
            aurocs_c2_base, aurocs_c2_plus, gains_c2 = [], [], []
            preds_base_all = np.zeros((10, len(y)))
            preds_plus_all = np.zeros((10, len(y)))
            preds_c2_base_all = np.zeros((10, len(y)))
            preds_c2_plus_all = np.zeros((10, len(y)))

            for seed_i in range(10):
                pb = cv_predict_proba(X_conf, y, n_splits=5, seed=seed_i)
                pp = cv_predict_proba(X_conf_instab, y, n_splits=5, seed=seed_i)
                preds_base_all[seed_i] = pb
                preds_plus_all[seed_i] = pp
                ab, ap = roc_auc_np(y, pb), roc_auc_np(y, pp)
                aurocs_base.append(ab)
                aurocs_plus.append(ap)
                gains.append(ap - ab)

                pcb = cv_predict_proba(X_conf_cls, y, n_splits=5, seed=seed_i)
                pcp = cv_predict_proba(X_conf_cls_instab, y, n_splits=5, seed=seed_i)
                preds_c2_base_all[seed_i] = pcb
                preds_c2_plus_all[seed_i] = pcp
                acb, acp = roc_auc_np(y, pcb), roc_auc_np(y, pcp)
                aurocs_c2_base.append(acb)
                aurocs_c2_plus.append(acp)
                gains_c2.append(acp - acb)

            # Bootstrap 95% CI over images for the 10-seed average gain
            rng_boot = np.random.default_rng(42)
            boot_gains = []
            boot_gains_c2 = []
            for _ in range(1000):
                idx = rng_boot.integers(0, len(y), len(y))
                if y[idx].min() == y[idx].max():
                    continue
                bg = np.mean([roc_auc_np(y[idx], preds_plus_all[s, idx]) - roc_auc_np(y[idx], preds_base_all[s, idx]) for s in range(10)])
                bg_c2 = np.mean([roc_auc_np(y[idx], preds_c2_plus_all[s, idx]) - roc_auc_np(y[idx], preds_c2_base_all[s, idx]) for s in range(10)])
                boot_gains.append(bg)
                boot_gains_c2.append(bg_c2)

            mean_base, mean_plus, mean_gain = float(np.mean(aurocs_base)), float(np.mean(aurocs_plus)), float(np.mean(gains))
            ci_lo, ci_hi = float(np.percentile(boot_gains, 2.5)), float(np.percentile(boot_gains, 97.5))

            mean_c2_base, mean_c2_plus, mean_c2_gain = float(np.mean(aurocs_c2_base)), float(np.mean(aurocs_c2_plus)), float(np.mean(gains_c2))
            ci_c2_lo, ci_c2_hi = float(np.percentile(boot_gains_c2, 2.5)), float(np.percentile(boot_gains_c2, 97.5))

            print(f"  [C1 {metric_col:8s}] {m:7s}: 10x5-fold conf_only={mean_base:.4f}, conf+instab={mean_plus:.4f}, mean_gain={mean_gain:+.4f} (95% boot CI [{ci_lo:+.4f}, {ci_hi:+.4f}])")
            print(f"  [C2 {metric_col:8s}] {m:7s}: 10x5-fold conf+pred_class={mean_c2_base:.4f}, conf+pred_class+instab={mean_c2_plus:.4f}, mean_gain={mean_c2_gain:+.4f} (95% boot CI [{ci_c2_lo:+.4f}, {ci_c2_hi:+.4f}])")
            add_num("C1", split_name, m, f"cv10x5_gain_{metric_col}", mean_gain, ci_lo, ci_hi, source_file=f"{folder}/stability.csv", note=f"conf_only={mean_base:.4f}, conf+instab={mean_plus:.4f}")
            add_num("C2", split_name, m, f"cv10x5_gain_with_pred_class_{metric_col}", mean_c2_gain, ci_c2_lo, ci_c2_hi, source_file=f"{folder}/stability.csv", note=f"conf+class={mean_c2_base:.4f}, conf+class+instab={mean_c2_plus:.4f}")

print("\n=================================================================")
print("CHECK C3: PER-CLASS STABILITY")
print("=================================================================")
for split_name, st_df, folder in [("val", st_val, "results"), ("test", st_test, "results_test")]:
    real = st_df[~st_df["perturb"].str.startswith("null")]
    pc_saved = pd.read_csv(ROOT / folder / "per_class_stability.csv")
    pc_re = (real.groupby(["method", "true_class"])
             .agg(n_images=("image_id", "nunique"), flip_rate=("flipped", "mean"),
                  mean_cosine=("cosine", "mean"), mean_spearman=("spearman", "mean"),
                  mean_iou_top20=("iou_top20", "mean")).reset_index())
    max_diff_pc = (pc_re[["flip_rate", "mean_cosine", "mean_spearman", "mean_iou_top20"]] - pc_saved[["flip_rate", "mean_cosine", "mean_spearman", "mean_iou_top20"]]).abs().max().max()
    print(f"--- {split_name.upper()} (max diff vs saved: {max_diff_pc:.2e}) ---")
    for _, r in pc_re.iterrows():
        flag = " [UNRELIABLE: n < 20]" if r["n_images"] < 20 else ""
        print(f"  {r['method']:7s} | {r['true_class']:5s} | n={int(r['n_images']):2d} | flip={r['flip_rate']:.4f} | spearman={r['mean_spearman']:.4f} | cosine={r['mean_cosine']:.4f} | iou={r['mean_iou_top20']:.4f}{flag}")
        add_num("C3", split_name, r["method"], f"class_{r['true_class']}_spearman", r["mean_spearman"], ref_value=r["flip_rate"], source_file=f"{folder}/per_class_stability.csv", note=f"n={int(r['n_images'])}{flag}, ref_value=flip_rate")

print("\n=================================================================")
print("CHECK C4: OVERALL AVERAGES ACROSS ALL 12 REAL CONDITIONS (VAL VS TEST)")
print("=================================================================")
for split_name, st_df, folder in [("val", st_val, "results"), ("test", st_test, "results_test")]:
    real = st_df[~st_df["perturb"].str.startswith("null")]
    for m, g in real.groupby("method"):
        print(f"  {split_name:4s} | {m:7s} | flip={g['flipped'].mean():.4f} | cosine={g['cosine'].mean():.4f} | spearman={g['spearman'].mean():.4f} | iou_top20={g['iou_top20'].mean():.4f}")
        add_num("C4", split_name, m, "mean_12real_spearman", g["spearman"].mean(), ref_value=g["cosine"].mean(), source_file=f"{folder}/stability.csv", note=f"iou={g['iou_top20'].mean():.4f}, flip={g['flipped'].mean():.4f}")

pd.DataFrame(audit_rows).to_csv(AUDIT_DIR / "audit_numbers.csv", index=False)
print(f"\nSaved {len(audit_rows)} verified audit records to {AUDIT_DIR / 'audit_numbers.csv'}")
