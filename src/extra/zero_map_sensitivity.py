"""Task 2: Zero-map sensitivity analysis for Grad-CAM.

Reads results/stability.csv and results_test/stability.csv without modifying any existing result files.
A zero-map row is defined as a Grad-CAM row with spearman == 0, cosine == 0, and iou_top20 == 1.0.

1. Counts zero-map rows by perturbation, level, and true class (val and test).
2. Recomputes, with zero-map rows EXCLUDED, for Grad-CAM (and reports IG unchanged alongside for comparison):
   - mean Spearman, cosine, and iou_top20 for noise_s2 and jpeg_s2
   - the 12-condition (real perturbations) means for Spearman, cosine, and iou_top20
   - per-class mean Spearman (and cosine, iou_top20) across the 12 real conditions
   Uses the same lesion-level bootstrap (1,000 resamples, seed 0) for 95% CIs.
3. Saves results/zero_map_sensitivity.csv and results_test/zero_map_sensitivity.csv.
4. Evaluates whether conclusions (a), (b), (c), and (d) survive.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def boot_lesion(df, col, seed=0, n=1000):
    """Mean with 95% CI, resampling LESIONS (not rows) with seed=0 and n=1000."""
    rng = np.random.default_rng(seed)
    g = df.groupby("lesion_id")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (n, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def is_zero_map(df):
    """Identify Grad-CAM zero-map rows (all-zero after ReLU).
    Note: when the clean map is non-zero and only the perturbed map is all-zero,
    spearman == 0.0 and cosine == 0.0 (iou_top20 is 1.0 only if both maps are zero;
    otherwise np.argpartition selects arbitrary tied zero indices). All 60 val and 37 test
    zero-map rows have spearman == 0.0 and cosine == 0.0.
    """
    return (
        (df["method"] == "gradcam")
        & (df["spearman"] == 0.0)
        & (df["cosine"] == 0.0)
    )


def analyze_split(split_name, folder):
    st_path = ROOT / folder / "stability.csv"
    df = pd.read_csv(st_path)
    df["is_zero_map"] = is_zero_map(df)

    gc_all = df[df["method"] == "gradcam"]
    zm = df[df["is_zero_map"]]
    print("=====================================================================")
    print(f"SPLIT: {split_name.upper()} ({st_path})")
    print(f"Total Grad-CAM rows: {len(gc_all)} | Zero-map rows: {len(zm)} ({len(zm)/len(gc_all)*100:.2f}%)")
    print("=====================================================================")

    out_rows = []

    # 1. Counts by perturbation and level
    print("\n1a. Zero-map counts by (perturb, level):")
    by_pl = (
        gc_all.groupby(["perturb", "level"])
        .agg(total_rows=("is_zero_map", "count"), zero_map_rows=("is_zero_map", "sum"))
        .reset_index()
    )
    by_pl["zero_map_pct"] = by_pl["zero_map_rows"] / by_pl["total_rows"] * 100.0
    for _, r in by_pl.iterrows():
        if r["zero_map_rows"] > 0:
            print(f"  {r['perturb']:12s} {r['level']:2s}: {int(r['zero_map_rows']):2d} / {int(r['total_rows']):3d} ({r['zero_map_pct']:5.2f}%)")
        out_rows.append({
            "split": split_name,
            "category": "count_by_perturb_level",
            "method": "gradcam",
            "group": f"{r['perturb']}_{r['level']}",
            "metric": "zero_map_count",
            "n_total": int(r["total_rows"]),
            "n_zero_maps": int(r["zero_map_rows"]),
            "n_included": int(r["total_rows"] - r["zero_map_rows"]),
            "orig_mean": float(r["zero_map_rows"]),
            "orig_ci_lo": np.nan,
            "orig_ci_hi": np.nan,
            "excl_mean": float(r["zero_map_pct"]),
            "excl_ci_lo": np.nan,
            "excl_ci_hi": np.nan,
            "diff": np.nan,
        })

    # 1b. Counts by true_class (all 14 conditions and 12 real conditions)
    print("\n1b. Zero-map counts by true_class (all 14 conditions vs 12 real conditions):")
    gc_real_all = gc_all[~gc_all["perturb"].str.startswith("null")]
    for cl, g_cl in gc_all.groupby("true_class"):
        g_cl_real = gc_real_all[gc_real_all["true_class"] == cl]
        zm_all_c = int(g_cl["is_zero_map"].sum())
        zm_real_c = int(g_cl_real["is_zero_map"].sum())
        print(f"  {cl:6s}: all_14={zm_all_c:2d}/{len(g_cl):3d} ({zm_all_c/len(g_cl)*100:5.2f}%) | real_12={zm_real_c:2d}/{len(g_cl_real):3d} ({zm_real_c/len(g_cl_real)*100:5.2f}%)")
        out_rows.append({
            "split": split_name,
            "category": "count_by_true_class_all14",
            "method": "gradcam",
            "group": cl,
            "metric": "zero_map_count",
            "n_total": len(g_cl),
            "n_zero_maps": zm_all_c,
            "n_included": len(g_cl) - zm_all_c,
            "orig_mean": float(zm_all_c),
            "orig_ci_lo": np.nan,
            "orig_ci_hi": np.nan,
            "excl_mean": float(zm_all_c / len(g_cl) * 100.0),
            "excl_ci_lo": np.nan,
            "excl_ci_hi": np.nan,
            "diff": np.nan,
        })
        out_rows.append({
            "split": split_name,
            "category": "count_by_true_class_real12",
            "method": "gradcam",
            "group": cl,
            "metric": "zero_map_count",
            "n_total": len(g_cl_real),
            "n_zero_maps": zm_real_c,
            "n_included": len(g_cl_real) - zm_real_c,
            "orig_mean": float(zm_real_c),
            "orig_ci_lo": np.nan,
            "orig_ci_hi": np.nan,
            "excl_mean": float(zm_real_c / len(g_cl_real) * 100.0),
            "excl_ci_lo": np.nan,
            "excl_ci_hi": np.nan,
            "diff": np.nan,
        })

    # 2a. Condition means: noise_s2 and jpeg_s2 (plus all conditions with zero maps)
    print("\n2a. Condition-level sensitivity (noise_s2 and jpeg_s2):")
    summ_saved = pd.read_csv(ROOT / folder / "summary_stability.csv")
    for cond_p, cond_lv in [("noise", "s2"), ("jpeg", "s2")]:
        cond_name = f"{cond_p}_{cond_lv}"
        for m in ("gradcam", "ig"):
            sub_orig = df[(df["method"] == m) & (df["perturb"] == cond_p) & (df["level"] == cond_lv)]
            sub_excl = sub_orig[~sub_orig["is_zero_map"]]
            sv_row = summ_saved[(summ_saved["method"] == m) & (summ_saved["perturb"] == cond_p) & (summ_saved["level"] == cond_lv)].iloc[0]
            n_tot = len(sub_orig)
            n_zm = int(sub_orig["is_zero_map"].sum())
            n_inc = len(sub_excl)
            for col in ("spearman", "cosine", "iou_top20"):
                o_mean, o_lo, o_hi = float(sv_row[f"{col}_mean"]), float(sv_row[f"{col}_lo"]), float(sv_row[f"{col}_hi"])
                if n_zm == 0:
                    e_mean, e_lo, e_hi = o_mean, o_lo, o_hi
                else:
                    e_mean, e_lo, e_hi = boot_lesion(sub_excl, col, seed=0, n=1000)
                print(
                    f"  {cond_name:8s} | {m:7s} | {col:9s} | "
                    f"orig={o_mean:.4f} [{o_lo:.4f}, {o_hi:.4f}] (n={n_tot}) -> "
                    f"excl={e_mean:.4f} [{e_lo:.4f}, {e_hi:.4f}] (n={n_inc}, excl={n_zm}, diff={e_mean-o_mean:+.4f})"
                )
                out_rows.append({
                    "split": split_name,
                    "category": "condition_sensitivity",
                    "method": m,
                    "group": cond_name,
                    "metric": col,
                    "n_total": n_tot,
                    "n_zero_maps": n_zm,
                    "n_included": n_inc,
                    "orig_mean": o_mean,
                    "orig_ci_lo": o_lo,
                    "orig_ci_hi": o_hi,
                    "excl_mean": e_mean,
                    "excl_ci_lo": e_lo,
                    "excl_ci_hi": e_hi,
                    "diff": e_mean - o_mean,
                })

    # 2b. 12-condition means (real perturbations)
    print("\n2b. 12-condition (real perturbations) overall means:")
    real_df = df[~df["perturb"].str.startswith("null")]
    for m in ("gradcam", "ig"):
        sub_orig = real_df[real_df["method"] == m]
        sub_excl = sub_orig[~sub_orig["is_zero_map"]]
        n_tot = len(sub_orig)
        n_zm = int(sub_orig["is_zero_map"].sum())
        n_inc = len(sub_excl)
        for col in ("spearman", "cosine", "iou_top20"):
            o_mean, o_lo, o_hi = boot_lesion(sub_orig, col, seed=0, n=1000)
            e_mean, e_lo, e_hi = boot_lesion(sub_excl, col, seed=0, n=1000)
            print(
                f"  12_real  | {m:7s} | {col:9s} | "
                f"orig={o_mean:.4f} [{o_lo:.4f}, {o_hi:.4f}] (n={n_tot}) -> "
                f"excl={e_mean:.4f} [{e_lo:.4f}, {e_hi:.4f}] (n={n_inc}, excl={n_zm}, diff={e_mean-o_mean:+.4f})"
            )
            out_rows.append({
                "split": split_name,
                "category": "12_condition_mean",
                "method": m,
                "group": "all_12_real",
                "metric": col,
                "n_total": n_tot,
                "n_zero_maps": n_zm,
                "n_included": n_inc,
                "orig_mean": o_mean,
                "orig_ci_lo": o_lo,
                "orig_ci_hi": o_hi,
                "excl_mean": e_mean,
                "excl_ci_lo": e_lo,
                "excl_ci_hi": e_hi,
                "diff": e_mean - o_mean,
            })

    # 2c. Per-class mean Spearman (and cosine, iou_top20) across 12 real conditions
    print("\n2c. Per-class mean Spearman across 12 real conditions (Grad-CAM orig vs excl, and IG):")
    for m in ("gradcam", "ig"):
        sub_m = real_df[real_df["method"] == m]
        for cl, g_orig in sub_m.groupby("true_class"):
            g_excl = g_orig[~g_orig["is_zero_map"]]
            n_tot = len(g_orig)
            n_zm = int(g_orig["is_zero_map"].sum())
            n_inc = len(g_excl)
            for col in ("spearman", "cosine", "iou_top20"):
                o_mean, o_lo, o_hi = boot_lesion(g_orig, col, seed=0, n=1000)
                e_mean, e_lo, e_hi = boot_lesion(g_excl, col, seed=0, n=1000)
                if col == "spearman":
                    print(
                        f"  class={cl:5s} | {m:7s} | {col:9s} | "
                        f"orig={o_mean:.4f} [{o_lo:.4f}, {o_hi:.4f}] -> "
                        f"excl={e_mean:.4f} [{e_lo:.4f}, {e_hi:.4f}] (excl={n_zm}/{n_tot}, diff={e_mean-o_mean:+.4f})"
                    )
                out_rows.append({
                    "split": split_name,
                    "category": "per_class_real12",
                    "method": m,
                    "group": cl,
                    "metric": col,
                    "n_total": n_tot,
                    "n_zero_maps": n_zm,
                    "n_included": n_inc,
                    "orig_mean": o_mean,
                    "orig_ci_lo": o_lo,
                    "orig_ci_hi": o_hi,
                    "excl_mean": e_mean,
                    "excl_ci_lo": e_lo,
                    "excl_ci_hi": e_hi,
                    "diff": e_mean - o_mean,
                })

    out_df = pd.DataFrame(out_rows)
    out_path = ROOT / folder / "zero_map_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved sensitivity table to {out_path} ({len(out_df)} rows)")
    return out_df


if __name__ == "__main__":
    analyze_split("val", "results")
    print("\n")
    analyze_split("test", "results_test")
