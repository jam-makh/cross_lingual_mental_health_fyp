#!/usr/bin/env python3
"""
model_comparison_stat.py
========================
Statistical comparison of model architectures WITHIN the same vectorizer
type, using test_macro_f1 as the performance metric.

The six comparisons are:

    1.  ML  / TF-IDF      : LR vs SVM vs MNB
    2.  ML  / Contextual  : LR vs SVM vs GNB  (pooled across DistilBERT /
                            CamemBERT / AraBERT, one per language)
    3.  ML  LR vs SVM     : head-to-head across all 6 scores (3 TF-IDF +
                            3 contextual), paired by (dataset × vectorizer)
    4.  DL  / TF-IDF      : LSTM vs BiLSTM vs CNN-RNN
    5.  DL  / Contextual  : LSTM vs BiLSTM vs CNN-RNN  (same pooling)
    6.  DL  LSTM vs BiLSTM: head-to-head across all 6 scores

Design notes
------------
• Each model gets exactly ONE score per (dataset, vectorizer) combination.
• Omnibus test: Kruskal-Wallis (nonparametric ANOVA for k≥3 groups).
• Post-hoc  : pairwise Wilcoxon signed-rank on matched (dataset) pairs,
  Bonferroni-corrected for k comparisons.
• For head-to-head (combos 3 & 6): simple Wilcoxon on all available
  paired scores across both vectorizer types.
• Effect size: rank-biserial r for Wilcoxon, eta-squared η² for KW.

Usage:
    python model_comparison_stat.py
    python model_comparison_stat.py --csv path/to/final_results_models.csv
    python model_comparison_stat.py --csv results.csv --out_dir my_results/
"""

from __future__ import annotations

import argparse
import warnings
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHA   = 0.05
METRIC  = "test_macro_f1"

CONTEXTUAL_VECS = {"DistilBERT", "CamemBERT", "AraBERT"}
TFIDF_VECS      = {"TF-IDF"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sig(p: float, alpha: float = ALPHA) -> str:
    return "✓ SIGNIFICANT" if p < alpha else "✗ not significant"


def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size for Wilcoxon signed-rank."""
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    n = len(x)
    W, _ = stats.wilcoxon(x, y)
    return float(1 - (2 * W) / (n * (n + 1) / 2))


def eta_squared_kw(H: float, n_total: int) -> float:
    """η² from Kruskal-Wallis H statistic."""
    return float((H - 1) / (n_total - 1)) if n_total > 1 else 0.0


def interpret_r(r: float) -> str:
    ar = abs(r)
    return "large" if ar >= 0.5 else ("medium" if ar >= 0.3 else "small")


def interpret_eta(eta: float) -> str:
    return "large" if eta >= 0.14 else ("medium" if eta >= 0.06 else "small")


def wilcoxon_pair(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    alpha: float,
) -> dict:
    """Run two-sided Wilcoxon on paired arrays. Returns result dict."""
    n = len(a)
    if n < 3:
        return {"feasible": False, "n": n,
                "note": f"n={n} < 3 — test not run."}
    diff = a - b
    if np.all(diff == 0):
        return {"feasible": False, "n": n,
                "note": "All differences zero — test not run."}
    W, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    r    = rank_biserial_r(a, b)
    mean_diff = float(np.mean(diff))
    direction = f"{label_a} > {label_b}" if mean_diff > 0 else f"{label_a} < {label_b}"
    return {
        "feasible": True, "n": n,
        "W": round(W, 3), "p_value": round(p, 6),
        "alpha": alpha, "significant": p < alpha,
        "mean_diff": round(mean_diff, 4),
        "direction": direction,
        "effect_r": round(r, 3),
        "effect_r_interp": interpret_r(r),
    }


def kruskal_wallis(groups: Dict[str, np.ndarray]) -> dict:
    """Run Kruskal-Wallis on k groups. Returns result dict."""
    names  = list(groups.keys())
    arrays = [groups[n] for n in names]
    n_total = sum(len(a) for a in arrays)

    if any(len(a) < 2 for a in arrays):
        return {"feasible": False,
                "note": "At least one group has < 2 observations."}
    if len(names) < 2:
        return {"feasible": False, "note": "Need at least 2 groups."}

    H, p = stats.kruskal(*arrays)
    eta  = eta_squared_kw(H, n_total)
    return {
        "feasible": True,
        "H": round(H, 4), "p_value": round(p, 6),
        "alpha": ALPHA, "significant": p < ALPHA,
        "eta_squared": round(eta, 4),
        "eta_interp": interpret_eta(eta),
        "n_total": n_total,
        "group_names": names,
        "group_means": {n: round(float(np.mean(groups[n])), 4) for n in names},
        "group_ns":    {n: len(groups[n]) for n in names},
    }


def posthoc_wilcoxon(
    groups: Dict[str, np.ndarray],
    datasets: Dict[str, List[str]],
) -> List[dict]:
    """
    All pairwise Wilcoxon tests with Bonferroni correction.
    Pairs are matched by dataset label within each group.
    """
    names = list(groups.keys())
    k = len(list(combinations(names, 2)))
    adj_alpha = ALPHA / k
    results = []

    for a_name, b_name in combinations(names, 2):
        # find datasets present in BOTH groups
        common = sorted(set(datasets[a_name]) & set(datasets[b_name]))
        if len(common) < 3:
            results.append({
                "comparison": f"{a_name} vs {b_name}",
                "feasible": False,
                "note": f"Only {len(common)} shared datasets — need ≥3.",
                "adj_alpha": adj_alpha,
            })
            continue

        # pull scores aligned by dataset
        ds_a = {d: s for d, s in zip(datasets[a_name], groups[a_name])}
        ds_b = {d: s for d, s in zip(datasets[b_name], groups[b_name])}
        a_arr = np.array([ds_a[d] for d in common])
        b_arr = np.array([ds_b[d] for d in common])

        r = wilcoxon_pair(a_arr, b_arr, a_name, b_name, adj_alpha)
        r["comparison"] = f"{a_name} vs {b_name}"
        r["adj_alpha"]  = adj_alpha
        r["k"]          = k
        r["datasets_used"] = common
        results.append(r)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    for col in ["model_type", "vectorizer", "model", "dataset"]:
        df[col] = df[col].str.strip()
    return df.dropna(subset=[METRIC])


def extract_groups(
    df: pd.DataFrame,
    model_type: str,
    vec_set: set,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    For a given model_type and vectorizer set, return:
        groups   : {model_name: array of scores}
        datasets : {model_name: list of dataset labels (aligned with scores)}

    For contextual vectorizers (DistilBERT / CamemBERT / AraBERT), each
    covers exactly one dataset, so pooling them gives one score per
    model per dataset — identical structure to TF-IDF.
    """
    sub = df[(df["model_type"] == model_type) & (df["vectorizer"].isin(vec_set))]
    groups: Dict[str, np.ndarray] = {}
    datasets: Dict[str, List[str]] = {}

    for model_name, grp in sub.groupby("model"):
        scores = grp[METRIC].values
        dsets  = grp["dataset"].tolist()
        groups[model_name]   = scores
        datasets[model_name] = dsets

    return groups, datasets


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    buf = StringIO()
    summary_rows: list = []

    def w(line=""):
        buf.write(line + "\n")

    SEP  = "=" * 72
    DASH = "─" * 60

    w(SEP)
    w("  MODEL COMPARISON — Within Same Vectorizer Type")
    w("  Metric : test_macro_f1")
    w("  Omnibus: Kruskal-Wallis  |  Post-hoc: Wilcoxon + Bonferroni")
    w(SEP)

    combo_id = 0

    # ─────────────────────────────────────────────────────────────────────
    # Combos 1–2: ML models within TF-IDF, then within Contextual
    # Combos 4–5: DL models within TF-IDF, then within Contextual
    # ─────────────────────────────────────────────────────────────────────
    for model_type in ["ML", "DL"]:
        for vec_label, vec_set in [("TF-IDF", TFIDF_VECS),
                                   ("Contextual", CONTEXTUAL_VECS)]:
            combo_id += 1
            groups, datasets = extract_groups(df, model_type, vec_set)

            w(f"\n{'#'*72}")
            w(f"  COMBO {combo_id}: {model_type} models within {vec_label}")
            w(f"{'#'*72}")

            if not groups:
                w("  No data found for this combination.")
                continue

            # ── Descriptive ──────────────────────────────────────────────
            w(f"\n  Models compared: {', '.join(sorted(groups.keys()))}")
            w(f"\n  Individual scores per model:")
            for m in sorted(groups.keys()):
                for d, s in zip(datasets[m], groups[m]):
                    w(f"    {m:<12}  {d:<10}  {vec_label if vec_label=='TF-IDF' else df[(df['model_type']==model_type)&(df['model']==m)&(df['dataset']==d)&(df['vectorizer'].isin(vec_set))]['vectorizer'].values[0]:<12}  F1 = {s:.4f}")

            w(f"\n  Mean F1 per model:")
            for m in sorted(groups.keys()):
                vals = groups[m]
                w(f"    {m:<12}  mean={np.mean(vals):.4f}  std={np.std(vals, ddof=1) if len(vals)>1 else 0:.4f}  n={len(vals)}")

            # ── Kruskal-Wallis ───────────────────────────────────────────
            kw = kruskal_wallis(groups)
            w(f"\n  [{combo_id}a] Kruskal-Wallis omnibus test")
            w(DASH)
            if not kw["feasible"]:
                w(f"  ⚠  Not run: {kw['note']}")
            else:
                w(f"  H  = {kw['H']:.4f}")
                w(f"  p  = {kw['p_value']:.6f}")
                w(f"  α  = {ALPHA}")
                w(f"  η² = {kw['eta_squared']:.4f}  ({kw['eta_interp']} effect)")
                w(f"  Result: {_sig(kw['p_value'])}")
                if kw["significant"]:
                    w(f"\n  → At least one model differs significantly. Proceeding to post-hoc.")
                else:
                    w(f"\n  → No significant difference across models. Post-hoc tests are exploratory.")

                summary_rows.append({
                    "Combo": f"COMBO {combo_id}: {model_type}/{vec_label} — KW omnibus",
                    "Test": "Kruskal-Wallis",
                    "Groups": " vs ".join(sorted(groups.keys())),
                    "n_obs": kw["n_total"],
                    "Statistic": kw["H"],
                    "p_value": kw["p_value"],
                    "Significant": kw["significant"],
                    "Effect": f"η²={kw['eta_squared']} ({kw['eta_interp']})",
                    "Direction": "—",
                    "Mean_diff": "—",
                })

            # ── Post-hoc Wilcoxon ────────────────────────────────────────
            ph = posthoc_wilcoxon(groups, datasets)
            k  = len(ph)
            w(f"\n  [{combo_id}b] Post-hoc pairwise Wilcoxon (Bonferroni k={k}, adj α={ALPHA/k:.4f})")
            w(DASH)

            for r in ph:
                w(f"\n  {r['comparison']}")
                if not r["feasible"]:
                    w(f"    ⚠  {r['note']}")
                    summary_rows.append({
                        "Combo": f"COMBO {combo_id}: {model_type}/{vec_label} — {r['comparison']}",
                        "Test": "Wilcoxon (post-hoc)",
                        "Groups": r["comparison"],
                        "n_obs": "N/A",
                        "Statistic": "N/A",
                        "p_value": "N/A",
                        "Significant": "N/A",
                        "Effect": "N/A",
                        "Direction": "N/A",
                        "Mean_diff": "N/A",
                    })
                else:
                    ds_str = ", ".join(r["datasets_used"])
                    w(f"    Datasets used  : {ds_str}  (n={r['n']})")
                    w(f"    W              = {r['W']}")
                    w(f"    p-value        = {r['p_value']:.6f}")
                    w(f"    adj. α         = {r['adj_alpha']:.4f}")
                    w(f"    Result         : {_sig(r['p_value'], r['adj_alpha'])}")
                    w(f"    Direction      : {r['direction']}")
                    w(f"    Mean diff      : {r['mean_diff']:+.4f}")
                    w(f"    Effect r       : {r['effect_r']}  ({r['effect_r_interp']})")

                    summary_rows.append({
                        "Combo": f"COMBO {combo_id}: {model_type}/{vec_label} — {r['comparison']}",
                        "Test": "Wilcoxon (post-hoc)",
                        "Groups": r["comparison"],
                        "n_obs": r["n"],
                        "Statistic": r["W"],
                        "p_value": r["p_value"],
                        "Significant": r["significant"],
                        "Effect": f"r={r['effect_r']} ({r['effect_r_interp']})",
                        "Direction": r["direction"],
                        "Mean_diff": r["mean_diff"],
                    })

    # ─────────────────────────────────────────────────────────────────────
    # Combo 3: ML — LR vs SVM head-to-head across all 6 scores
    # (both appear in TF-IDF AND contextual, 3 datasets each = 6 pairs)
    # Combo 6: DL — LSTM vs BiLSTM head-to-head across all 6 scores
    # ─────────────────────────────────────────────────────────────────────
    for model_type, m1, m2 in [("ML", "LR", "SVM"),
                                ("DL", "LSTM", "BiLSTM")]:
        combo_id += 1

        w(f"\n{'#'*72}")
        w(f"  COMBO {combo_id}: {model_type} — {m1} vs {m2} "
          f"(all vectorizers, all datasets, n=6 pairs)")
        w(f"{'#'*72}")

        sub = df[(df["model_type"] == model_type) &
                 (df["model"].isin([m1, m2]))]

        # Build pairs: same (dataset, vectorizer)
        pivot = sub.pivot_table(
            index=["dataset", "vectorizer"],
            columns="model",
            values=METRIC,
        ).dropna()

        if m1 not in pivot.columns or m2 not in pivot.columns:
            w(f"  ⚠  Cannot build pairs — one model is missing.")
            continue

        a_arr = pivot[m1].values
        b_arr = pivot[m2].values
        pair_labels = [f"{d} / {v}" for d, v in pivot.index]

        w(f"\n  Matched pairs (same dataset × vectorizer):")
        for lbl, a, b in zip(pair_labels, a_arr, b_arr):
            w(f"    {lbl:<30}  {m1}={a:.4f}  {m2}={b:.4f}  diff={a-b:+.4f}")

        r = wilcoxon_pair(a_arr, b_arr, m1, m2, ALPHA)

        w(f"\n  Wilcoxon Signed-Rank Test")
        w(DASH)
        if not r["feasible"]:
            w(f"  ⚠  {r['note']}")
        else:
            w(f"  n              = {r['n']}")
            w(f"  W              = {r['W']}")
            w(f"  p-value        = {r['p_value']:.6f}")
            w(f"  α              = {ALPHA}")
            w(f"  Result         : {_sig(r['p_value'])}")
            w(f"  Direction      : {r['direction']}")
            w(f"  Mean diff      : {r['mean_diff']:+.4f}  ({m1} − {m2})")
            w(f"  Effect r       : {r['effect_r']}  ({r['effect_r_interp']})")

            summary_rows.append({
                "Combo": f"COMBO {combo_id}: {model_type} {m1} vs {m2} (all vec/dataset)",
                "Test": "Wilcoxon",
                "Groups": f"{m1} vs {m2}",
                "n_obs": r["n"],
                "Statistic": r["W"],
                "p_value": r["p_value"],
                "Significant": r["significant"],
                "Effect": f"r={r['effect_r']} ({r['effect_r_interp']})",
                "Direction": r["direction"],
                "Mean_diff": r["mean_diff"],
            })

    # ─────────────────────────────────────────────────────────────────────
    # Interpretation guide
    # ─────────────────────────────────────────────────────────────────────
    w(f"\n{SEP}")
    w("  INTERPRETATION GUIDE")
    w(SEP)
    w("""
  Kruskal-Wallis (omnibus)
  ─────────────────────────
  Tests whether k≥3 independent groups come from the same distribution.
  H0: all model score distributions are equal.
  Effect size η² : ≥0.14 = large, ≥0.06 = medium, <0.06 = small.

  Wilcoxon Signed-Rank (post-hoc & head-to-head)
  ────────────────────────────────────────────────
  Nonparametric paired test. Pairs matched by dataset (and vectorizer
  for head-to-head combos). Bonferroni correction applied for k
  post-hoc comparisons: adj α = 0.05 / k.

  Effect size r (rank-biserial):
    |r| ≥ 0.5  →  large
    |r| ≥ 0.3  →  medium
    |r| < 0.3  →  small

  Power note
  ──────────
  With n=3 paired observations (one per dataset), the minimum
  achievable two-sided Wilcoxon p-value is 0.250 — significance at
  α=0.05 is mathematically impossible. Results are therefore
  descriptive and should be interpreted via effect size, not p-value.
  With n=6 (head-to-head combos), min p=0.031, borderline feasible.
""")

    summary_df = pd.DataFrame(summary_rows)
    return buf.getvalue(), summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-vectorizer model architecture comparison."
    )
    parser.add_argument("--csv", default="final_results_models.csv")
    parser.add_argument("--out_dir", default="model_stat_results")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.csv}")
    df = load_data(args.csv)
    print(f"  Rows loaded: {len(df)}")

    report, summary_df = build_report(df)

    print(report)

    report_path = out / "model_comparison_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved    → {report_path}")

    csv_path = out / "model_comparison_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary CSV     → {csv_path}")


if __name__ == "__main__":
    main()