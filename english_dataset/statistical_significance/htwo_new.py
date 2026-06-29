#!/usr/bin/env python3
"""
h2_analysis.py
==============
Full statistical analysis for Hypothesis H2:

    H2: Despite Anxiety posts being the most verbose, models achieve on average
    their lowest average recall on the Anxiety class due to it being the most
    underrepresented label in the dataset, demonstrating that linguistic richness
    does not compensate for statistical underrepresentation during training.

INPUT:  anxiety.csv  (must be in the same folder as this script)
        Expected columns (evaluation.py grouped-fill format):
            Model, Vectorizer, Split, Class,
            Accuracy, Precision, Recall, F1-score, Support, Macro F1

        NOTE: Model, Vectorizer, Split, Accuracy, and Macro F1 are only
        populated on the FIRST row of each model group (rows 2-4 are blank).
        This script forward-fills those columns automatically.

OUTPUT: h2_results/
    ├── h2_console_report.txt          Full test log
    ├── h2_statistical_summary.csv     One row per statistical test
    ├── h2_recall_heatmap.png          Recall per model × class
    ├── h2_recall_boxplot.png          Recall distribution per class
    ├── h2_support_vs_recall.png       Class imbalance vs mean recall
    └── h2_rank_chart.png              Recall rank per model

Usage:
    python h2_analysis.py
    python h2_analysis.py --input path/to/anxiety.csv --output_dir my_results/
"""

from __future__ import annotations

import argparse
import sys
import warnings
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05        # familywise significance level
BONFERRONI_K = 3    # pairwise comparisons: Anxiety vs (Dep, Suc, Nor)
ADJ_ALPHA = ALPHA / BONFERRONI_K   # ≈ 0.0167

CLASS_ORDER  = ["Anxiety", "Depression", "Normal", "Suicidal"]
CLASS_COLORS = {
    "Anxiety":    "#FF6B6B",
    "Depression": "#463EBD",
    "Normal":     "#D8BE3D",
    "Suicidal":   "#55D7BF",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sig(p: float, alpha: float) -> str:
    return "✓ SIGNIFICANT" if p < alpha else "✗ not significant"


def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    Effect size for Wilcoxon Signed-Rank test.
    r = 1 − (2·W) / (n·(n+1)/2)
    Interpretation: 0.1=small, 0.3=medium, 0.5=large
    """
    W, _ = stats.wilcoxon(x, y)
    n = len(x)
    max_W = n * (n + 1) / 2
    return float(1 - (2 * W) / max_W)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (mean diff / pooled SD)."""
    d = np.mean(x - y)
    s = np.std(x - y, ddof=1)
    return float(d / s) if s > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — handles evaluation.py grouped-fill format
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load anxiety.csv and normalise the grouped-fill format produced by
    evaluation.py.

    evaluation.py only writes Model, Vectorizer, Split, Accuracy, and
    Macro F1 on the first row of each 4-row model block.  The other
    three rows contain empty strings for those fields.  This function
    forward-fills them so every row is self-contained before any
    filtering or grouping is applied.
    """
    df = pd.read_csv(path, dtype=str)   # read everything as str first
    df.columns = df.columns.str.strip()

    required = {"Model", "Vectorizer", "Class", "Recall", "Support"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # ── Strip whitespace from all string cells ────────────────────────
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

    # ── Replace empty strings with NaN so forward-fill works ─────────
    grouped_fill_cols = ["Model", "Vectorizer", "Split", "Accuracy", "Macro F1"]
    for col in grouped_fill_cols:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)

    # ── Forward-fill the grouped columns ─────────────────────────────
    for col in grouped_fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ── Convert numeric columns ───────────────────────────────────────
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "Support", "Macro F1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Filter to Test split only (if Split column exists) ───────────
    if "Split" in df.columns:
        df = df[df["Split"].str.strip().str.lower() == "test"].reset_index(drop=True)

    # ── Drop rows where Class or Recall is missing ────────────────────
    df = df.dropna(subset=["Class", "Recall"]).reset_index(drop=True)

    # ── Validate that we have the expected classes ────────────────────
    found_classes = set(df["Class"].unique())
    expected_classes = set(CLASS_ORDER)
    missing_classes = expected_classes - found_classes
    extra_classes   = found_classes - expected_classes
    if missing_classes:
        print(f"  [WARNING] Expected classes not found in data: {missing_classes}")
        print(f"            Available classes: {sorted(found_classes)}")
    if extra_classes:
        print(f"  [INFO]    Extra classes found (will be included in analysis): {extra_classes}")

    print(
        f"  Loaded {len(df)} rows | "
        f"Models: {df['Model'].nunique()} | "
        f"Vectorizers: {df['Vectorizer'].nunique()} | "
        f"Classes: {sorted(df['Class'].unique())}"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def describe_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Report class support (test-set sample count) per class.

    Support is the number of test samples per class. Because evaluation.py
    writes the same support value on every row for a given class (it reflects
    the actual class size in the test split, not per-model), we take the
    median across all model rows for each class to get a robust estimate,
    then round to int.
    """
    support: Dict[str, int] = {}
    for cls in CLASS_ORDER:
        vals = df[df["Class"] == cls]["Support"].dropna()
        if len(vals) == 0:
            support[cls] = 0
        else:
            # Use median in case of any minor float discrepancies across rows
            support[cls] = int(round(vals.median()))
    return support


def describe_recall_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics of Recall for each class."""
    records = []
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]
    for cls in present_classes:
        vals = df[df["Class"] == cls]["Recall"].dropna().values
        if len(vals) == 0:
            continue
        records.append({
            "Class":         cls,
            "N_models":      len(vals),
            "Mean_Recall":   round(float(np.mean(vals)),   4),
            "Std_Recall":    round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 4),
            "Min_Recall":    round(float(np.min(vals)),    4),
            "Max_Recall":    round(float(np.max(vals)),    4),
            "Median_Recall": round(float(np.median(vals)), 4),
        })
    return pd.DataFrame(records)


def recall_rank_per_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each model × vectorizer combination, rank the 4 classes by recall
    (rank 1 = worst, rank 4 = best).

    Returns:
        pivot_df : wide DataFrame of recall values (rows = models, cols = classes)
        ranks_df : same shape but contains integer ranks
    """
    # Only keep classes that are in CLASS_ORDER and present in data
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]

    pivot = df.pivot_table(
        index=["Model", "Vectorizer"],
        columns="Class",
        values="Recall",
        aggfunc="first",    # one row per model × vectorizer × class
    )

    # Reorder columns to CLASS_ORDER (drop any not present)
    pivot = pivot.reindex(columns=present_classes)

    # Rank across classes for each model (axis=1), 1=lowest recall
    ranks = pivot.rank(axis=1, ascending=True, method="average").round().astype("Int64")

    return pivot, ranks


def friedman_across_classes(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Friedman test: H0 = all class recall distributions are equal across models.
    Each model × vectorizer is one 'block'; the four classes are the groups.

    Requires at least 3 complete observations per class.
    """
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]
    if len(present_classes) < 3:
        print(f"  [WARNING] Only {len(present_classes)} classes available; Friedman requires ≥3.")
        return float("nan"), float("nan")

    # Align observations: only use model rows that have recall for ALL classes
    pivot = df.pivot_table(
        index=["Model", "Vectorizer"],
        columns="Class",
        values="Recall",
        aggfunc="first",
    ).reindex(columns=present_classes).dropna()

    if len(pivot) < 3:
        print(f"  [WARNING] Only {len(pivot)} complete observations for Friedman test (need ≥3).")
        return float("nan"), float("nan")

    groups = [pivot[cls].values for cls in present_classes]
    stat, p = stats.friedmanchisquare(*groups)
    return float(stat), float(p)


def wilcoxon_anxiety_vs_others(df: pd.DataFrame) -> List[Dict]:
    """
    Pairwise Wilcoxon signed-rank test (two-sided) between Anxiety recall
    and each other class recall, across all model × vectorizer combinations.
    Bonferroni correction applied (k=3 comparisons, adjusted α ≈ 0.0167).

    Returns a list of result dicts, one per comparison.
    """
    if "Anxiety" not in df["Class"].values:
        print("  [WARNING] 'Anxiety' class not found in data; skipping Wilcoxon tests.")
        return []

    # Build aligned paired arrays using the pivot (only complete rows)
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]
    pivot = df.pivot_table(
        index=["Model", "Vectorizer"],
        columns="Class",
        values="Recall",
        aggfunc="first",
    ).reindex(columns=present_classes).dropna()

    anxiety_vals = pivot["Anxiety"].values if "Anxiety" in pivot.columns else np.array([])

    results = []
    other_classes = [c for c in present_classes if c != "Anxiety"]

    for other_cls in other_classes:
        other_vals = pivot[other_cls].values

        n = len(anxiety_vals)
        a = anxiety_vals
        b = other_vals

        diff = a - b
        if np.all(diff == 0):
            results.append({
                "Comparison":                    f"Anxiety vs {other_cls}",
                "W_statistic":                   float("nan"),
                "p_value":                       1.0,
                "Bonferroni_alpha":              ADJ_ALPHA,
                "Significant":                   False,
                "Mean_diff (Anxiety − Other)":   0.0,
                "Effect_r":                      float("nan"),
                "Cohen_d":                       0.0,
                "Direction":                     "tie",
                "n":                             n,
            })
            continue

        try:
            W, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        except ValueError as e:
            # Can occur when all differences are zero (already handled above)
            # or when n < 3 — report gracefully
            print(f"  [WARNING] Wilcoxon test skipped for Anxiety vs {other_cls}: {e}")
            results.append({
                "Comparison":                    f"Anxiety vs {other_cls}",
                "W_statistic":                   float("nan"),
                "p_value":                       float("nan"),
                "Bonferroni_alpha":              ADJ_ALPHA,
                "Significant":                   False,
                "Mean_diff (Anxiety − Other)":   round(float(np.mean(diff)), 4),
                "Effect_r":                      float("nan"),
                "Cohen_d":                       float("nan"),
                "Direction":                     "Anxiety > Other" if np.mean(diff) > 0 else "Anxiety < Other",
                "n":                             n,
            })
            continue

        # Effect size: rank-biserial r and Cohen's d
        try:
            r = rank_biserial_r(a, b)
        except Exception:
            r = float("nan")
        d = cohen_d(a, b)
        mean_diff = float(np.mean(diff))
        direction = "Anxiety > Other" if mean_diff > 0 else "Anxiety < Other"

        results.append({
            "Comparison":                    f"Anxiety vs {other_cls}",
            "W_statistic":                   round(W, 3),
            "p_value":                       round(p, 6),
            "Bonferroni_alpha":              ADJ_ALPHA,
            "Significant":                   p < ADJ_ALPHA,
            "Mean_diff (Anxiety − Other)":   round(mean_diff, 4),
            "Effect_r":                      round(r, 3) if not np.isnan(r) else float("nan"),
            "Cohen_d":                       round(d, 3),
            "Direction":                     direction,
            "n":                             n,
        })

    return results


def spearman_support_vs_recall(df: pd.DataFrame, support_map: Dict[str, int]) -> Tuple[float, float]:
    """
    Spearman correlation between class support size (test-set count) and
    mean recall across models.  Tests the hypothesis that more training
    data → higher recall.

    Uses the pre-computed support_map (from describe_class_distribution)
    to ensure consistency with what is reported in the text summary.
    """
    classes = [c for c in CLASS_ORDER if c in df["Class"].values and c in support_map]
    if len(classes) < 4:
        print(f"  [WARNING] Only {len(classes)} classes available for Spearman correlation.")
    if len(classes) < 3:
        return float("nan"), float("nan")

    supports     = [support_map[c] for c in classes]
    mean_recalls = [df[df["Class"] == c]["Recall"].mean() for c in classes]

    rho, p = stats.spearmanr(supports, mean_recalls)
    return float(rho), float(p)


def anxiety_rank_frequency(ranks_df: pd.DataFrame) -> Dict[str, int]:
    """Count how often Anxiety holds each rank position across models."""
    if "Anxiety" not in ranks_df.columns:
        return {}
    rank_counts = ranks_df["Anxiety"].value_counts().sort_index().to_dict()
    return {f"Rank {int(k)}": int(v) for k, v in rank_counts.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _model_label(index: pd.MultiIndex) -> List[str]:
    """Format multi-index (Model, Vectorizer) into readable plot labels."""
    return [f"{m}\n({v})" for m, v in index]


def plot_recall_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]
    pivot = df.pivot_table(
        index=["Model", "Vectorizer"],
        columns="Class",
        values="Recall",
        aggfunc="first",
    ).reindex(columns=present_classes)

    pivot.index = _model_label(pivot.index)

    fig, ax = plt.subplots(figsize=(max(9, len(present_classes) * 2), max(5, len(pivot) * 0.7 + 2)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Recall"},
    )
    ax.set_title("Recall per Model × Class (H2 Analysis)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model (Vectorizer)", fontsize=11)

    # Highlight Anxiety column header if present
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    for i, lbl in enumerate(tick_labels):
        if lbl == "Anxiety":
            ax.get_xticklabels()[i].set_color(CLASS_COLORS["Anxiety"])
            ax.get_xticklabels()[i].set_weight("bold")

    plt.tight_layout()
    plt.savefig(out_dir / "h2_recall_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓ h2_recall_heatmap.png")


def plot_recall_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]
    palette = [CLASS_COLORS.get(c, "#999999") for c in present_classes]

    data   = [df[df["Class"] == c]["Recall"].dropna().values for c in present_classes]
    means  = [np.mean(d) if len(d) > 0 else 0.0 for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data,
        labels=present_classes,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Individual data points with jitter
    rng = np.random.default_rng(42)
    for i, (vals, cls) in enumerate(zip(data, present_classes), start=1):
        if len(vals) > 0:
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, color="black", alpha=0.6, s=30, zorder=5)

    grand_mean = np.mean([m for m in means if not np.isnan(m)])
    ax.axhline(grand_mean, linestyle="--", color="gray", linewidth=1.2, label=f"Grand mean ({grand_mean:.3f})")

    for i, (cls, m) in enumerate(zip(present_classes, means), start=1):
        if not np.isnan(m):
            ax.text(i, m + 0.03, f"μ={m:.3f}", ha="center", fontsize=9, weight="bold")

    ax.set_title("Recall Distribution per Class — All Models", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylim(max(0.0, min(v for d in data for v in d) - 0.1), 1.08)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "h2_recall_boxplot.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓ h2_recall_boxplot.png")


def plot_support_vs_recall(
    df: pd.DataFrame,
    support_map: Dict[str, int],
    out_dir: Path,
) -> None:
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values and c in support_map]
    supports     = [support_map[c] for c in present_classes]
    mean_recalls = [df[df["Class"] == c]["Recall"].mean() for c in present_classes]
    colors       = [CLASS_COLORS.get(c, "#999999") for c in present_classes]

    fig, ax = plt.subplots(figsize=(7, 5))

    for cls, sup, rec, col in zip(present_classes, supports, mean_recalls, colors):
        ax.scatter(sup, rec, color=col, s=160, zorder=5, edgecolors="black", linewidth=0.8)
        ax.annotate(
            cls,
            xy=(sup, rec),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            color=col,
        )

    # Trend line (requires ≥2 points)
    if len(supports) >= 2:
        m_coef, b_coef = np.polyfit(supports, mean_recalls, 1)
        xs = np.linspace(min(supports) * 0.9, max(supports) * 1.05, 100)
        ax.plot(xs, m_coef * xs + b_coef, "k--", linewidth=1.2, alpha=0.5,
                label=f"Trend (slope={m_coef:.2e})")

    if len(supports) >= 3:
        rho, p_rho = stats.spearmanr(supports, mean_recalls)
        ax.text(
            0.05, 0.92,
            f"Spearman ρ = {rho:.3f}  (p = {p_rho:.3f})",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
        )

    ax.set_xlabel("Test-set Support (# samples)", fontsize=11)
    ax.set_ylabel("Mean Recall across models", fontsize=11)
    ax.set_title("Class Imbalance vs Model Recall", fontsize=13, weight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "h2_support_vs_recall.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓ h2_support_vs_recall.png")


def plot_rank_chart(ranks_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart of recall rank per model, coloured by class."""
    present_classes = [c for c in CLASS_ORDER if c in ranks_df.columns]
    models = _model_label(ranks_df.index)
    x      = np.arange(len(models))
    n_cls  = len(present_classes)
    width  = 0.8 / n_cls
    offsets = np.linspace(-(n_cls - 1) / 2, (n_cls - 1) / 2, n_cls) * width

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 5))
    for i, cls in enumerate(present_classes):
        ax.bar(
            x + offsets[i],
            ranks_df[cls].astype(float).values,
            width=width,
            label=cls,
            color=CLASS_COLORS.get(cls, "#999999"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Recall Rank (1=worst, 4=best)", fontsize=11)
    ax.set_title("Per-Model Recall Rank by Class", fontsize=13, weight="bold", pad=12)
    ax.set_yticks(range(1, n_cls + 1))
    ax.axhline((n_cls + 1) / 2, linestyle="--", color="gray", alpha=0.5)
    ax.legend(title="Class", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "h2_rank_chart.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓ h2_rank_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    df: pd.DataFrame,
    support_map: Dict[str, int],
    desc_df: pd.DataFrame,
    friedman_stat: float,
    friedman_p: float,
    wilcoxon_results: List[Dict],
    spearman_rho: float,
    spearman_p: float,
    pivot_df: pd.DataFrame,
    ranks_df: pd.DataFrame,
    rank_freq: Dict[str, int],
) -> str:
    buf = StringIO()
    present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]

    def w(line=""):
        buf.write(line + "\n")

    sep = "=" * 72

    w(sep)
    w("  H2 STATISTICAL ANALYSIS — Mental Health Text Classification FYP")
    w("  Hypothesis: Anxiety is the most underrepresented class and models")
    w("  achieve their lowest average recall on it despite its verbosity.")
    w(sep)

    # ── [1] Class distribution ────────────────────────────────────────
    w("\n[1] CLASS DISTRIBUTION (Test Set)")
    w("─" * 60)
    total = sum(support_map.values())
    if total == 0:
        w("  [WARNING] All support values are zero — check anxiety.csv Support column.")
    else:
        for cls in present_classes:
            sup = support_map.get(cls, 0)
            pct = 100 * sup / total if total > 0 else 0
            bar = "█" * max(0, int(pct / 2))
            w(f"  {cls:<15} support={sup:>5}  ({pct:5.1f}%)  {bar}")

        anxiety_sup = support_map.get("Anxiety", 0)
        if anxiety_sup > 0:
            ratios = "  ".join(
                f"{c} ×{support_map.get(c, 0) / anxiety_sup:.2f}"
                for c in present_classes if c != "Anxiety"
            )
            w(f"\n  Anxiety support = {anxiety_sup} ({100 * anxiety_sup / total:.1f}% of test set).")
            w(f"  Imbalance ratios vs Anxiety:  {ratios}")

    # ── [2] Descriptive recall ────────────────────────────────────────
    w("\n[2] DESCRIPTIVE STATISTICS — Recall per Class")
    w("─" * 60)
    sorted_classes = desc_df.sort_values("Mean_Recall").reset_index(drop=True)
    sorted_classes["Recall_rank"] = range(1, len(sorted_classes) + 1)
    w(f"  {'Class':<15}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}  {'Rank':>5}")
    for idx, row in sorted_classes.iterrows():
        suffix = "  ← LOWEST" if idx == 0 else ""
        if row["Class"] == "Anxiety" and idx != 0:
            suffix = "  ← ANXIETY"
        elif row["Class"] == "Anxiety" and idx == 0:
            suffix = "  ← LOWEST (ANXIETY)"
        w(
            f"  {row['Class']:<15}  "
            f"{row['Mean_Recall']:>6.4f}  "
            f"{row['Std_Recall']:>6.4f}  "
            f"{row['Min_Recall']:>6.4f}  "
            f"{row['Max_Recall']:>6.4f}  "
            f"{int(row['Recall_rank']):>5}{suffix}"
        )
    anx_row = sorted_classes[sorted_classes["Class"] == "Anxiety"]
    if len(anx_row) > 0:
        anx_rank = int(anx_row["Recall_rank"].values[0])
        anx_mean = float(anx_row["Mean_Recall"].values[0])
        anx_std  = float(anx_row["Std_Recall"].values[0])
        w(f"\n  Anxiety mean recall = {anx_mean:.4f}")
        w(f"  Anxiety recall rank = {anx_rank} out of {len(sorted_classes)}  (1=lowest)")
        w(f"  Anxiety recall std  = {anx_std:.4f}")

    # ── [3] Per-model pivot ───────────────────────────────────────────
    w("\n[3] PER-MODEL RECALL (rows=models, cols=classes)")
    w("─" * 60)
    pivot_str = pivot_df.reindex(columns=present_classes).to_string(float_format=lambda x: f"{x:.4f}")
    for line_str in pivot_str.split("\n"):
        w("  " + line_str)

    w("\n  Recall rank per model (1=worst):")
    rank_str = ranks_df.reindex(columns=present_classes).to_string()
    for line_str in rank_str.split("\n"):
        w("  " + line_str)

    if rank_freq:
        w(f"\n  Anxiety rank frequency across {len(ranks_df)} model combinations:")
        for rk, cnt in sorted(rank_freq.items()):
            w(f"    {rk}: {cnt} model(s)")
        avg_anx_rank = float(ranks_df["Anxiety"].mean()) if "Anxiety" in ranks_df.columns else float("nan")
        w(f"  Average Anxiety rank: {avg_anx_rank:.3f} / {len(present_classes)}.0")

    # ── [4] Friedman ──────────────────────────────────────────────────
    w("\n[4] FRIEDMAN TEST — Do recall distributions differ across classes?")
    w("─" * 60)
    w("  H0: All classes have the same recall distribution across models.")
    w("  H1: At least one class differs.")
    if np.isnan(friedman_stat):
        w("\n  [SKIPPED] Insufficient data for Friedman test.")
    else:
        w(f"\n  χ²  = {friedman_stat:.4f}")
        w(f"  p   = {friedman_p:.6f}")
        w(f"  α   = {ALPHA}")
        w(f"  Result: {_sig(friedman_p, ALPHA)}")
        if friedman_p < ALPHA:
            w(f"\n  → Recall distributions differ significantly across classes.")
            w(f"    Proceeding to pairwise Wilcoxon tests (Bonferroni α = {ADJ_ALPHA:.4f}).")
        else:
            w(f"\n  → No significant omnibus difference. Pairwise tests are exploratory.")

    # ── [5] Wilcoxon pairwise ─────────────────────────────────────────
    w(f"\n[5] PAIRWISE WILCOXON SIGNED-RANK TESTS (Anxiety vs each class)")
    w(f"    Bonferroni correction: k={BONFERRONI_K}, adjusted α = {ADJ_ALPHA:.4f}")
    w("─" * 60)
    if not wilcoxon_results:
        w("  [SKIPPED] No Wilcoxon results available.")
    else:
        for r in wilcoxon_results:
            w(f"\n  {r['Comparison']}")
            w(f"    W          = {r['W_statistic']}")
            w(f"    p-value    = {r['p_value']}")
            w(f"    adj. α     = {r['Bonferroni_alpha']:.4f}")
            sig_str = _sig(r['p_value'], r['Bonferroni_alpha']) if not (
                isinstance(r['p_value'], float) and np.isnan(r['p_value'])
            ) else "✗ could not compute"
            w(f"    Significant: {sig_str}")
            w(f"    Direction  : {r['Direction']}")
            w(f"    Mean diff  : {r['Mean_diff (Anxiety − Other)']:+.4f}  (Anxiety − Other)")
            w(f"    Effect r   : {r['Effect_r']}  (|r|≥0.5 = large)")
            w(f"    Cohen's d  : {r['Cohen_d']}  (|d|≥0.8 = large)")
            w(f"    n          : {r['n']} paired observations")

    # ── [6] Spearman ──────────────────────────────────────────────────
    w("\n[6] SPEARMAN CORRELATION — Support vs Mean Recall")
    w("─" * 60)
    w("  H0: No monotonic relationship between class support and mean recall.")
    if np.isnan(spearman_rho):
        w("  [SKIPPED] Insufficient classes for Spearman correlation.")
    else:
        w(f"  ρ = {spearman_rho:.4f}")
        w(f"  p = {spearman_p:.4f}  (α = {ALPHA})")
        w(f"  Result: {_sig(spearman_p, ALPHA)}")
        if spearman_p >= ALPHA:
            w(f"\n  → The correlation between support and recall is NOT statistically")
            w(f"    significant — class size alone does not explain the recall pattern.")
            w(f"    (Note: with n={len(present_classes)} classes Spearman has limited power.)")
        else:
            w(f"\n  → Significant positive correlation: larger classes → higher recall.")

    # ── H2 verdict ────────────────────────────────────────────────────
    w("\n" + sep)
    w("  H2 VERDICT")
    w(sep)

    anx_mean_str = f"{anx_mean:.4f}" if "anx_mean" in dir() else "N/A"
    anx_rank_str = str(anx_rank) if "anx_rank" in dir() else "N/A"
    n_classes_str = str(len(present_classes))

    w(f"""
  H2 states:  "Models achieve their LOWEST average recall on Anxiety due
  to it being the most underrepresented label."

  CLAIM 1 — Anxiety is the most underrepresented class
    → See [1] above for support counts and imbalance ratios.

  CLAIM 2 — Models achieve LOWEST average recall on Anxiety
    → Anxiety mean recall rank = {anx_rank_str} of {n_classes_str} (1=lowest).
    → See [2] for per-class recall statistics and actual rank.

  NUANCE:
    • The Friedman test (if significant) confirms recall is NOT equal
      across classes — the class performance hierarchy is real.
    • Wilcoxon pairwise tests reveal WHICH specific class pairs drive
      the difference and with what effect size.
    • Spearman ρ tests whether raw sample count predicts recall —
      if not significant, imbalance alone does not explain the pattern,
      implicating inter-class linguistic similarity as an additional factor.
    • High Anxiety recall variance (std) means model/vectorizer choice
      matters more for this class than for others: some combinations
      handle it well while others struggle significantly.

  IMPLICATION FOR METRIC CHOICE:
    H2 motivates using Macro F1 over accuracy throughout this project.
    Because Anxiety is underrepresented, a model can ignore it and still
    look good on accuracy.  Macro F1 penalises this, making it the
    correct primary evaluation metric for this multilabel, imbalanced task.
""")

    w(sep)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    print(f"\nLoading: {input_path}")
    df = load_data(input_path)

    # ── Analysis ──────────────────────────────────────────────────────
    support_map = describe_class_distribution(df)
    desc_df     = describe_recall_per_class(df)

    friedman_stat, friedman_p = friedman_across_classes(df)

    wilcoxon_results = wilcoxon_anxiety_vs_others(df)

    spearman_rho, spearman_p = spearman_support_vs_recall(df, support_map)

    pivot_df, ranks_df = recall_rank_per_model(df)
    rank_freq           = anxiety_rank_frequency(ranks_df)

    # ── Console report ────────────────────────────────────────────────
    report = build_report(
        df, support_map, desc_df,
        friedman_stat, friedman_p,
        wilcoxon_results,
        spearman_rho, spearman_p,
        pivot_df, ranks_df, rank_freq,
    )
    print(report)

    report_path = out / "h2_console_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Console report saved → {report_path}")

    # ── Statistical summary CSV ───────────────────────────────────────
    test_rows = []

    if not np.isnan(friedman_stat):
        n_complete = df.pivot_table(
            index=["Model", "Vectorizer"], columns="Class", values="Recall", aggfunc="first"
        ).dropna().shape[0]
        test_rows.append({
            "Test":        "Friedman",
            "Comparison":  "All classes",
            "Statistic":   round(friedman_stat, 4),
            "p_value":     round(friedman_p, 6),
            "Alpha":       ALPHA,
            "Significant": friedman_p < ALPHA,
            "Effect_r":    "",
            "Cohen_d":     "",
            "n_obs":       n_complete,
            "Note":        "Omnibus: H0=equal recall distributions across classes",
        })

    for r in wilcoxon_results:
        test_rows.append({
            "Test":        "Wilcoxon",
            "Comparison":  r["Comparison"],
            "Statistic":   r["W_statistic"],
            "p_value":     r["p_value"],
            "Alpha":       r["Bonferroni_alpha"],
            "Significant": r["Significant"],
            "Effect_r":    r["Effect_r"],
            "Cohen_d":     r["Cohen_d"],
            "n_obs":       r["n"],
            "Note":        f"{r['Direction']}, mean diff={r['Mean_diff (Anxiety − Other)']:+.4f}",
        })

    if not np.isnan(spearman_rho):
        present_classes = [c for c in CLASS_ORDER if c in df["Class"].values]
        test_rows.append({
            "Test":        "Spearman",
            "Comparison":  "Class Support vs Mean Recall",
            "Statistic":   round(spearman_rho, 4),
            "p_value":     round(spearman_p, 4),
            "Alpha":       ALPHA,
            "Significant": spearman_p < ALPHA,
            "Effect_r":    "",
            "Cohen_d":     "",
            "n_obs":       len(present_classes),
            "Note":        "Support size as predictor of recall",
        })

    stats_df = pd.DataFrame(test_rows)
    stats_csv = out / "h2_statistical_summary.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Statistical summary saved → {stats_csv}")

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    sns.set_theme(style="whitegrid")

    plot_recall_heatmap(df, out)
    plot_recall_boxplot(df, out)
    plot_support_vs_recall(df, support_map, out)

    present_classes = [c for c in CLASS_ORDER if c in ranks_df.columns]
    plot_rank_chart(ranks_df[present_classes], out)

    print(f"\nAll outputs saved to: {out.resolve()}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2 analysis: Anxiety underrepresentation vs model recall."
    )
    parser.add_argument(
        "--input",
        default="anxiety.csv",
        help="Path to anxiety.csv (default: anxiety.csv in current directory).",
    )
    parser.add_argument(
        "--output_dir",
        default="h2_results",
        help="Directory to save all outputs (default: h2_results/).",
    )
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()