"""Targeting analysis: coverage gap, exclusion/inclusion errors, precision-recall,
fixed-budget targeting, and bootstrap comparisons."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nigeria_resilience.visualization.save import save_and_show

# =====================================================================
# Configuration
# =====================================================================

COLMAP = {
    "MDN":         "resilience_score_mdn",
    "NN-1D":       "resilience_score_nn",
    "XGB-1D":      "resilience_score_xgb_1d",
    "XGB-Mix":     "resilience_score_xgb_md",
    "Elastic Net": "resilience_score_elastic_net",
    "OLS+GLM":     "resilience_score_ols_glm",
}

PANELS = [
    ("$1.90/day", "_pl_1_90", 1.90),
    ("$3.20/day", "_pl_3_20", 3.20),
    ("$5.50/day", "_pl_5_50", 5.50),
    ("National line", "_at_nat", None),
]

MODEL_ORDER = ["OLS+GLM", "Elastic Net", "XGB-1D", "XGB-Mix", "NN-1D", "MDN"]


# =====================================================================
# Helpers
# =====================================================================

def _resolve_colname(df: pd.DataFrame, base: str, suffix: str):
    cand = f"{base}{suffix}"
    if cand in df.columns:
        return cand
    if suffix == "_pl_1_90" and base in df.columns:
        return base
    return None


def make_binary_outcome(log_y, poverty_line_usd):
    threshold = np.log(poverty_line_usd)
    return (np.asarray(log_y) >= threshold).astype(int)


# =====================================================================
# Core targeting metrics
# =====================================================================

def compute_targeting_metrics(y_true, resilience_scores, threshold):
    """Compute targeting metrics at a given resilience *threshold*."""
    predicted_poor = (resilience_scores < threshold).astype(int)
    predicted_nonpoor = (resilience_scores >= threshold).astype(int)

    actual_poor = (y_true == 0).astype(int)
    actual_nonpoor = (y_true == 1).astype(int)

    true_positives = np.sum((predicted_poor == 1) & (actual_poor == 1))
    false_positives = np.sum((predicted_poor == 1) & (actual_nonpoor == 1))
    false_negatives = np.sum((predicted_nonpoor == 1) & (actual_poor == 1))
    true_negatives = np.sum((predicted_nonpoor == 1) & (actual_nonpoor == 1))

    predicted_coverage = float(np.mean(predicted_poor))
    actual_poverty_rate = float(np.mean(actual_poor))
    coverage_gap = predicted_coverage - actual_poverty_rate

    total_poor = np.sum(actual_poor)
    total_predicted_poor = np.sum(predicted_poor)

    if total_poor > 0:
        exclusion_error = false_negatives / total_poor
        recall = true_positives / total_poor
    else:
        exclusion_error = 0.0
        recall = 0.0

    if total_predicted_poor > 0:
        inclusion_error = false_positives / total_predicted_poor
        precision = true_positives / total_predicted_poor
    else:
        inclusion_error = 0.0
        precision = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        "threshold": threshold,
        "predicted_coverage": predicted_coverage,
        "actual_poverty_rate": actual_poverty_rate,
        "coverage_gap": coverage_gap,
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_negatives": int(true_negatives),
        "exclusion_error": exclusion_error,
        "inclusion_error": inclusion_error,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def analyze_targeting_all_models(test_df, poverty_line_usd, suffix, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.8]
    log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
    y_true = make_binary_outcome(log_y, poverty_line_usd)
    results = {}
    for model in MODEL_ORDER:
        base = COLMAP[model]
        col = _resolve_colname(test_df, base, suffix)
        if col is None:
            continue
        resilience = np.asarray(test_df[col], float)
        results[model] = {}
        for thresh in thresholds:
            results[model][thresh] = compute_targeting_metrics(y_true, resilience, thresh)
    return results


def bootstrap_targeting_metrics(test_df, poverty_line_usd, suffix, threshold, B=400, seed=42):
    """Bootstrap targeting metrics for all models at a single threshold."""
    rng = np.random.default_rng(seed)
    N = len(test_df)
    log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()

    model_scores = {}
    for model in MODEL_ORDER:
        base = COLMAP[model]
        col = _resolve_colname(test_df, base, suffix)
        if col is not None:
            model_scores[model] = np.asarray(test_df[col], float)

    boot_results = {
        model: {
            "coverage_gap": [], "exclusion_error": [], "inclusion_error": [],
            "precision": [], "recall": [], "f1_score": [],
        }
        for model in model_scores
    }

    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        log_y_b = log_y[idx]
        y_true_b = make_binary_outcome(log_y_b, poverty_line_usd)
        for model, scores in model_scores.items():
            scores_b = scores[idx]
            metrics = compute_targeting_metrics(y_true_b, scores_b, threshold)
            for k in boot_results[model]:
                boot_results[model][k].append(metrics[k])

    summary = {}
    for model in model_scores:
        summary[model] = {}
        for metric in boot_results[model]:
            arr = np.array(boot_results[model][metric])
            summary[model][metric] = {
                "mean": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "boot": arr,
            }
    return summary


# =====================================================================
# Visualization functions
# =====================================================================

def plot_targeting_metrics_comparison_bootstrap(
    test_df, poverty_lines_info,
    threshold=0.5, metrics_to_plot=None,
    B=400, seed=42, figsize=(16, 12),
    out_path=None,
):
    if metrics_to_plot is None:
        metrics_to_plot = ["coverage_gap", "exclusion_error", "inclusion_error", "f1_score"]

    all_results = {}
    poverty_line_labels_ordered = []

    for label, suffix, pl_value in poverty_lines_info:
        if pl_value is None:
            continue
        boot_results = bootstrap_targeting_metrics(test_df, pl_value, suffix, threshold, B=B, seed=seed)
        all_results[label] = boot_results
        poverty_line_labels_ordered.append(label)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    metric_labels = {
        "coverage_gap": "Coverage Gap (percentage points)",
        "exclusion_error": "Exclusion Error (% of poor excluded)",
        "inclusion_error": "Inclusion Error (% of non-poor included)",
        "precision": "Precision (% of targeted who are poor)",
        "recall": "Recall (% of poor reached)",
        "f1_score": "F1 Score",
    }

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(poverty_line_labels_ordered)))

    for ax_idx, metric in enumerate(metrics_to_plot):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        n_lines = len(poverty_line_labels_ordered)
        bar_width = 0.8 / n_lines
        first_pl = poverty_line_labels_ordered[0]
        models = [m for m in MODEL_ORDER if m in all_results[first_pl]]
        x_base = np.arange(len(models))

        for pl_idx, pl_label in enumerate(poverty_line_labels_ordered):
            boot_results = all_results[pl_label]
            offset = (pl_idx - n_lines / 2 + 0.5) * bar_width
            x_pos = x_base + offset
            scale = 100 if metric in ("exclusion_error", "inclusion_error", "coverage_gap") else 1
            means = np.array([boot_results[m][metric]["mean"] * scale for m in models if m in boot_results])
            lows = np.array([boot_results[m][metric]["ci_low"] * scale for m in models if m in boot_results])
            highs = np.array([boot_results[m][metric]["ci_high"] * scale for m in models if m in boot_results])
            ax.bar(x_pos, means, bar_width, color=colors[pl_idx], alpha=0.7, edgecolor="black", linewidth=0.5, label=pl_label)
            ax.errorbar(x_pos, means, yerr=[means - lows, highs - means], fmt="none", ecolor="black", capsize=3, linewidth=1, alpha=0.8)

        if metric == "coverage_gap":
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.5, zorder=0)

        ax.set_xticks(x_base)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=11)
        ax.set_title(f"{metric_labels.get(metric, metric)}\n(Threshold = {threshold})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(metrics_to_plot), 4):
        axes[j].axis("off")

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig, all_results


def plot_coverage_gap_by_threshold(test_df, poverty_lines_info, figsize=(14, 10), out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    thresholds = np.linspace(0.3, 0.8, 50)

    for ax, (label, suffix, pl_value) in zip(axes, poverty_lines_info):
        if pl_value is None:
            ax.axis("off")
            continue
        log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
        y_true = make_binary_outcome(log_y, pl_value)
        actual_poverty_rate = np.mean(y_true == 0)

        for model in MODEL_ORDER:
            base = COLMAP[model]
            col = _resolve_colname(test_df, base, suffix)
            if col is None:
                continue
            resilience = np.asarray(test_df[col], float)
            gaps = []
            for t in thresholds:
                predicted_poor = (resilience < t).astype(int)
                coverage = np.mean(predicted_poor)
                gap = coverage - actual_poverty_rate
                gaps.append(gap * 100)
            ax.plot(thresholds, gaps, linewidth=2, label=model, marker="", alpha=0.8)

        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Perfect targeting")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Resilience threshold for program eligibility", fontsize=10)
        ax.set_ylabel("Coverage gap (percentage points)", fontsize=10)
        ax.set_title(f"{label}\nActual poverty rate: {actual_poverty_rate:.1%}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.3, 0.8)

    for i in range(len(poverty_lines_info), 4):
        axes[i].axis("off")

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


def plot_exclusion_error_by_threshold(test_df, poverty_lines_info, figsize=(14, 10), out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    thresholds = np.linspace(0.2, 0.8, 50)

    for ax, (label, suffix, pl_value) in zip(axes, poverty_lines_info):
        if pl_value is None:
            ax.axis("off")
            continue
        log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
        y_true = make_binary_outcome(log_y, pl_value)
        actual_poor = (y_true == 0).astype(int)
        total_poor = np.sum(actual_poor)
        actual_poverty_rate = total_poor / len(y_true)

        for model in MODEL_ORDER:
            base = COLMAP[model]
            col = _resolve_colname(test_df, base, suffix)
            if col is None:
                continue
            resilience = np.asarray(test_df[col], float)
            exclusions = []
            for t in thresholds:
                predicted_poor = (resilience < t).astype(int)
                false_negatives = np.sum((predicted_poor == 0) & (actual_poor == 1))
                exclusion = (false_negatives / total_poor * 100) if total_poor > 0 else 0
                exclusions.append(exclusion)
            ax.plot(thresholds, exclusions, linewidth=2, label=model, alpha=0.8)

        random_exclusion = (1 - thresholds) * 100
        ax.plot(thresholds, random_exclusion, "k--", linewidth=1.5, alpha=0.5, label="Random targeting")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Resilience threshold for program eligibility", fontsize=10)
        ax.set_ylabel("Exclusion error (% of poor excluded)", fontsize=10)
        ax.set_title(f"{label}\nPoverty rate: {actual_poverty_rate:.1%}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0, 100)

    for i in range(len(poverty_lines_info), 4):
        axes[i].axis("off")

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


def plot_inclusion_error_by_threshold(test_df, poverty_lines_info, figsize=(14, 10), out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    thresholds = np.linspace(0.2, 0.8, 50)

    for ax, (label, suffix, pl_value) in zip(axes, poverty_lines_info):
        if pl_value is None:
            ax.axis("off")
            continue
        log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
        y_true = make_binary_outcome(log_y, pl_value)
        actual_poor = (y_true == 0).astype(int)
        actual_nonpoor = (y_true == 1).astype(int)
        actual_poverty_rate = np.mean(actual_poor)

        for model in MODEL_ORDER:
            base = COLMAP[model]
            col = _resolve_colname(test_df, base, suffix)
            if col is None:
                continue
            resilience = np.asarray(test_df[col], float)
            inclusions = []
            for t in thresholds:
                predicted_poor = (resilience < t).astype(int)
                total_targeted = np.sum(predicted_poor)
                false_positives = np.sum((predicted_poor == 1) & (actual_nonpoor == 1))
                inclusion = (false_positives / total_targeted * 100) if total_targeted > 0 else 0
                inclusions.append(inclusion)
            ax.plot(thresholds, inclusions, linewidth=2, label=model, alpha=0.8)

        ax.axhline(
            (1 - actual_poverty_rate) * 100, color="black", linestyle="--",
            linewidth=1.5, alpha=0.5, label=f"Random ({(1 - actual_poverty_rate) * 100:.0f}%)",
        )
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.axvline(0.8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Resilience threshold for program eligibility", fontsize=10)
        ax.set_ylabel("Inclusion error (% of targeted who are non-poor)", fontsize=10)
        ax.set_title(f"{label}\nPoverty rate: {actual_poverty_rate:.1%}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0, 100)

    for i in range(len(poverty_lines_info), 4):
        axes[i].axis("off")

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


def plot_precision_recall_curves(test_df, poverty_lines_info, figsize=(14, 10), out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    thresholds = np.linspace(0.1, 0.9, 50)

    for ax, (label, suffix, pl_value) in zip(axes, poverty_lines_info):
        if pl_value is None:
            ax.axis("off")
            continue
        log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
        y_true = make_binary_outcome(log_y, pl_value)
        actual_poor = (y_true == 0).astype(int)
        poverty_rate = np.mean(actual_poor)

        for model in MODEL_ORDER:
            base = COLMAP[model]
            col = _resolve_colname(test_df, base, suffix)
            if col is None:
                continue
            resilience = np.asarray(test_df[col], float)
            precisions, recalls = [], []
            for t in thresholds:
                predicted_poor = (resilience < t).astype(int)
                tp = np.sum((predicted_poor == 1) & (actual_poor == 1))
                fp = np.sum((predicted_poor == 1) & (actual_poor == 0))
                fn = np.sum((predicted_poor == 0) & (actual_poor == 1))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                precisions.append(prec)
                recalls.append(rec)
            ax.plot(recalls, precisions, linewidth=2, label=model, alpha=0.8)

        ax.axhline(poverty_rate, color="black", linestyle="--", linewidth=1.5, alpha=0.5, label=f"Random (precision={poverty_rate:.2f})")
        ax.set_xlabel("Recall (% of poor reached)", fontsize=10)
        ax.set_ylabel("Precision (% of targeted who are poor)", fontsize=10)
        ax.set_title(f"{label}\nPoverty rate: {poverty_rate:.1%}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for i in range(len(poverty_lines_info), 4):
        axes[i].axis("off")

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


def plot_fixed_budget_targeting(test_df, poverty_lines_info, figsize=(14, 10), out_path=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    budgets = np.linspace(0.1, 0.7, 30)

    for ax, (label, suffix, pl_value) in zip(axes, poverty_lines_info):
        if pl_value is None:
            ax.axis("off")
            continue
        log_y = test_df["lead_ltotcons_per_capita_per_day_2016cst"].to_numpy()
        y_true = make_binary_outcome(log_y, pl_value)
        actual_poor = (y_true == 0).astype(int)
        total_poor = np.sum(actual_poor)

        for model in MODEL_ORDER:
            base = COLMAP[model]
            col = _resolve_colname(test_df, base, suffix)
            if col is None:
                continue
            resilience = np.asarray(test_df[col], float)
            coverages = []
            for budget in budgets:
                n_target = int(len(resilience) * budget)
                targeted_indices = np.argsort(resilience)[:n_target]
                poor_reached = np.sum(actual_poor[targeted_indices])
                coverage = (poor_reached / total_poor * 100) if total_poor > 0 else 0
                coverages.append(coverage)
            ax.plot(budgets * 100, coverages, linewidth=2, label=model, marker="o", markersize=4, markevery=5, alpha=0.8)

        ax.plot(budgets * 100, np.minimum(budgets * 100 / (total_poor / len(y_true)), 100), "k--", linewidth=1.5, label="Perfect targeting", alpha=0.5)
        ax.plot(budgets * 100, budgets * 100, "k:", linewidth=1.5, label="Random targeting", alpha=0.5)
        ax.set_xlabel("Program budget (% of population)", fontsize=10)
        ax.set_ylabel("Poor coverage (% of poor reached)", fontsize=10)
        ax.set_title(f"{label}\nPoverty rate: {total_poor / len(y_true):.1%}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(10, 70)
        ax.set_ylim(0, 100)

    for i in range(len(poverty_lines_info), 4):
        axes[i].axis("off")

    plt.tight_layout()

    if out_path:
        save_and_show(fig, out_path)
    else:
        plt.show()

    return fig


def create_targeting_summary_table(test_df, poverty_lines_info, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.8]
    results_list = []
    for label, suffix, pl_value in poverty_lines_info:
        if pl_value is None:
            continue
        for thresh in thresholds:
            results = analyze_targeting_all_models(test_df, pl_value, suffix, [thresh])
            for model in MODEL_ORDER:
                if model not in results:
                    continue
                metrics = results[model][thresh]
                results_list.append({
                    "Poverty Line": label,
                    "Threshold": thresh,
                    "Model": model,
                    "Predicted Coverage (%)": metrics["predicted_coverage"] * 100,
                    "Actual Poverty Rate (%)": metrics["actual_poverty_rate"] * 100,
                    "Coverage Gap (pp)": metrics["coverage_gap"] * 100,
                    "Exclusion Error (%)": metrics["exclusion_error"] * 100,
                    "Inclusion Error (%)": metrics["inclusion_error"] * 100,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"],
                })
    return pd.DataFrame(results_list)


def run_complete_targeting_analysis(
    test_df, national_poverty_line_value,
    output_dir="results/targeting/",
    B=400, seed=42,
):
    """Run complete targeting analysis with all visualizations and bootstrap CIs."""
    os.makedirs(output_dir, exist_ok=True)

    poverty_lines_info = [
        ("$1.90/day", "_pl_1_90", 1.90),
        ("$3.20/day", "_pl_3_20", 3.20),
        ("$5.50/day", "_pl_5_50", 5.50),
        ("National line", "_at_nat", national_poverty_line_value),
    ]

    print("=" * 80)
    print("COMPREHENSIVE TARGETING ANALYSIS")
    print("=" * 80)

    summary_table = create_targeting_summary_table(test_df, poverty_lines_info, thresholds=[0.5, 0.8])
    summary_table.to_csv(f"{output_dir}targeting_summary_table.csv", index=False)

    fig1 = plot_coverage_gap_by_threshold(test_df, poverty_lines_info, out_path=f"{output_dir}coverage_gap_by_threshold.png")
    plt.close(fig1)

    fig2 = plot_exclusion_error_by_threshold(test_df, poverty_lines_info, out_path=f"{output_dir}exclusion_error_by_threshold.png")
    plt.close(fig2)

    fig3 = plot_inclusion_error_by_threshold(test_df, poverty_lines_info, out_path=f"{output_dir}inclusion_error_by_threshold.png")
    plt.close(fig3)

    fig4 = plot_precision_recall_curves(test_df, poverty_lines_info, out_path=f"{output_dir}precision_recall_curves.png")
    plt.close(fig4)

    fig5 = plot_fixed_budget_targeting(test_df, poverty_lines_info, out_path=f"{output_dir}fixed_budget_targeting.png")
    plt.close(fig5)

    fig6, boot_results_05 = plot_targeting_metrics_comparison_bootstrap(
        test_df, poverty_lines_info, threshold=0.5, B=B, seed=seed,
        out_path=f"{output_dir}bootstrap_comparison_threshold_05.png",
    )
    plt.close(fig6)

    fig7, boot_results_08 = plot_targeting_metrics_comparison_bootstrap(
        test_df, poverty_lines_info, threshold=0.8, B=B, seed=seed,
        out_path=f"{output_dir}bootstrap_comparison_threshold_08.png",
    )
    plt.close(fig7)

    print("ANALYSIS COMPLETE!")
    return summary_table, boot_results_05, boot_results_08
