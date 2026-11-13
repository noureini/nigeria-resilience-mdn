"""
Meta-Regression Analysis: Learning Correlates of Resilience Scores
===================================================================

Runs regression models to learn what household characteristics predict
each model's resilience scores. Uses GLM with logit link or OLS,
bootstrap inference, and feature selection.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from nigeria_resilience.analysis.smd import (
    COLMAP,
    MODEL_ORDER,
    PUBLICATION_LABELS,
    _resolve_colname,
)
from nigeria_resilience.visualization.save import save_and_show

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.families.links import Logit
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')

MAIN_PANELS = [
    ("$1.90/day Poverty Line", "_pl_1_90"),
    ("National Poverty Line", "_at_nat"),
]

MODEL_GROUPS = {
    "Distributional": ["MDN", "NN-1D", "XGB-1D", "XGB-Mix"],
    "Two-Stage": ["Elastic Net", "OLS+GLM"],
}


# =====================================================================
# Feature selection
# =====================================================================

def select_features_lasso(
    X: pd.DataFrame,
    y: np.ndarray,
    alpha: float = 0.01,
    max_features: int = 20,
) -> List[str]:
    """Select features using Lasso (L1 regularization)."""
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)

    coef_abs = np.abs(lasso.coef_)
    selected_idx = np.where(coef_abs > 0)[0]

    if len(selected_idx) > max_features:
        top_idx = np.argsort(coef_abs)[-max_features:]
        selected_idx = top_idx

    return X.columns[selected_idx].tolist()


def select_features_univariate(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 20,
) -> List[str]:
    """Select top-K features by univariate Spearman correlation."""
    from scipy.stats import spearmanr

    correlations = {}
    for col in X.columns:
        x_clean = X[col].fillna(X[col].mean())
        mask = np.isfinite(y) & np.isfinite(x_clean)
        if mask.sum() > 10:
            corr, _ = spearmanr(x_clean[mask], y[mask])
            correlations[col] = abs(corr)
        else:
            correlations[col] = 0

    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    return [feat for feat, _ in sorted_features[:top_k]]


def select_features_union(
    X: pd.DataFrame,
    y_dict: Dict[str, np.ndarray],
    method: str = "lasso",
    top_k_per_model: int = 15,
    overall_top_k: int = 20,
) -> List[str]:
    """Select features as union across all models."""
    all_selected = set()

    for model, y in y_dict.items():
        mask = np.isfinite(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if method == "lasso":
            selected = select_features_lasso(X_clean, y_clean, max_features=top_k_per_model)
        else:
            selected = select_features_univariate(X_clean, y_clean, top_k=top_k_per_model)
        all_selected.update(selected)

    if len(all_selected) > overall_top_k:
        feature_counts = {feat: 0 for feat in all_selected}
        for model, y in y_dict.items():
            mask = np.isfinite(y)
            X_clean = X[mask]
            y_clean = y[mask]
            if method == "lasso":
                selected = select_features_lasso(X_clean, y_clean, max_features=top_k_per_model)
            else:
                selected = select_features_univariate(X_clean, y_clean, top_k=top_k_per_model)
            for feat in selected:
                if feat in feature_counts:
                    feature_counts[feat] += 1
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in sorted_features[:overall_top_k]]

    return list(all_selected)


# =====================================================================
# Meta-regression analysis
# =====================================================================

def meta_regression_analysis(
    df: pd.DataFrame,
    X_cols: List[str],
    models: List[str] = None,
    poverty_line: str = "$1.90/day Poverty Line",
    method: str = "glm",
    feature_selection: str = "significant",
    significance_level: float = 0.05,
    top_k: int = 30,
    n_boot: int = 400,
    seed: int = 123,
    verbose: bool = True,
) -> Dict:
    """Run meta-regression: predict each model's resilience scores using features.

    Parameters
    ----------
    df : DataFrame
        Dataset with resilience scores and covariates.
    X_cols : list
        Covariate column names.
    models : list, optional
        Models to analyze (default: MODEL_ORDER).
    poverty_line : str
        Which poverty line to use.
    method : str
        "glm" (logit link) or "ols".
    feature_selection : str
        "significant", "union", or "all".
    significance_level : float
        P-value threshold for "significant" method.
    top_k : int
        Maximum features.
    n_boot : int
        Bootstrap replications.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Regression results, matrices, and metadata.
    """
    if models is None:
        models = list(MODEL_ORDER)

    suffix_map = {
        "$1.90/day Poverty Line": "_pl_1_90",
        "$3.20/day Poverty Line": "_pl_3_20",
        "$5.50/day Poverty Line": "_pl_5_50",
        "National Poverty Line": "_at_nat",
    }
    suffix = suffix_map.get(poverty_line, "_pl_1_90")

    X = df[X_cols].copy()
    X = X.fillna(X.mean())

    y_dict = {}
    for model in models:
        score_col = _resolve_colname(df, COLMAP[model], suffix)
        if score_col is not None:
            y = df[score_col].astype(float).to_numpy()
            y_dict[model] = y

    X_with_const = sm.add_constant(X) if STATSMODELS_AVAILABLE else X

    if verbose:
        print(f"\n{'='*70}")
        print(f"META-REGRESSION ANALYSIS: {poverty_line}")
        print(f"{'='*70}")
        print(f"Method: {method.upper()} with {n_boot} bootstrap replications")
        print(f"Total features available: {len(X_cols)}")

    # Feature selection
    if feature_selection == "significant":
        if verbose:
            print(f"\nPhase 1: Identifying significant features (p<{significance_level})...")

        sig_features_by_model = {}
        for model in models:
            if model not in y_dict:
                continue
            y = y_dict[model]
            mask = np.isfinite(y)
            y_clean = y[mask]
            X_clean = X_with_const[mask]
            if len(y_clean) < 100:
                continue
            try:
                if method == "glm":
                    epsilon = 1e-6
                    y_trans = np.clip(y_clean, epsilon, 1 - epsilon)
                    glm_model = sm.GLM(y_trans, X_clean, family=Binomial(link=Logit()))
                    fitted = glm_model.fit(disp=False)
                else:
                    fitted = sm.OLS(y_clean, X_clean).fit()
                pvals = fitted.pvalues.drop(
                    'const' if 'const' in fitted.pvalues.index else 'Intercept',
                    errors='ignore',
                )
                sig_features = pvals[pvals < significance_level].index.tolist()
                sig_features_by_model[model] = sig_features
                if verbose:
                    print(f"  {model}: {len(sig_features)} significant features")
            except Exception as e:
                if verbose:
                    print(f"  {model}: Error in initial regression: {e}")

        all_sig_features = set()
        for features in sig_features_by_model.values():
            all_sig_features.update(features)
        selected_features = list(all_sig_features)

        if len(selected_features) > top_k:
            selected_features = selected_features[:top_k]

        if verbose:
            print(f"\nSelected {len(selected_features)} features significant in at least one model")

    elif feature_selection == "union":
        selected_features = select_features_union(
            X, y_dict, method="lasso", top_k_per_model=15, overall_top_k=top_k
        )
    elif feature_selection == "all":
        selected_features = X_cols
    else:
        raise ValueError(f"Unknown feature_selection: {feature_selection}")

    if verbose:
        print(f"\nPhase 2: Final regressions with {len(selected_features)} selected features")
        for i, feat in enumerate(selected_features, 1):
            label = PUBLICATION_LABELS.get(feat, feat)
            print(f"  {i:2d}. {label}")

    X_selected = X[selected_features].copy()
    X_selected = sm.add_constant(X_selected) if STATSMODELS_AVAILABLE else X_selected

    rng = np.random.default_rng(seed)
    results = {}

    for model in models:
        if model not in y_dict:
            if verbose:
                print(f"\nWarning: {model} scores not found, skipping")
            continue

        y = y_dict[model]
        mask = np.isfinite(y)
        y_clean = y[mask]
        X_clean = X_selected[mask]

        if len(y_clean) < 100:
            continue
        if not STATSMODELS_AVAILABLE:
            continue

        try:
            if method == "glm":
                epsilon = 1e-6
                y_trans = np.clip(y_clean, epsilon, 1 - epsilon)
                glm_model = sm.GLM(y_trans, X_clean, family=Binomial(link=Logit()))
                fitted = glm_model.fit(disp=False)
            else:
                fitted = sm.OLS(y_clean, X_clean).fit()

            if verbose:
                print(f"\n{'-'*70}")
                print(f"{model} - Bootstrap ({n_boot} replications)...")

            boot_coefs = []
            pbar = tqdm(range(n_boot), desc=f"  {model}", leave=False) if verbose else range(n_boot)

            for b in pbar:
                boot_idx = rng.integers(0, len(y_clean), size=len(y_clean))
                y_boot = y_clean[boot_idx]
                X_boot = X_clean.iloc[boot_idx] if isinstance(X_clean, pd.DataFrame) else X_clean[boot_idx]

                try:
                    if method == "glm":
                        y_boot_trans = np.clip(y_boot, epsilon, 1 - epsilon)
                        fit_boot = sm.GLM(y_boot_trans, X_boot, family=Binomial(link=Logit())).fit(disp=False)
                    else:
                        fit_boot = sm.OLS(y_boot, X_boot).fit()
                    boot_coefs.append(fit_boot.params.values)
                except Exception:
                    continue

            boot_coefs = np.array(boot_coefs)
            boot_ci_low = np.percentile(boot_coefs, 2.5, axis=0)
            boot_ci_high = np.percentile(boot_coefs, 97.5, axis=0)
            boot_se = np.std(boot_coefs, axis=0, ddof=1)

            results[model] = {
                'fitted': fitted,
                'params': fitted.params,
                'pvalues': fitted.pvalues,
                'conf_int': fitted.conf_int(),
                'bse': fitted.bse,
                'boot_ci_low': pd.Series(boot_ci_low, index=fitted.params.index),
                'boot_ci_high': pd.Series(boot_ci_high, index=fitted.params.index),
                'boot_se': pd.Series(boot_se, index=fitted.params.index),
                'nobs': fitted.nobs,
                'n_boot_success': len(boot_coefs),
            }

            if hasattr(fitted, 'rsquared'):
                results[model]['rsquared'] = fitted.rsquared
            elif hasattr(fitted, 'pseudo_rsquared'):
                results[model]['rsquared'] = fitted.pseudo_rsquared()
            else:
                results[model]['rsquared'] = None

            if verbose:
                print(f"{model}: N={fitted.nobs:,}, Bootstrap={len(boot_coefs)}/{n_boot}")

        except Exception as e:
            if verbose:
                print(f"\nError fitting {model}: {e}")

    # Build matrices
    coef_matrix = pd.DataFrame(index=selected_features, columns=models)
    pval_matrix = pd.DataFrame(index=selected_features, columns=models)
    se_matrix = pd.DataFrame(index=selected_features, columns=models)
    boot_ci_low_matrix = pd.DataFrame(index=selected_features, columns=models)
    boot_ci_high_matrix = pd.DataFrame(index=selected_features, columns=models)

    for model in models:
        if model not in results:
            continue
        for feat in selected_features:
            if feat in results[model]['params'].index:
                coef_matrix.loc[feat, model] = results[model]['params'][feat]
                pval_matrix.loc[feat, model] = results[model]['pvalues'][feat]
                se_matrix.loc[feat, model] = results[model]['boot_se'][feat]
                boot_ci_low_matrix.loc[feat, model] = results[model]['boot_ci_low'][feat]
                boot_ci_high_matrix.loc[feat, model] = results[model]['boot_ci_high'][feat]

    coef_matrix = coef_matrix.astype(float)
    pval_matrix = pval_matrix.astype(float)
    se_matrix = se_matrix.astype(float)
    boot_ci_low_matrix = boot_ci_low_matrix.astype(float)
    boot_ci_high_matrix = boot_ci_high_matrix.astype(float)

    return {
        'results': results,
        'selected_features': selected_features,
        'coef_matrix': coef_matrix,
        'pval_matrix': pval_matrix,
        'se_matrix': se_matrix,
        'boot_ci_low': boot_ci_low_matrix,
        'boot_ci_high': boot_ci_high_matrix,
        'poverty_line': poverty_line,
        'method': method,
        'n_boot': n_boot,
    }


# =====================================================================
# Heatmap grid (2x2 for 4 poverty lines)
# =====================================================================

def plot_coefficient_heatmap_grid(
    results_dict: Dict[str, Dict],
    poverty_lines: List[str] = None,
    figsize: Tuple[float, float] = (18, 14),
    dpi: int = 300,
    show_se: bool = True,
    outfile: str = None,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Create 2x2 grid of coefficient heatmaps for multiple poverty lines.

    Parameters
    ----------
    results_dict : dict
        poverty_line -> analysis_results
    out_path : str, optional
        Save path (uses save_and_show).
    """
    if poverty_lines is None:
        poverty_lines = [
            "$1.90/day Poverty Line",
            "$3.20/day Poverty Line",
            "$5.50/day Poverty Line",
            "National Poverty Line",
        ]

    available_lines = [pl for pl in poverty_lines if pl in results_dict]
    if len(available_lines) < len(poverty_lines):
        poverty_lines = available_lines

    all_features_raw = set()
    for pl in poverty_lines:
        if pl in results_dict:
            all_features_raw.update(results_dict[pl]['selected_features'])
    all_features_raw = sorted(all_features_raw)
    pretty_labels = [PUBLICATION_LABELS.get(f, f) for f in all_features_raw]

    first_pl = poverty_lines[0]
    models = results_dict[first_pl]['coef_matrix'].columns.tolist()

    vmax = 0.0
    for pl in poverty_lines:
        if pl in results_dict:
            coef_values = results_dict[pl]['coef_matrix'].reindex(all_features_raw).values
            vmax = max(vmax, np.nanmax(np.abs(coef_values)))
    vmax = max(vmax, 0.1)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes = axes.flatten()

    panel_titles = {
        "$1.90/day Poverty Line": "$1.90/day",
        "$3.20/day Poverty Line": "$3.20/day",
        "$5.50/day Poverty Line": "$5.50/day",
        "National Poverty Line": "National",
    }

    im = None
    for idx, pov_line in enumerate(poverty_lines):
        ax = axes[idx]
        if pov_line not in results_dict:
            ax.axis('off')
            continue

        coef_matrix = results_dict[pov_line]['coef_matrix'].reindex(all_features_raw)
        se_matrix_pl = results_dict[pov_line]['se_matrix'].reindex(all_features_raw)

        im = ax.imshow(coef_matrix.values, cmap='RdBu', aspect='auto', vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10, fontweight='bold')

        if idx % 2 == 0:
            ax.set_yticks(range(len(pretty_labels)))
            ax.set_yticklabels(pretty_labels, fontsize=9)
        else:
            ax.set_yticks(range(len(pretty_labels)))
            ax.set_yticklabels([''] * len(pretty_labels))

        ax.set_title(panel_titles.get(pov_line, pov_line), fontsize=12, fontweight='bold', pad=10)

        for i in range(len(all_features_raw)):
            for j, model in enumerate(models):
                coef = coef_matrix.iloc[i, j]
                if pd.isna(coef) or abs(coef) < 0.001:
                    continue
                se = se_matrix_pl.iloc[i, j]
                color = "white" if abs(coef) > vmax * 0.6 else "black"
                if show_se and pd.notna(se):
                    text = f'{coef:.3f}\n({se:.3f})'
                    fontsize = 6.5
                else:
                    text = f'{coef:.3f}'
                    fontsize = 7.5
                ax.text(j, i, text, ha="center", va="center", color=color,
                        fontsize=fontsize, fontweight='normal', linespacing=0.9)

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Coefficient', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0.04, 0.90, 0.96])

    _out = out_path or outfile
    if _out:
        save_and_show(fig, _out)

    return fig


def plot_coefficient_heatmap_single(
    results_dict: Dict[str, Dict],
    pov_line: str,
    outfile: str,
    poverty_lines: List[str] = None,
    figsize: Tuple[float, float] = (10, 7.5),
    dpi: int = 300,
) -> plt.Figure:
    """One poverty line's coefficient heatmap (Fig. 10 and Figs. A.4-A.6 of the paper).

    Rows and colour scale are shared across the poverty lines, as in the 2x2 grid,
    so the four figures can be compared: a row is a feature significant at any
    line, left blank where it is not significant at this one.
    """
    if poverty_lines is None:
        poverty_lines = ["$1.90/day Poverty Line", "$3.20/day Poverty Line",
                         "$5.50/day Poverty Line", "National Poverty Line"]
    lines = [pl for pl in poverty_lines if pl in results_dict]
    features = sorted(set().union(*(results_dict[pl]['selected_features'] for pl in lines)))
    labels = [PUBLICATION_LABELS.get(f, f) for f in features]
    vmax = max(max(np.nanmax(np.abs(results_dict[pl]['coef_matrix'].reindex(features).values))
                   for pl in lines), 0.1)
    coef = results_dict[pov_line]['coef_matrix'].reindex(features)
    models = coef.columns.tolist()
    titles = {"$1.90/day Poverty Line": "$1.90/day", "$3.20/day Poverty Line": "$3.20/day",
              "$5.50/day Poverty Line": "$5.50/day", "National Poverty Line": "National poverty line"}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(coef.values, cmap='RdBu', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(titles.get(pov_line, pov_line), fontsize=12, fontweight='bold', pad=10)
    for i in range(len(features)):
        for j in range(len(models)):
            c = coef.iloc[i, j]
            if pd.isna(c) or abs(c) < 0.001:
                continue
            ax.text(j, i, f'{c:.3f}', ha="center", va="center", fontsize=7.5,
                    color="white" if abs(c) > vmax * 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('Coefficient', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    save_and_show(fig, outfile)
    return fig


# =====================================================================
# Convenience: run for all poverty lines
# =====================================================================

def run_all_poverty_lines(
    df: pd.DataFrame,
    X_cols: List[str],
    models: List[str] = None,
    poverty_lines: List[str] = None,
    method: str = "glm",
    feature_selection: str = "significant",
    significance_level: float = 0.05,
    top_k: int = 30,
    n_boot: int = 400,
    seed: int = 123,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Run meta-regression for all poverty lines.

    Returns dict mapping poverty_line -> analysis results.
    """
    if poverty_lines is None:
        poverty_lines = [
            "$1.90/day Poverty Line",
            "$3.20/day Poverty Line",
            "$5.50/day Poverty Line",
            "National Poverty Line",
        ]

    results_dict = {}
    for pl in poverty_lines:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Processing: {pl}")
            print(f"{'#'*70}")
        try:
            results = meta_regression_analysis(
                df=df,
                X_cols=X_cols,
                models=models,
                poverty_line=pl,
                method=method,
                feature_selection=feature_selection,
                significance_level=significance_level,
                top_k=top_k,
                n_boot=n_boot,
                seed=seed,
                verbose=verbose,
            )
            results_dict[pl] = results
        except Exception as e:
            if verbose:
                print(f"\nError processing {pl}: {e}")
    return results_dict
