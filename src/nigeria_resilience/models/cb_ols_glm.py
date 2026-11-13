"""
Cragg & Bardoff (C&B) OLS + GLM two-stage model.

Stage 1: OLS regression for the conditional mean.
Stage 2: Poisson GLM on squared residuals for the conditional variance.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nigeria_resilience.models.seed import setup_seed


# ---------------------------------------------------------------------------
# Significance asterisks (for coefficient reporting)
# ---------------------------------------------------------------------------
def add_significance_asterisks(p_values, coefs):
    asterisks = []
    for p in p_values:
        if p < 0.01:
            asterisks.append('***')
        elif p < 0.05:
            asterisks.append('**')
        elif p < 0.1:
            asterisks.append('*')
        else:
            asterisks.append('')
    return [f'{coef:.3f}{ast}' for coef, ast in zip(coefs, asterisks)]


# ---------------------------------------------------------------------------
# Core two-stage estimator (train on a single dataset)
# ---------------------------------------------------------------------------
def _train_mean_variance_model(data, explanatory_vars, target):
    """
    Train a two-step model:
      Step 1 -- OLS regression for the conditional mean.
      Step 2 -- Poisson GLM on squared residuals for the conditional variance.

    Parameters
    ----------
    data : DataFrame
        Training data (combined train+val for final refit, or train only).
    explanatory_vars : list
        Column names used as regressors.
    target : str
        Name of the dependent variable column.

    Returns
    -------
    ols_results : statsmodels RegressionResults
    poisson_results : statsmodels GLMResults
    """
    exog_vars = data[explanatory_vars]
    exog_vars = sm.add_constant(exog_vars, has_constant='add')

    # Step 1: OLS
    ols_model = sm.OLS(endog=data[target], exog=exog_vars, missing='drop')
    ols_results = ols_model.fit(cov_type='HC1')

    # Squared residuals (target - predicted)
    predicted_mu = ols_results.predict(exog_vars)
    resid_sqr = (data[target].values - predicted_mu.values) ** 2

    # Step 2: Poisson GLM on squared residuals
    poisson_model = sm.GLM(
        endog=resid_sqr, exog=exog_vars,
        family=sm.families.Poisson(), missing='drop'
    )
    poisson_results = poisson_model.fit()

    return ols_results, poisson_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train(train_df, val_df, explanatory_vars, target, seed=42):
    """
    Train the C&B OLS+GLM model.

    The model is first fit on *train_df* alone (for diagnostics / selection),
    then refit on train+val combined for the final model.

    Parameters
    ----------
    train_df : DataFrame
    val_df : DataFrame
    explanatory_vars : list
    target : str
    seed : int

    Returns
    -------
    dict with keys:
        'ols_results'     -- OLS results (final, train+val)
        'poisson_results' -- Poisson GLM results (final, train+val)
        'explanatory_vars'-- list of feature names
        'target'          -- target column name
    """
    setup_seed(seed)

    # Final refit on train + val
    full_train = pd.concat([train_df, val_df], axis=0)
    ols_results, poisson_results = _train_mean_variance_model(
        full_train, explanatory_vars, target
    )

    artifacts = {
        'ols_results': ols_results,
        'poisson_results': poisson_results,
        'explanatory_vars': explanatory_vars,
        'target': target,
    }
    return artifacts


def predict(artifacts, test_df):
    """
    Predict conditional mean (mu) and conditional variance (sigma^2) on new data.

    Parameters
    ----------
    artifacts : dict
        Output of ``train()``.
    test_df : DataFrame

    Returns
    -------
    mu : np.ndarray, shape (n,)
    sigma2 : np.ndarray, shape (n,)
    """
    ols_results = artifacts['ols_results']
    poisson_results = artifacts['poisson_results']
    explanatory_vars = artifacts['explanatory_vars']

    exog = sm.add_constant(test_df[explanatory_vars], has_constant='add')

    mu = ols_results.predict(exog).values
    sigma2 = poisson_results.predict(exog).values
    sigma2 = np.maximum(sigma2, 1e-8)

    return mu, sigma2


def coefficient_tables(artifacts, output_dir=None):
    """
    Build significance-annotated coefficient tables for both stages.

    Parameters
    ----------
    artifacts : dict
        Output of ``train()``.
    output_dir : str or None
        If given, write Excel files to this directory.

    Returns
    -------
    ols_df, poisson_df : DataFrames
    """
    ols_results = artifacts['ols_results']
    poisson_results = artifacts['poisson_results']

    ols_data = {
        'Variable': ols_results.model.exog_names,
        'Coefficient': add_significance_asterisks(ols_results.pvalues, ols_results.params),
        'Standard Error': ols_results.bse,
        'P-Value': ols_results.pvalues,
    }
    poisson_data = {
        'Variable': poisson_results.model.exog_names,
        'Coefficient': add_significance_asterisks(poisson_results.pvalues, poisson_results.params),
        'Standard Error': poisson_results.bse,
        'P-Value': poisson_results.pvalues,
    }

    ols_df = pd.DataFrame(ols_data)
    poisson_df = pd.DataFrame(poisson_data)

    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        ols_df.to_excel(os.path.join(output_dir, "ols_output_with_significance_temporal.xlsx"))
        poisson_df.to_excel(os.path.join(output_dir, "poisson_output_with_significance_temporal.xlsx"))

    return ols_df, poisson_df
