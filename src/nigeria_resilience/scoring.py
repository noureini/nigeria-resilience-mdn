"""Inference: predicted distributions, PIT values and resilience scores of the test set.

Used by ``scripts/model/train_models.py`` (on the models it has just saved and
reloaded) and by ``scripts/model/predict.py`` (on the shipped models), so the
published predictions and a replicator's predictions come from the same code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats

from nigeria_resilience.config import TARGET_COL, poverty_lines_usd
from nigeria_resilience.data.loading import load_nigeria_data
from nigeria_resilience.data.splitting import splitting_data
from nigeria_resilience.models import (
    cb_elastic_net,
    cb_ols_glm,
    mdn,
    nn_gaussian,
    xgboost_mixture,
    xgboost_single,
)
from nigeria_resilience.models.seed import setup_seed
from nigeria_resilience.models.xgboost_mixture import (
    compute_national_poverty_line,
    slice_parameter_vectors,
)
from nigeria_resilience.resilience.scores import add_scores_for_line, add_scores_for_national_line

tfd = tfp.distributions
SPLIT_TYPE = {"nigeria_temporal": "temporal", "nigeria_lga_temporal": "lga_temporal"}


def load_split(scenario: str):
    """(full_train, train, validation, test) of a scenario, from the analysis panel."""
    return splitting_data(load_nigeria_data(), split_type=SPLIT_TYPE[scenario])


def _mixture(alpha, mu, sigma):
    return tfd.MixtureSameFamily(
        tfd.Categorical(probs=tf.convert_to_tensor(alpha, tf.float32)),
        tfd.Normal(loc=tf.convert_to_tensor(mu, tf.float32),
                   scale=tf.convert_to_tensor(sigma, tf.float32)))


def _normal_columns(df, name, mu, sd, pit_col):
    df[f"mean_predicted_{name}"] = mu
    df[f"std_predicted_{name}"] = sd
    df[pit_col] = stats.norm.cdf(df[TARGET_COL], df[f"mean_predicted_{name}"],
                                 scale=df[f"std_predicted_{name}"]).round(4)


def score_test_set(models: dict, test_df: pd.DataFrame, seed: int = 42):
    """Predict the six distributions for ``test_df``; return (scored DataFrame, mixtures).

    ``models`` maps ols_glm, elastic_net, xgb_1d, xgb_mix, mdn, nn_1d to the
    artifacts accepted by each model module's ``predict``.
    """
    setup_seed(seed)
    df = test_df.copy()

    mu, s2 = cb_ols_glm.predict(models["ols_glm"], df)
    _normal_columns(df, "ols_glm", mu, np.sqrt(s2), "cdf_ols_glm")
    mu, s2 = cb_elastic_net.predict(models["elastic_net"], df)
    _normal_columns(df, "elastic_net", mu, np.sqrt(s2), "cdf_elastic_net")
    p1 = xgboost_single.predict(models["xgb_1d"], df)
    _normal_columns(df, "xgb_1D", p1["loc"].values, p1["scale"].values, "cdf_xgb_1d")

    y_tf = tf.convert_to_tensor(df[TARGET_COL].values.astype("float32").reshape(-1), tf.float32)

    a_nn, m_nn, s_nn = slice_parameter_vectors(nn_gaussian.predict(models["nn_1d"], df))
    gm_nn = _mixture(a_nn, m_nn, s_nn)
    df["mean_predicted_nn"] = gm_nn.mean().numpy()
    df["std_predicted_nn"] = gm_nn.stddev().numpy()
    df["cdf_nn"] = gm_nn.cdf(y_tf).numpy().round(4)

    a_mdn, m_mdn, s_mdn = slice_parameter_vectors(mdn.predict(models["mdn"], df))
    gm_mdn = _mixture(a_mdn, m_mdn, s_mdn)
    df["mean_predicted_mdn"] = gm_mdn.mean().numpy()
    df["std_predicted_mdn"] = gm_mdn.stddev().numpy()
    df["cdf_mdn"] = gm_mdn.cdf(y_tf).numpy().round(4)

    pm = xgboost_mixture.predict(models["xgb_mix"], df)
    m_mix = pm[[c for c in pm.columns if c.startswith("loc")]].to_numpy(dtype=np.float32)
    s_mix = pm[[c for c in pm.columns if c.startswith("scale")]].to_numpy(dtype=np.float32)
    a_mix = pm[[c for c in pm.columns if c.startswith("mix_prob")]].to_numpy(dtype=np.float32)
    gm_mix = _mixture(a_mix, m_mix, s_mix)
    df["mean_predicted_xgb_MD"] = gm_mix.mean().numpy()
    df["std_predicted_xgb_MD"] = gm_mix.stddev().numpy()
    df["cdf_xgb_md"] = gm_mix.cdf(y_tf).numpy().round(4)

    for label, pl in poverty_lines_usd.items():
        add_scores_for_line(df, label, pl, gm_mdn=gm_mdn, dist_xgb_md=gm_mix, gm_nn=gm_nn)
    add_scores_for_national_line(df, compute_national_poverty_line(),
                                 gm_mdn=gm_mdn, dist_xgb_md=gm_mix, gm_nn=gm_nn)

    mixtures = {"mdn_alpha": a_mdn, "mdn_mu": m_mdn, "mdn_sigma": s_mdn,
                    "nn_alpha": a_nn, "nn_mu": m_nn, "nn_sigma": s_nn,
                    "xgbmix_alpha": a_mix, "xgbmix_mu": m_mix, "xgbmix_sigma": s_mix}
    return df, mixtures


def write_predictions(out: Path, scored: pd.DataFrame, mixtures: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out / "test_scored.parquet", index=False)
    np.savez_compressed(out / "mixture_params.npz", **mixtures)
