"""Fast checks of the replication package (no model training)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nigeria_resilience import config, paper
from nigeria_resilience.data.splitting import splitting_data
from nigeria_resilience.evaluation.metrics import cdf_accuracy, crps_normal
from nigeria_resilience.models.xgboost_mixture import compute_national_poverty_line
from nigeria_resilience.resilience.scores import cdf_above_threshold_normal

ROOT = Path(__file__).resolve().parents[1]


def test_every_step_script_exists():
    for step in paper.DATA_STEPS + paper.MODEL_STEPS + paper.ANALYSIS_STEPS:
        assert (ROOT / step.split()[0]).is_file(), step


def test_exhibit_map_points_to_analysis_scripts():
    scripts = {Path(s).name for s in paper.ANALYSIS_STEPS}
    for number, _, script, _, _ in paper.EXHIBITS:
        assert script is None or script in scripts, number


def test_every_paper_exhibit_is_mapped():
    numbers = {e[0] for e in paper.EXHIBITS}
    expected = ({f"Fig. {i}" for i in range(1, 14)} | {f"Table A.{i}" for i in range(1, 6)}
                | {f"Fig. A.{i}" for i in range(1, 7)} | {f"Fig. B.{i}" for i in range(1, 9)})
    assert expected == numbers


@pytest.mark.parametrize("scenario", config.SCENARIOS)
def test_shipped_hyperparameters(scenario):
    h = json.loads((ROOT / "config" / "hparams" / f"{scenario}.json").read_text())
    assert h["mdn"]["params"]["n_components"] >= 2
    assert h["nn_1d"]["params"]["n_components"] == 1
    assert h["xgb_mix"]["M"] == 3
    for m in ("mdn", "nn_1d"):
        assert 1.0 < h[m]["target_nll"] < 2.5


def test_national_poverty_line():
    assert compute_national_poverty_line() == pytest.approx(2.54)


def test_feature_list_has_no_duplicates():
    assert len(config.list_all) == len(set(config.list_all))
    assert set(config.features_numeric) <= set(config.list_all)


def test_temporal_and_lga_split_are_disjoint():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"surveyprd": rng.choice([2010, 2011, 2012, 2013, 2015], 2000),
                       "lga_norm": rng.integers(0, 100, 2000)})
    full, tr, va, te = splitting_data(df, split_type="temporal")
    assert set(tr.surveyprd) <= {2010, 2011, 2012} and set(va.surveyprd) == {2013}
    assert set(te.surveyprd) == {2015}
    full, tr, va, te = splitting_data(df, split_type="lga_temporal")
    assert not (set(tr.lga_norm) & set(te.lga_norm))
    assert not (set(va.lga_norm) & set(te.lga_norm))
    assert not (set(tr.lga_norm) & set(va.lga_norm))


def test_scoring_helpers():
    assert cdf_accuracy(np.linspace(0, 1, 10_000)) > 0.99
    assert cdf_accuracy(np.zeros(1000)) < 0.1
    assert crps_normal([0.0], [0.0], [1.0])[0] == pytest.approx(0.2336949, rel=1e-6)
    p = cdf_above_threshold_normal(np.log(1.9), np.log(1.9), 1.0)
    assert float(np.asarray(p)) == pytest.approx(0.5)


@pytest.mark.parametrize("scenario", config.SCENARIOS)
def test_shipped_model_outputs(scenario):
    d = ROOT / "models" / scenario
    if not (d / "test_scored.parquet").exists():
        pytest.skip("model outputs not generated yet")
    df = pd.read_parquet(d / "test_scored.parquet")
    for c in ("cdf_mdn", "resilience_score_mdn_pl_1_90", "resilience_score_ols_glm_at_nat"):
        assert df[c].between(0, 1).all(), c
    z = np.load(d / "mixture_params.npz")
    assert np.allclose(z["mdn_alpha"].sum(axis=1), 1, atol=1e-5)
    assert len(z["mdn_mu"]) == len(df)


@pytest.mark.parametrize("scenario", config.SCENARIOS)
def test_shipped_models_reproduce_their_predictions(scenario):
    """Inference with the shipped model files gives the shipped predictions."""
    from nigeria_resilience.models import store
    from nigeria_resilience.scoring import load_split, score_test_set

    d = ROOT / "models" / scenario
    if not (d / "model").exists():
        pytest.skip("fitted models not shipped yet")
    if not config.PANEL_FILE.exists():
        pytest.skip("analysis panel not built (needs the GHS-Panel files: python reproduce.py --stage data)")
    test_df = load_split(scenario)[3]
    scored, mix = score_test_set(store.load_models(d / "model"), test_df)
    saved = pd.read_parquet(d / "test_scored.parquet")
    cols = [c for c in saved.columns if c.startswith(("mean_predicted_", "cdf_", "resilience_score_"))]
    assert len(cols) > 30
    for c in cols:  # floating-point noise only (PIT values are rounded to 4 decimals)
        assert np.abs(scored[c].to_numpy(float) - saved[c].to_numpy(float)).max() <= 1.001e-4, c
