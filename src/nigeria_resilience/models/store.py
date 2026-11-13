"""Save the six fitted models to portable files, and load them back for inference.

Layout of ``models/<scenario>/model/`` (all text formats except the network weights):

    ols_glm.json        OLS mean and Poisson-GLM variance coefficients (Cissé-Barrett)
    elastic_net.json    two-stage Elastic Net coefficients and feature scaler
    xgb_1d.json         XGBoostLSS booster (Gaussian), XGBoost's JSON format
    xgb_mix.json        XGBoostLSS booster (mixture of M Gaussians)
    xgb_meta.json       start values and settings of the two XGBoostLSS models
    mdn.weights.h5      MDN weights (Keras HDF5)
    nn_1d.weights.h5    NN-1D weights (Keras HDF5)
    networks.json       architecture and feature scaler of the two networks

``load_models`` returns objects that the ``predict`` functions of each model module
accept, so inference runs the same code as the fitted models.  The linear models are
built from their coefficients with the same arithmetic as statsmodels
(``exog @ params``, and ``exp`` for the log-link GLMs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nigeria_resilience.models import mdn
from nigeria_resilience.models.seed import setup_seed
from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.Mixture import Mixture
from xgboostlss.model import XGBoostLSS

FILES = ("ols_glm.json", "elastic_net.json", "xgb_1d.json", "xgb_mix.json", "xgb_meta.json",
         "mdn.weights.h5", "nn_1d.weights.h5", "networks.json")


# ---------------------------------------------------------------------------
# Linear models: coefficients <-> predict()
# ---------------------------------------------------------------------------

class LinearPredictor:
    """``predict(exog)`` of a fitted statsmodels OLS (identity) or log-link GLM."""

    def __init__(self, params, link="identity"):
        self.params = np.asarray(params, dtype=float)
        self.link = link

    def predict(self, exog):
        lin = np.dot(np.asarray(exog, dtype=float), self.params)
        out = np.exp(lin) if self.link == "log" else lin
        return pd.Series(out, index=exog.index) if isinstance(exog, pd.DataFrame) else out


def _params(res):
    return [float(v) for v in np.asarray(res.params, dtype=float)]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _standard_scaler(d):
    if d is None:
        return None
    s = StandardScaler()
    s.mean_, s.scale_ = np.array(d["mean"]), np.array(d["scale"])
    s.var_ = s.scale_ ** 2
    s.n_features_in_ = len(s.mean_)
    s.n_samples_seen_ = d["n_samples_seen"]
    return s


def _minmax_scaler(d):
    s = MinMaxScaler()
    for k in ("min_", "scale_", "data_min_", "data_max_", "data_range_"):
        setattr(s, k, np.array(d[k]))
    s.n_features_in_ = len(s.min_)
    s.feature_names_in_ = np.array(d["feature_names_in_"], dtype=object)
    s.n_samples_seen_ = d["n_samples_seen"]
    return s


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_models(out: Path, ols_art, enet_art, xgb1d_art, xgbmix_art, mdn_art, nn_art) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    ols, glm = ols_art["ols_results"], ols_art["poisson_results"]
    _write_json(out / "ols_glm.json", {
        "explanatory_vars": list(ols_art["explanatory_vars"]),
        "exog_names": list(ols.model.exog_names),
        "mean_params": _params(ols), "variance_params": _params(glm),
        "variance_link": "log"})

    a = enet_art["artifacts"]
    sc = a["scaler"]
    _write_json(out / "elastic_net.json", {
        "explanatory_vars": list(a["explanatory_vars"]),
        "numeric_vars": list(a["numeric_vars"]),
        "add_constant": bool(a["add_constant"]),
        "var_family": a["var_family"],
        "best_hparams": {k: (None if v is None else float(v)) for k, v in a["best_hparams"].items()},
        "scaler": None if sc is None else {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                                           "n_samples_seen": int(sc.n_samples_seen_)},
        "mean_params": _params(enet_art["mean_final"]),
        "variance_params": _params(enet_art["var_final"]),
        "variance_link": "log"})

    meta = {}
    for name, art in (("xgb_1d", xgb1d_art), ("xgb_mix", xgbmix_art)):
        art["model"].booster.save_model(str(out / f"{name}.json"))
        meta[name] = {"start_values": np.asarray(art["model"].start_values, dtype=float).tolist(),
                      "features": list(art["features"]), "M": art.get("M")}
    _write_json(out / "xgb_meta.json", meta)

    nets = {}
    for name, art in (("mdn", mdn_art), ("nn_1d", nn_art)):
        art["model"].save_weights(str(out / f"{name}.weights.h5"))
        s = art["scaler"]
        nets[name] = {
            "params": art["best_params"], "n_features": int(art["model"].input_shape[-1]),
            "features": list(art["features"]), "numeric_features": list(art["numeric_features"]),
            "scaler": {**{k: getattr(s, k).tolist() for k in
                          ("min_", "scale_", "data_min_", "data_max_", "data_range_")},
                       "feature_names_in_": [str(c) for c in s.feature_names_in_],
                       "n_samples_seen": int(s.n_samples_seen_)}}
    _write_json(out / "networks.json", nets)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_models(model_dir: Path) -> dict:
    """Return artifacts for each model module's ``predict``, keyed by model name."""
    d = Path(model_dir)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"model files missing in {d}: {', '.join(missing)}")

    o = _read_json(d / "ols_glm.json")
    ols = {"ols_results": LinearPredictor(o["mean_params"]),
           "poisson_results": LinearPredictor(o["variance_params"], o["variance_link"]),
           "explanatory_vars": o["explanatory_vars"]}

    e = _read_json(d / "elastic_net.json")
    enet = {"mean_final": LinearPredictor(e["mean_params"]),
            "var_final": LinearPredictor(e["variance_params"], e["variance_link"]),
            "artifacts": {"scaler": _standard_scaler(e["scaler"]),
                          "explanatory_vars": e["explanatory_vars"],
                          "numeric_vars": e["numeric_vars"], "add_constant": e["add_constant"],
                          "var_family": e["var_family"], "best_hparams": e["best_hparams"]}}

    meta = _read_json(d / "xgb_meta.json")
    gauss = {"stabilization": "MAD", "response_fn": "softplus", "loss_fn": "nll"}
    out = {"ols_glm": ols, "elastic_net": enet}
    for name in ("xgb_1d", "xgb_mix"):
        m = meta[name]
        dist = (Gaussian(**gauss) if name == "xgb_1d" else
                Mixture(Gaussian(**gauss), M=m["M"], tau=1.0, hessian_mode="individual"))
        model = XGBoostLSS(dist)
        model.booster = xgb.Booster()
        model.booster.load_model(str(d / f"{name}.json"))
        model.start_values = np.array(m["start_values"])
        out[name] = {"model": model, "features": m["features"], "M": m["M"]}

    # Building a network draws initial weights (overwritten by load_weights); with
    # TensorFlow determinism on, those draws need a seed.
    setup_seed(42)
    nets = _read_json(d / "networks.json")
    for name in ("mdn", "nn_1d"):
        n, p = nets[name], nets[name]["params"]
        model = mdn.build_mdn(n_features=n["n_features"], n_hidden=p["n_hidden"],
                              neurons=p["neurons"], n_components=p["n_components"],
                              drop_rate=p["dropout"], l2_reg_v1=p["l2_reg_v1"],
                              l2_reg_v2=p["l2_reg_v2"], learning_rate=p["learning_rate"])
        model.load_weights(str(d / f"{name}.weights.h5"))
        out[name] = {"model": model, "scaler": _minmax_scaler(n["scaler"]),
                     "features": n["features"], "numeric_features": n["numeric_features"]}
    return out
