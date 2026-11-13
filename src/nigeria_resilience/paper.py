"""Single source of truth: paper exhibit <-> script <-> output file, and the run order.

Paths are relative to the repository root; output names are relative to
outputs/figures/ (figures, written as .png and .pdf) or outputs/tables/.
"""
from __future__ import annotations

# ---- Run order used by reproduce.py --------------------------------------
DATA_STEPS = [
    "scripts/data/00_build_extract.py",
    "scripts/data/01_build_panel.py",
]

MODEL_STEPS = [
    "scripts/model/train_models.py --scenario nigeria_temporal",
    "scripts/model/train_models.py --scenario nigeria_lga_temporal",
]

PREDICT_STEPS = [
    "scripts/model/predict.py --scenario nigeria_temporal",
    "scripts/model/predict.py --scenario nigeria_lga_temporal",
]

ANALYSIS_STEPS = [
    "scripts/analysis/01_fig2_pit_histograms.py",
    "scripts/analysis/02_fig3_pit_qq.py",
    "scripts/analysis/03_fig4_inverse_quantiles.py",
    "scripts/analysis/04_fig5_figA1_forecast_metrics.py",
    "scripts/analysis/05_fig6_fig7_brier.py",
    "scripts/analysis/06_fig8_resilience_distributions.py",
    "scripts/analysis/07_fig9_mean_resilience_ratios.py",
    "scripts/analysis/08_fig10_resilience_correlates.py",
    "scripts/analysis/09_fig11_figB1_B5_targeting.py",
    "scripts/analysis/10_fig12_fig13_new_lgas.py",
    "scripts/analysis/11_figA2_agreement.py",
    "scripts/analysis/12_figA3_figB7_disagreement.py",
    "scripts/analysis/13_figB6_category_shares.py",
    "scripts/analysis/14_figB8_effort.py",
    "scripts/analysis/15_tablesA4_A5_attrition.py",
    "scripts/analysis/16_tablesA1_A3_specification.py",
]

# ---- Exhibits: (number, content, script, figure outputs, table outputs) ----
EXHIBITS = [
    # Main text
    ("Fig. 1", "Decision tree of the six model specifications (diagram, no data)",
     None, ["docs/Fig1_model_decision_tree.dot"], []),
    ("Fig. 2", "PIT histograms", "01_fig2_pit_histograms.py",
     ["Fig2_pit_histograms"], ["Fig2_pit_values.csv"]),
    ("Fig. 3", "PIT Q-Q plots: all, rural, urban", "02_fig3_pit_qq.py",
     ["Fig3a_all_pit_qq", "Fig3b_rural_pit_qq", "Fig3c_urban_pit_qq"], []),
    ("Fig. 4", "Inverse-quantile plots by household type", "03_fig4_inverse_quantiles.py",
     ["Fig4_inverse_quantiles"], ["Fig4_inverse_quantiles.csv"]),
    ("Fig. 5", "MSE / NLL / CRPS / EMD accuracy relative to the MDN",
     "04_fig5_figA1_forecast_metrics.py",
     ["Fig5_relative_performance_vs_mdn"], ["Fig5_FigA1_forecast_metrics.csv"]),
    ("Fig. 6", "Brier skill score ratios to the MDN", "05_fig6_fig7_brier.py",
     ["Fig6_brier_skill_ratio_vs_mdn"], ["Fig6_brier_skill_ratio.csv"]),
    ("Fig. 7", "Murphy decomposition: reliability vs resolution", "05_fig6_fig7_brier.py",
     ["Fig7_murphy_decomposition"], ["Fig7_murphy_decomposition.csv"]),
    ("Fig. 8", "Distribution (survival function) of predicted resilience scores",
     "06_fig8_resilience_distributions.py",
     ["Fig8_resilience_score_distributions"], ["Fig8_resilience_scores_summary.csv"]),
    ("Fig. 9", "Mean resilience score ratios to the MDN", "07_fig9_mean_resilience_ratios.py",
     ["Fig9_mean_resilience_ratio_vs_mdn"], ["Fig9_mean_resilience_ratio.csv"]),
    ("Fig. 10", "Correlates of the predicted resilience scores, $1.90/day line (GLM, logit link)",
     "08_fig10_resilience_correlates.py",
     ["Fig10_resilience_correlates"], ["Fig10_resilience_correlates.csv"]),
    ("Fig. 11", "Targeting performance at tau = 0.5", "09_fig11_figB1_B5_targeting.py",
     ["Fig11_targeting_tau05"],
     ["Fig11_targeting_summary.csv", "Fig11_targeting_bootstrap_tau05.csv"]),
    ("Fig. 12", "PIT Q-Q plots for new LGAs at t+1", "10_fig12_fig13_new_lgas.py",
     ["Fig12_new_lgas_pit_qq"], []),
    ("Fig. 13", "Forecast metrics relative to the MDN, new LGAs at t+1",
     "10_fig12_fig13_new_lgas.py",
     ["Fig13_new_lgas_relative_performance"], ["Fig13_new_lgas_forecast_metrics.csv"]),
    # Appendix
    ("Table A.1", "Selected neural-network hyperparameters", "16_tablesA1_A3_specification.py",
     [], ["TableA1_nn_hyperparameters.tex", "TableA1_nn_hyperparameters.csv"]),
    ("Table A.2", "Selected XGBoost hyperparameters", "16_tablesA1_A3_specification.py",
     [], ["TableA2_xgb_hyperparameters.tex", "TableA2_xgb_hyperparameters.csv"]),
    ("Table A.3", "List of household covariates", "16_tablesA1_A3_specification.py",
     [], ["TableA3_covariates.tex"]),
    ("Table A.4", "Attrition: balanced vs non-balanced training set",
     "15_tablesA4_A5_attrition.py", [],
     ["TableA4_attrition_train.tex", "TableA4_attrition_train.csv"]),
    ("Table A.5", "Attrition: balanced vs non-balanced test set",
     "15_tablesA4_A5_attrition.py", [],
     ["TableA5_attrition_test.tex", "TableA5_attrition_test.csv"]),
    ("Fig. A.1", "Out-of-sample forecast metrics (levels)", "04_fig5_figA1_forecast_metrics.py",
     ["FigA1_forecast_metrics"], ["Fig5_FigA1_forecast_metrics.csv"]),
    ("Fig. A.2", "Agreement with the MDN resilience classes", "11_figA2_agreement.py",
     ["FigA2_agreement_with_mdn"], ["FigA2_agreement_with_mdn.csv"]),
    ("Fig. A.3", "Type 2 disagreement: household characteristics (SMD)",
     "12_figA3_figB7_disagreement.py",
     ["FigA3_disagreement_type2"], ["FigA3_FigB7_disagreement_smd.csv"]),
    ("Fig. A.4", "Correlates of the predicted resilience scores, $3.20/day line",
     "08_fig10_resilience_correlates.py",
     ["FigA4_resilience_correlates_320"], ["Fig10_resilience_correlates.csv"]),
    ("Fig. A.5", "Correlates of the predicted resilience scores, $5.50/day line",
     "08_fig10_resilience_correlates.py",
     ["FigA5_resilience_correlates_550"], ["Fig10_resilience_correlates.csv"]),
    ("Fig. A.6", "Correlates of the predicted resilience scores, national poverty line",
     "08_fig10_resilience_correlates.py",
     ["FigA6_resilience_correlates_national"], ["Fig10_resilience_correlates.csv"]),
    ("Fig. B.1", "Coverage gap by eligibility threshold", "09_fig11_figB1_B5_targeting.py",
     ["FigB1_coverage_gap"], []),
    ("Fig. B.2", "Exclusion error by threshold", "09_fig11_figB1_B5_targeting.py",
     ["FigB2_exclusion_error"], []),
    ("Fig. B.3", "Inclusion error by threshold", "09_fig11_figB1_B5_targeting.py",
     ["FigB3_inclusion_error"], []),
    ("Fig. B.4", "Precision-recall trade-offs", "09_fig11_figB1_B5_targeting.py",
     ["FigB4_precision_recall"], []),
    ("Fig. B.5", "Targeting under a fixed budget", "09_fig11_figB1_B5_targeting.py",
     ["FigB5_fixed_budget_targeting"], []),
    ("Fig. B.6", "Households across resilience categories", "13_figB6_category_shares.py",
     ["FigB6_resilience_categories"], ["FigB6_resilience_categories.csv"]),
    ("Fig. B.7", "Type 1 disagreement: household characteristics (SMD)",
     "12_figA3_figB7_disagreement.py",
     ["FigB7_disagreement_type1"], ["FigA3_FigB7_disagreement_smd.csv"]),
    ("Fig. B.8", "Effort to reach resilience thresholds, relative to the MDN",
     "14_figB8_effort.py", ["FigB8_effort_vs_mdn"], ["FigB8_effort_vs_mdn.csv"]),
]


def figure_files() -> list[str]:
    return sorted({f for _, _, s, figs, _ in EXHIBITS if s for f in figs})


def table_files() -> list[str]:
    return sorted({t for *_, tabs in EXHIBITS for t in tabs})
