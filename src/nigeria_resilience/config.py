"""Paths, seed, feature lists and poverty lines shared by every script.

Paths resolve from the repository root (or from $NIGERIA_RESILIENCE_ROOT), so
nothing in the package depends on the working directory.  The folder holding
the model outputs (predictions + hyperparameters) can be redirected with
$NIGERIA_RESILIENCE_MODELS_DIR, e.g. to use a retrained set.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("NIGERIA_RESILIENCE_ROOT",
                                   Path(__file__).resolve().parents[2]))

PATHS = {
    "data": PROJECT_ROOT / "data",
    "raw": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "config": PROJECT_ROOT / "config",
    "models": Path(os.environ.get("NIGERIA_RESILIENCE_MODELS_DIR", PROJECT_ROOT / "models")),
    # Predictions recomputed from the fitted models (scripts/model/predict.py),
    # read by every analysis script.
    "predictions": Path(os.environ.get("NIGERIA_RESILIENCE_PREDICTIONS_DIR",
                                       PROJECT_ROOT / "outputs" / "predictions")),
    "results": PROJECT_ROOT / "outputs",
    "figures": PROJECT_ROOT / "outputs" / "figures",
    "tables": PROJECT_ROOT / "outputs" / "tables",
    "reference": PROJECT_ROOT / "reference",
}

# Analysis panel and attrition sample. Both contain GHS-Panel household data, which the
# World Bank's terms do not allow us to redistribute: a replicator builds them with the
# data stage (scripts/data/00_build_extract.py, 01_build_panel.py) into outputs/data/.
# The authors' copies (data/processed/, data/raw/lsms/, not in the repository) are
# used instead when present.
_AUTHORS_PANEL = PATHS["processed"] / "final_data_with_shocks_vf_NGA_for_ML.xlsx"
_AUTHORS_ATTRITION = PATHS["raw"] / "lsms" / "Attrition_sample_for_testing.xlsx"
PANEL_FILE = (_AUTHORS_PANEL if _AUTHORS_PANEL.exists()
              else PATHS["results"] / "data" / "analysis_panel.xlsx")
ATTRITION_FILE = (_AUTHORS_ATTRITION if _AUTHORS_ATTRITION.exists()
                  else PATHS["results"] / "data" / "attrition_sample.xlsx")

SCENARIOS = ("nigeria_temporal", "nigeria_lga_temporal")

SEED = 42

TARGET_COL = "lead_ltotcons_per_capita_per_day_2016cst"

# ---------- Feature lists ----------

list_exp = [
    'ltotcons_per_capita_per_day_2016cst',
    'ltotcons_per_capita_per_day_2016cst_sqr',
    'ltotcons_per_capita_per_day_2016cst_cub',
]

list_hh_characteristcs = [
    'fhh', 'rural',
    'marital_status_head_hh_married_monogamous',
    'marital_status_head_hh_married_polygamous',
    'marital_status_head_hh_widowed',
    'age_head_hh', 'agactivities_hh',
    'prop_female_age_sup_66',
    'prop_female_age_16_65',
    'prop_male_age_16_65',
    'prop_female_age_5_15',
    'prop_male_age_5_15',
    'prop_all_age_0_4',
    'education_level_head_hh_higher', 'education_level_head_hh_others',
    'education_level_head_hh_primary',
    'education_level_head_hh_secondary',
    'own_car',
    'own_computer',
    'own_motorcycle',
    'own_tv',
    'own_house',
    'number_rooms',
    'connection_electricity',
    'saving',
    'loan',
    'main_cook_fuel_firewood',
    'main_cook_fuel_kerosene',
    'main_source_water_borehole',
    'main_source_water_pipe borne',
    'main_source_water_river',
    'main_source_water_well',
    'main_toilet_facility_flush to septic tank',
    'main_toilet_facility_none',
    'main_toilet_facility_pit latrine',
    'main_toilet_facility_v.i.p latrine',
    'farm_area',
]

list_geo_variables = [
    'dist_road', 'dist_market',
    'srtm_nga_5_15_high_altitude_plains',
    'srtm_nga_5_15_hills',
    'srtm_nga_5_15_low_plateaus',
    'srtm_nga_5_15_lowlands',
    'srtm_nga_5_15_mid_altitude_plains',
    'srtm_nga_5_15_plains',
    'srtm_nga_5_15_platform_very_low_plateaus',
    'af_bio_1', 'af_bio_12',
]

list_shocks_hh = [
    'death_adultworking', 'illness_incomearningsmbers',
    'increase_price_input', 'increase_price_food',
]

list_shocks_comm = [
    'event_drought', 'event_flood', 'event_crop_disease', 'event_livestock_disease',
]

list_exog_shocks = [
    'SPI_3_end',
    'NDVI_z_mean', 'battle_dist',
    'protest_dist',
    'riot_dist',
    'vac_dist',
    'battle_time_pre_1_3',
    'battle_time_pre_4_6',
    'protest_time_pre_1_3',
    'protest_time_pre_4_6',
    'riot_time_pre_1_3',
    'riot_time_pre_4_6',
    'vac_time_pre_1_3',
    'vac_time_pre_4_6',
]

timing = [
    'lead_months_since_prev',
    'lead_season_post_harvest',
]

target = ['lead_ltotcons_per_capita_per_day_2016cst']

features_numeric = [
    'ltotcons_per_capita_per_day_2016cst',
    'ltotcons_per_capita_per_day_2016cst_sqr',
    'ltotcons_per_capita_per_day_2016cst_cub',
    'age_head_hh',
    'prop_female_age_sup_66',
    'prop_female_age_16_65',
    'prop_male_age_16_65',
    'prop_female_age_5_15',
    'prop_male_age_5_15',
    'prop_all_age_0_4',
    'dist_road', 'number_rooms', 'farm_area',
    'dist_market', 'af_bio_1',
    'af_bio_12',
    'SPI_3_end',
    'NDVI_z_mean', 'battle_dist',
    'protest_dist',
    'riot_dist',
    'vac_dist', 'lead_months_since_prev',
]

list_all = list_exp + timing + list_hh_characteristcs + list_geo_variables + list_shocks_hh + list_shocks_comm + list_exog_shocks

# ---------- Poverty lines (2011 PPP) ----------

poverty_lines_usd = {
    "1_90": 1.90,
    "3_20": 3.20,
    "5_50": 5.50,
}

# ---------- Model display order ----------

MODEL_ORDER = ["MDN", "XGB-Mix", "NN-1D", "XGB-1D", "Elastic Net", "OLS+GLM"]

MODEL_SUFFIXES = {
    "MDN": "mdn",
    "XGB-Mix": "xgb_md",
    "NN-1D": "nn",
    "XGB-1D": "xgb_1D",
    "Elastic Net": "elastic_net",
    "OLS+GLM": "ols_glm",
}
