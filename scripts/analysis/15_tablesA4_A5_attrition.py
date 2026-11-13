"""Tables A.4 and A.5 -- Balanced panel vs balanced + attritor households (train / test).

Compares the BALANCED panel (main analysis data) against the NON-BALANCED panel
(balanced + attritor households), updated to include ALL current `list_all`
variables -- in particular the new exogenous shock variables (SPI, NDVI, conflict
distances and event counts) -- and reporting Cohen's d.

The new EA-level shock variables, absent from the attritor file, are transferred
onto attritor households from the balanced data by (EA x survey period).

Reads   the analysis panel and the attrition sample (config.PANEL_FILE, ATTRITION_FILE:
        outputs/data/ from the data stage, or the authors' copies)
        data/raw/deflators/{nigeria_ppp,CPI_Nigeria_BS_25022022}.csv
Writes  outputs/tables/TableA4_attrition_train.{tex,csv}
        outputs/tables/TableA5_attrition_test.{tex,csv}
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from nigeria_resilience.attrition.check import (
    combine_data,
    compute_log_consumption_and_prepare_for_analysis,
    convert_consumption_to_constant_naira,
    convert_to_ppp_and_calculate_poverty_dummies,
)
from nigeria_resilience.config import (
    ATTRITION_FILE,
    PATHS,
    TARGET_COL,
    list_exog_shocks,
    list_exp,
    list_geo_variables,
    list_hh_characteristcs,
    list_shocks_comm,
    list_shocks_hh,
)
from nigeria_resilience.data.loading import load_nigeria_data
from nigeria_resilience.data.splitting import splitting_data

TRAIN_YEARS = [2010, 2011, 2012, 2013]
TEST_YEAR = 2015
BASE = 'ltotcons_per_capita_per_day_2016cst'

# --- Ordered variable list + publication labels (published rows + NEW shocks) ---
VAR_LABELS = [
    (TARGET_COL,                                   "Log consumption (next period, t+1) [Outcome]"),
    (BASE,                                         "Log consumption (current period, t)"),
    (BASE + '_sqr',                                "Log consumption (t), squared"),
    (BASE + '_cub',                                "Log consumption (t), cubed"),
    ('lead_months_since_prev',                     "Months to next survey"),
    ('lead_season_post_harvest',                   "Next survey in post-harvest season"),
    ('fhh',                                        "Hh is female"),
    ('rural',                                      "Household live in rural area"),
    ('marital_status_head_hh_married_monogamous',  "Hh is maried and monogamous"),
    ('marital_status_head_hh_married_polygamous',  "Hh is married and polygamous"),
    ('marital_status_head_hh_widowed',             "Hh is widowed"),
    ('age_head_hh',                                "Age of Hh"),
    ('agactivities_hh',                            "Agricultural household"),
    ('prop_female_age_sup_66',                     "Proportion of female (age>66 years)"),
    ('prop_female_age_16_65',                      "Proportion of female (16<=age<=65 years)"),
    ('prop_male_age_16_65',                        "Proportion of male (16<=age<=65 years)"),
    ('prop_female_age_5_15',                       "Proportion of female (5<=age<=15 years)"),
    ('prop_male_age_5_15',                         "Proportion of male (5<=age<=15 years)"),
    ('prop_all_age_0_4',                           "Proportion of individuals (0<age<=4)"),
    ('education_level_head_hh_primary',            "Hh education level is primary"),
    ('education_level_head_hh_secondary',          "Hh education level is secondary"),
    ('education_level_head_hh_higher',             "Hh education level is higher"),
    ('education_level_head_hh_others',             "Hh education level is other"),
    ('own_car',                                    "Household own a car"),
    ('own_computer',                               "Household own a computer"),
    ('own_motorcycle',                             "Household own a motorcycle"),
    ('own_tv',                                     "Household own a tv"),
    ('own_house',                                  "Household own a house"),
    ('number_rooms',                               "Number of rooms in the house"),
    ('connection_electricity',                     "Household has a connection to electricity"),
    ('saving',                                     "Household has saving"),
    ('loan',                                       "Household has loan"),
    ('main_cook_fuel_firewood',                    "Main cooking fuel : Firewood"),
    ('main_cook_fuel_kerosene',                    "Main cooking fuel : Kerosene"),
    ('main_source_water_borehole',                 "Main source of water: Borehole"),
    ('main_source_water_pipe borne',              "Main source of water: Pipe borne"),
    ('main_source_water_river',                    "Main source of water: River"),
    ('main_source_water_well',                     "Main source of water: Well"),
    ('main_toilet_facility_flush to septic tank', "Main toilet facility: Flush to septic tank"),
    ('main_toilet_facility_none',                  "Main toilet facility: None"),
    ('main_toilet_facility_pit latrine',          "Main toilet facility: Pit latrine"),
    ('main_toilet_facility_v.i.p latrine',        "Main toilet facility: v.i.p latrine"),
    ('farm_area',                                  "Farm area"),
    ('dist_road',                                  "Distance to nearest major road"),
    ('dist_market',                                "Distance to nearest market"),
    ('srtm_nga_5_15_high_altitude_plains',         "Terrain Roughness : high altitude plains"),
    ('srtm_nga_5_15_hills',                        "Terrain Roughness : hills"),
    ('srtm_nga_5_15_low_plateaus',                 "Terrain Roughness : low plateaus"),
    ('srtm_nga_5_15_lowlands',                     "Terrain Roughness : lowlands"),
    ('srtm_nga_5_15_mid_altitude_plains',          "Terrain Roughness : mid altitude plains"),
    ('srtm_nga_5_15_plains',                       "Terrain Roughness : plains"),
    ('srtm_nga_5_15_platform_very_low_plateaus',   "Terrain Roughness : platform very low plateaus"),
    ('af_bio_1',                                   "Annual Mean Temperature"),
    ('af_bio_12',                                  "Annual Precipitation"),
    ('death_adultworking',                         "Hh schocks : Death of an adult working member"),
    ('illness_incomearningsmbers',                 "Hh schocks : Illness of income earning member"),
    ('increase_price_input',                       "Hh schocks : Increase in input prices"),
    ('increase_price_food',                        "Hh schocks: Increase in food prices"),
    ('event_drought',                              "Communal shocks: drought"),
    ('event_flood',                                "Communal shocks : floods"),
    ('event_crop_disease',                         "Communal shocks : crop disease"),
    ('event_livestock_disease',                    "Communal shocks : livestock disease"),
    # ---------------- NEW variables (exogenous / conflict / environment) ----------------
    ('SPI_3_end',                                  "Standardized Precipitation Index (3-month)"),
    ('NDVI_z_mean',                                "Vegetation Index (NDVI, z-score)"),
    ('battle_dist',                                "Distance to nearest battle event"),
    ('protest_dist',                               "Distance to nearest protest"),
    ('riot_dist',                                  "Distance to nearest riot"),
    ('vac_dist',                                   "Distance to violence against civilians"),
    ('battle_time_pre_1_3',                        "Battle events (1-3 months prior)"),
    ('battle_time_pre_4_6',                        "Battle events (4-6 months prior)"),
    ('protest_time_pre_1_3',                       "Protest events (1-3 months prior)"),
    ('protest_time_pre_4_6',                       "Protest events (4-6 months prior)"),
    ('riot_time_pre_1_3',                          "Riot events (1-3 months prior)"),
    ('riot_time_pre_4_6',                          "Riot events (4-6 months prior)"),
    ('vac_time_pre_1_3',                           "Violence against civilians (1-3 months prior)"),
    ('vac_time_pre_4_6',                           "Violence against civilians (4-6 months prior)"),
]
NEW_SHOCKS = list_exog_shocks


def stars(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))


def cohen_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return np.nan if (not np.isfinite(sp) or sp == 0) else (x1.mean() - x2.mean()) / sp


def compare(balanced, nonbalanced):
    rows = []
    for col, label in VAR_LABELS:
        if col not in balanced.columns or col not in nonbalanced.columns:
            continue
        b = balanced[col].dropna()
        nb = nonbalanced[col].dropna()
        if len(b) < 2 or len(nb) < 2:
            continue
        t, p = ttest_ind(b, nb, equal_var=True)
        se = np.sqrt(b.std(ddof=1)**2 / len(b) + nb.std(ddof=1)**2 / len(nb))
        rows.append({
            'Variable': label,
            'Mean_balanced': round(b.mean(), 2),
            'Std_balanced': round(b.std(), 2),
            'Mean_nonbalanced': round(nb.mean(), 2),
            'Std_nonbalanced': round(nb.std(), 2),
            'Mean_Difference': round(b.mean() - nb.mean(), 3),
            'Std_Difference': round(se, 3),
            'Significance': stars(p),
            'Cohen_d': round(cohen_d(b, nb), 3),
            'N_balanced': len(b),
            'N_nonbalanced': len(nb),
        })
    return pd.DataFrame(rows)


def to_latex(df, caption, label, n_bal, n_nonbal):
    head = (
        "\\begin{table}[H]\n\\centering\n\\tiny\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}llllllll@{}}\n\\toprule\n"
        f" & \\multicolumn{{2}}{{c}}{{\\textbf{{Balanced (N={n_bal})}}}} "
        f"& \\multicolumn{{2}}{{c}}{{\\textbf{{Non-balanced (N={n_nonbal})}}}} "
        "& \\multicolumn{3}{c}{\\textbf{Difference (T-test)}} \\\\ \\midrule\n"
        "\\textbf{Variable} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Mean} & \\textbf{Std} "
        "& \\textbf{Mean} & \\textbf{Std} & \\textbf{Cohen d} \\\\ \\midrule\n"
    )
    def esc(s):
        return (s.replace('&', '\\&').replace('%', '\\%')
                 .replace('<', '\\textless{}').replace('>', '\\textgreater{}'))
    body = ""
    for _, r in df.iterrows():
        body += (f"{esc(r['Variable']):60s} & {r['Mean_balanced']} & {r['Std_balanced']} & "
                 f"{r['Mean_nonbalanced']} & {r['Std_nonbalanced']} & "
                 f"{r['Mean_Difference']}{r['Significance']} & {r['Std_Difference']} & {r['Cohen_d']} \\\\\n")
    tail = "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    return head + body + tail


def to_latex_twopanel(df, caption, label, n_bal, n_nonbal):
    """Two panels side-by-side (rows split in half) so a long table fits one page."""
    def esc(s):
        return (s.replace('&', '\\&').replace('%', '\\%')
                 .replace('<', '\\textless{}').replace('>', '\\textgreater{}'))
    rows = df.to_dict('records')
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    while len(right) < len(left):
        right.append(None)

    panel_hdr = ("\\textbf{Variable} & \\textbf{M} & \\textbf{S} & \\textbf{M} & \\textbf{S} "
                 "& \\textbf{$\\Delta$M} & \\textbf{$\\Delta$S} & \\textbf{d}")
    grp_hdr = ("& \\multicolumn{2}{c}{\\textbf{Bal.}} & \\multicolumn{2}{c}{\\textbf{Non-bal.}} "
               "& \\multicolumn{3}{c}{\\textbf{Diff. (T-test)}}")
    head = (
        "\\begin{table}[H]\n\\centering\n\\tiny\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\par\\smallskip\\footnotesize Balanced N={n_bal}; Non-balanced N={n_nonbal}. "
        "M=mean, S=std, $\\Delta$M=mean difference (with significance), $\\Delta$S=std. error, d=Cohen's d.\\par\\smallskip\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}llllllll|llllllll@{}}\n\\toprule\n"
        f"{grp_hdr} & {grp_hdr} \\\\\n"
        f"{panel_hdr} & {panel_hdr} \\\\ \\midrule\n"
    )
    def fmt(v):
        return "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v}"
    def cells(r):
        if r is None:
            return " & " * 7 + " "
        return (f"{esc(r['Variable'])} & {fmt(r['Mean_balanced'])} & {fmt(r['Std_balanced'])} & "
                f"{fmt(r['Mean_nonbalanced'])} & {fmt(r['Std_nonbalanced'])} & "
                f"{fmt(r['Mean_Difference'])}{r['Significance']} & {fmt(r['Std_Difference'])} & {fmt(r['Cohen_d'])}")
    body = ""
    for lr, rr in zip(left, right):
        body += cells(lr) + " & " + cells(rr) + " \\\\\n"
    tail = "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    return head + body + tail


def main():
    out_dir = PATHS['tables']
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Balanced sample
    data = load_nigeria_data()
    full_train, _, _, test_df = splitting_data(data, split_type="temporal")
    bal_train, bal_test = full_train.copy(), test_df.copy()

    # 2. Attritor sample
    att = compute_log_consumption_and_prepare_for_analysis(
            convert_to_ppp_and_calculate_poverty_dummies(
            convert_consumption_to_constant_naira(
            combine_data(pd.read_excel(ATTRITION_FILE),
                         pd.read_csv(PATHS['raw'] / 'deflators' / 'nigeria_ppp.csv', sep=';'),
                         pd.read_csv(PATHS['raw'] / 'deflators' / 'CPI_Nigeria_BS_25022022.csv', sep=';')))))
    att[BASE + '_sqr'] = att[BASE] ** 2
    att[BASE + '_cub'] = att[BASE] ** 3

    # 3. Transfer EA x period shock values to attritors
    ea_lookup = data.groupby(['ea', 'surveyprd'])[NEW_SHOCKS].mean().reset_index()
    att = att.merge(ea_lookup, on=['ea', 'surveyprd'], how='left')

    # 4. Non-balanced = balanced UNION attritors (attritors need core covariates)
    core = (list_exp + list_hh_characteristcs + list_geo_variables
            + list_shocks_hh + list_shocks_comm)
    att_ok = att.dropna(subset=[c for c in core if c in att.columns])
    att_train = att_ok[att_ok['surveyprd'].isin(TRAIN_YEARS)]
    att_test = att_ok[att_ok['surveyprd'] == TEST_YEAR]
    nonbal_train = pd.concat([bal_train, att_train], ignore_index=True)
    nonbal_test = pd.concat([bal_test, att_test], ignore_index=True)

    # 5. Comparison tables
    res_train = compare(bal_train, nonbal_train)
    res_test = compare(bal_test, nonbal_test)

    n_bt, n_nbt = len(bal_train), len(nonbal_train)
    n_be, n_nbe = len(bal_test), len(nonbal_test)

    res_train.to_csv(out_dir / 'TableA4_attrition_train.csv', index=False)
    res_test.to_csv(out_dir / 'TableA5_attrition_test.csv', index=False)
    (out_dir / 'TableA4_attrition_train.tex').write_text(
        to_latex_twopanel(res_train, "Comparison of descriptive statistics of the two types of "
                          "training set for Nigeria", "tab:app_tab_attrition_train", n_bt, n_nbt),
        encoding='utf-8')
    (out_dir / 'TableA5_attrition_test.tex').write_text(
        to_latex_twopanel(res_test, "Comparison of descriptive statistics of the two types of "
                          "test set for Nigeria", "tab:app_tab_attrition_test", n_be, n_nbe),
        encoding='utf-8')

    # --- verify the table variable set == actual model feature set (list_all) + target ---
    from nigeria_resilience.config import list_all
    table_vars = [c for c, _ in VAR_LABELS]
    expected = [TARGET_COL] + list(list_all)
    missing_from_table = [v for v in expected if v not in table_vars]
    extra_in_table = [v for v in table_vars if v not in expected]
    print("VARIABLE-SET CHECK vs model features (list_all + target):")
    print(f"  table rows={len(table_vars)}  expected={len(expected)}  "
          f"missing={missing_from_table}  extra={extra_in_table}")
    assert not missing_from_table and not extra_in_table, "Table variables != model features!"
    print("  -> OK: table uses exactly the variables fed to the models.\n")

    print(f"Balanced     train={n_bt}  test={n_be}")
    print(f"Non-balanced train={n_nbt}  test={n_nbe}")
    print(f"Variables compared: {len(res_train)} (incl. {len(NEW_SHOCKS)} new exogenous shocks)")
    print(f"Outputs written to: {out_dir}")
    print("\nNew-shock rows (train):")
    newlabels = {lbl for c, lbl in VAR_LABELS if c in NEW_SHOCKS}
    print(res_train[res_train['Variable'].isin(newlabels)].to_string(index=False))


if __name__ == '__main__':
    main()
