"""Build the GHS-Panel household extract from the World Bank and EPAR files.

The extract has one row per household and survey round (2010, 2011, 2012, 2013,
2015, 2016) and the 82 columns listed in EXTRACT_COLUMNS. Decimal survey values
(stored by the World Bank as float32) are written with 8 significant digits.

Inputs (see data/README.md for how to obtain them; checksums in data/checksums.csv):

* World Bank, Nigeria General Household Survey-Panel, waves 1-3 (Stata files),
  anywhere under data/raw/ghs/ with their original file names;
* EPAR household indicators (output of EPAR's Stata code on the same files),
  data/raw/epar/Nigeria_GHS_W{1,2,3}_household_variables.dta;
* data/sample/attrition_rows.csv (household-rounds of the attrition comparison,
  Tables A.4-A.5).

The 3,441 households of the estimation sample are selected by select_sample()
(missing data, same household, 3-s.d. outlier screen, balanced panel; see the
comments above it); data/sample/extract_households.csv lists them.

Every coding decision is stated where it is made (household 80143's livestock
units, the education of person no. 1, the grouping of education levels, rooms in
wave 3...).
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from nigeria_resilience.config import PATHS

GHS_DIR = PATHS["raw"] / "ghs"
EPAR_DIR = PATHS["raw"] / "epar"
SAMPLE_DIR = PATHS["data"] / "sample"

PRD = {(1, 1): 2010, (1, 2): 2011, (2, 1): 2012, (2, 2): 2013, (3, 1): 2015, (3, 2): 2016}
SEASON = {1: "post_planting", 2: "post_harvest"}
YEAR = {1: "2008-09", 2: "2012-13", 3: "2015-16"}        # survey-year labels of the extract
VISIT_NAME = {1: "planting", 2: "harvest"}
KEY = ["hhid", "surveyprd"]

# Column order of the extract.
EXTRACT_COLUMNS = [
    "zone_norm", "state_norm", "lga_norm", "ea", "hhid", "surveyprd", "rururb", "hhsize",
    "totcons", "hhweight", "season", "year", "wave", "totcons_per_capita_per_day",
    "ln_totcons_per_capita_per_day", "rural", "dist_road", "dist_market", "af_bio_1",
    "af_bio_8", "af_bio_12", "srtm_nga_5_15", "sq7", "fhh", "ag_hh", "agactivities_hh",
    "livestock_hh", "religion_head_hh", "death_adultworking", "death_sendremittances",
    "illness_incomearningsmbers", "faillure_non_farmbusiness", "poor_rain", "flooding",
    "death_lvst", "increase_price_input", "increase_price_food", "event_drought",
    "event_flood", "event_crop_disease", "event_livestock_disease", "event_human_epidemic",
    "event_sharp_change_price", "event_dev_project", "event_health_facilities",
    "event_new_road", "event_new_school", "crop_income", "lvstck_holding_tlu",
    "livestock_income", "farm_area", "value_assets", "crop_livestock_hh", "w_value_assets",
    "nonfarm_income", "sector", "prop_female_age_sup_66", "prop_male_age_sup_66",
    "prop_female_age_16_65", "prop_male_age_16_65", "prop_female_age_5_15",
    "prop_male_age_5_15", "prop_all_age_0_4", "prop_female_age_0_4", "prop_male_age_0_4",
    "hh_size", "sex_head_hh", "age_head_hh", "marital_status_head_hh",
    "education_level_head_hh", "own_car", "own_computer", "own_motorcycle", "own_tv",
    "own_house", "number_rooms", "main_cook_fuel", "connection_electricity",
    "main_source_water", "main_toilet_facility", "saving", "loan"]

# World Bank files read, by wave (file names as distributed by the Microdata Library).
GHS_FILES = {
    w: [f"cons_agg_wave{w}_visit1", f"cons_agg_wave{w}_visit2", f"sect15a_harvestw{w}",
        f"sectc6_harvestw{w}", f"sect1_plantingw{w}", f"sect1_harvestw{w}"]
    for w in (1, 2, 3)}
GHS_FILES[1] += ["NGA_HouseholdGeovariables_Y1", "sect2_plantingw1", "sect8_harvestw1",
                 "sect5_plantingw1", "sect4_plantingw1"]
GHS_FILES[2] += ["NGA_HouseholdGeovars_Y2", "sect2_plantingw2", "sect8_harvestw2",
                 "sect5a_plantingw2", "sect4a_plantingw2"]
GHS_FILES[3] += ["NGA_HouseholdGeovars_Y3", "sect2_harvestw3", "sect11_plantingw3",
                 "sect5_plantingw3", "sect4a_plantingw3", "sect4c1_plantingw3"]
EPAR_FILES = [f"Nigeria_GHS_W{w}_household_variables" for w in (1, 2, 3)]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def module(name: str) -> Path:
    """Path of a World Bank file (e.g. 'sect15a_harvestw1') under data/raw/ghs/."""
    hits = [p for p in GHS_DIR.rglob(f"{name}.dta")]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"{name}.dta: expected one file under {GHS_DIR}, found {len(hits)}. "
            "See data/README.md for the GHS-Panel files to download.")
    return hits[0]


def read(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A Stata file with numeric codes, and the same file with value labels.

    Labels are mapped by hand: some World Bank files give one label to two codes
    (two LGAs are called "surulere"), which pandas cannot turn into categoricals.
    """
    codes = pd.read_stata(path, convert_categoricals=False)
    with pd.io.stata.StataReader(path) as r:
        r.read(convert_categoricals=False)
        tables, lbl = r.value_labels(), dict(zip(r.varlist, r.lbllist))
    labels = codes.copy()
    for c in codes.columns:
        t = tables.get(lbl.get(c, ""))
        if t:
            labels[c] = codes[c].map(lambda v, t=t: t.get(v, v) if pd.notna(v) else np.nan)
    return codes, labels


def both_visits(d: pd.DataFrame, w: int) -> pd.DataFrame:
    """Repeat a wave-level table for the wave's two survey rounds."""
    return pd.concat([d.assign(surveyprd=PRD[(w, v)]) for v in (1, 2)], ignore_index=True)


# ---------------------------------------------------------------------------
# Formatting conventions
# ---------------------------------------------------------------------------

RENAME = {"cross-rivers": "cross rivers", "kogi(k.k)": "kogi"}


def norm_label(s) -> str:
    s = re.sub(r"^\s*\d+\.\s*", "", str(s)).strip().lower()
    return RENAME.get(s, s)


def snake(s) -> str:
    s = re.sub(r"[()\-/,]+", " ", norm_label(s))
    return re.sub(r"\s+", "_", s.strip())


def g8(v) -> float:
    """A float32 survey value written with 8 significant digits, rounded half up."""
    if pd.isna(v):
        return np.nan
    d = Decimal(float(v))
    if d == 0:
        return 0.0
    return float(d.quantize(Decimal(1).scaleb(d.adjusted() - 7), rounding=ROUND_HALF_UP))


def round2(x):
    """Round half up to 2 decimals (0.125 -> 0.13)."""
    return np.floor(x * 100 + 0.5 + 1e-9) / 100


def _map(series, table, default):
    return series.map(lambda s: table.get(norm_label(s), default) if pd.notna(s) else default)


# ---------------------------------------------------------------------------
# 1. Location, consumption and weights (consumption aggregates)
# ---------------------------------------------------------------------------

def consumption() -> pd.DataFrame:
    rows = []
    for w in (1, 2, 3):
        for v in (1, 2):
            num, lab = read(module(f"cons_agg_wave{w}_visit{v}"))
            rows.append(pd.DataFrame({
                "zone_norm": lab.zone.map(norm_label), "state_norm": lab.state.map(norm_label),
                "lga_norm": lab.lga.map(norm_label), "ea": num.ea.astype(int),
                "hhid": num.hhid.astype(int), "surveyprd": PRD[(w, v)],
                "rururb": lab.rururb.astype(str), "hhsize": num.hhsize.astype(int),
                "totcons": num.totcons.map(g8), "hhweight": num.hhweight.map(g8),
                "season": SEASON[v], "year": YEAR[w], "wave": f"wave{w}"}))
    d = pd.concat(rows, ignore_index=True)
    d["totcons_per_capita_per_day"] = d.totcons / 365   # totcons is per person and year
    d["ln_totcons_per_capita_per_day"] = np.log(d.totcons_per_capita_per_day)
    d["sector"] = d.rururb.str.lower()
    return d


def households(w: int) -> pd.DataFrame:
    """hhid, lga and ea of every household with a consumption aggregate in wave w."""
    rows = [pd.read_stata(module(f"cons_agg_wave{w}_visit{v}"), convert_categoricals=False,
                          columns=["hhid", "lga", "ea"]) for v in (1, 2)]
    return pd.concat(rows).drop_duplicates("hhid").astype(int).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Geovariables (dist_road is the wave-1 value in every round)
# ---------------------------------------------------------------------------

GEO_FILE = {1: "NGA_HouseholdGeovariables_Y1", 2: "NGA_HouseholdGeovars_Y2", 3: "NGA_HouseholdGeovars_Y3"}
SRTM_KEEP = {"plains", "mid_altitude_plains", "lowlands", "platform_very_low_plateaus",
             "high_altitude_plains", "low_plateaus", "hills"}   # other terrain -> "others"


def geovariables() -> pd.DataFrame:
    out, road = [], None
    for w in (1, 2, 3):
        num, lab = read(module(GEO_FILE[w]))
        srtm = lab.srtm_nga_5_15.map(lambda v: snake(v) if pd.notna(v) else np.nan)
        d = pd.DataFrame({"hhid": num.hhid.astype(int),
                          "dist_market": num.dist_market.astype(float),
                          "af_bio_1": num.af_bio_1.astype(float) / 10,
                          "af_bio_8": num.af_bio_8, "af_bio_12": num.af_bio_12,
                          "srtm_nga_5_15": srtm.where(srtm.isin(SRTM_KEEP) | srtm.isna(), "others"),
                          "sq7": lab.sq7.map(lambda v: snake(v) if pd.notna(v) else np.nan)})
        if w == 1:
            road = num[["hhid", "dist_road"]].astype({"hhid": int})
        out.append(both_visits(d, w))
    return pd.concat(out, ignore_index=True).merge(road, on="hhid", how="left")


# ---------------------------------------------------------------------------
# 3. EPAR household indicators (one value per wave)
# ---------------------------------------------------------------------------

EPAR_VARS = ["rural", "fhh", "ag_hh", "agactivities_hh", "livestock_hh", "crop_income",
             "lvstck_holding_tlu", "livestock_income", "farm_area", "value_assets",
             "w_value_assets", "nonfarm_income"]
# Data-entry correction: hhid 80143 reports 4,000 small ruminants in wave 1 (5 and
# 11 in waves 2 and 3), which gives 1,155 TLU; we use 1.16.
TLU_CORRECTIONS = {(80143, 1): 1.16}


def epar_content_hash(w: int) -> str:
    """SHA-256 of the EPAR variables used here, by household (independent of the file's
    save date, which Stata writes into every .dta file). Lines end in "\\n" on every
    operating system (pandas otherwise uses the system's line ending)."""
    import hashlib
    d = pd.read_stata(EPAR_DIR / f"Nigeria_GHS_W{w}_household_variables.dta",
                      convert_categoricals=False)[["hhid", "crop_hh"] + EPAR_VARS]
    d = d.astype({"hhid": "int64"}).sort_values("hhid").reset_index(drop=True)
    text = d.to_csv(index=False, float_format="%.9g", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def epar() -> pd.DataFrame:
    out = []
    for w in (1, 2, 3):
        d = pd.read_stata(EPAR_DIR / f"Nigeria_GHS_W{w}_household_variables.dta",
                          convert_categoricals=False)[["hhid", "crop_hh"] + EPAR_VARS].copy()
        d["hhid"] = d.hhid.astype(int)
        # Households with neither crops nor livestock are coded "crop_and_livestock".
        d["crop_livestock_hh"] = np.select(
            [(d.crop_hh == 1) & (d.livestock_hh == 1), d.crop_hh == 1, d.livestock_hh == 1],
            ["crop_and_livestock", "only_crop", "only_livestock"], default="crop_and_livestock")
        for (hh, wave), v in TLU_CORRECTIONS.items():
            if wave == w:
                d.loc[d.hhid == hh, "lvstck_holding_tlu"] = v
        out.append(both_visits(d.drop(columns="crop_hh"), w))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Household shocks (post-harvest shock module)
# ---------------------------------------------------------------------------

SHOCKS = {1: "death_adultworking", 2: "death_sendremittances", 3: "illness_incomearningsmbers",
          8: "faillure_non_farmbusiness", 12: "poor_rain", 13: "flooding", 17: "death_lvst",
          18: "increase_price_input", 20: "increase_price_food"}


def shocks() -> pd.DataFrame:
    """1 if the household reports the shock (s15aq1 == yes) in the wave."""
    out = []
    for w in (1, 2, 3):
        n, _ = read(module(f"sect15a_harvestw{w}"))
        yes = n[n.s15aq1 == 1]
        d = households(w)[["hhid"]].copy()
        for k, name in SHOCKS.items():
            d[name] = d.hhid.isin(set(yes.loc[yes.shock_cd == k, "hhid"].astype(int))).astype(int)
        out.append(both_visits(d, w))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Community events (post-harvest community key-events module)
# ---------------------------------------------------------------------------

EVENTS = {1: "event_drought", 2: "event_flood", 3: "event_crop_disease",
          4: "event_livestock_disease", 5: "event_human_epidemic",
          6: "event_sharp_change_price", 11: "event_dev_project",
          13: "event_health_facilities", 14: "event_new_road", 15: "event_new_school"}


def community() -> pd.DataFrame:
    """1 if the household's community (lga, ea) reports the event.

    Events whose share of the community affected is 0 or 450 (a coding error) are
    dropped.
    """
    out = []
    for w in (1, 2, 3):
        n = pd.read_stata(module(f"sectc6_harvestw{w}"), convert_categoricals=False)
        share = "sc6q3" if "sc6q3" in n else "c6q3"
        n = n[n.event_cd.isin(EVENTS) & ~n[share].isin([0, 450])]
        d = households(w)
        for k, name in EVENTS.items():
            ev = n.loc[n.event_cd == k, ["lga", "ea"]].drop_duplicates().astype(int).assign(**{name: 1})
            d = d.merge(ev, on=["lga", "ea"], how="left")
            d[name] = d[name].fillna(0).astype(int)
        out.append(both_visits(d.drop(columns=["lga", "ea"]), w))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 6. Household roster: size, composition, head
# ---------------------------------------------------------------------------

ROSTER_AGE = {(1, 1): "s1q4", (1, 2): "s1q4", (2, 1): "s1q6", (2, 2): "s1q4", (3, 1): "s1q6", (3, 2): "s1q4"}
ROSTER_MARITAL = {(1, 1): "s1q8", (1, 2): "s1q7", (2, 1): "s1q8", (2, 2): "s1q7", (3, 1): "s1q8", (3, 2): "s1q7"}
AGE_GROUPS = {"0_4": (0, 4), "5_15": (5, 15), "16_65": (16, 65), "sup_66": (66, 999)}
MARITAL = {"married (monogamous)": "married_monogamous", "married (polygamous)": "married_polygamous",
           "widowed": "widowed"}                                  # other statuses -> "others"
RELIGION = {"christianity": "christianity", "islam": "islam", "traditional": "traditionnal",
            "others": "others"}                                   # category labels used by the models
# Highest level completed, in four groups ("1st degree" and "lower 6" are grouped
# with "primary", "nce" with "secondary"). Other levels -> "others"; never attended
# (missing) -> "none".
EDUC_SOURCE = {1: ("sect2_plantingw1", "s2q7", 1), 2: ("sect2_plantingw2", "s2q8", 1),
               3: ("sect2_harvestw3", "s2aq9", 2)}
EDUCATION = {"none": "none", "higher degree": "higher",
             **{f"p{i}": "primary" for i in range(1, 7)}, "1st degree": "primary", "lower 6": "primary",
             **{f"js{i}": "secondary" for i in (1, 2, 3)}, **{f"ss{i}": "secondary" for i in (1, 2, 3)},
             "modern school": "secondary", "nce": "secondary", "upper 6": "secondary"}


def roster(w: int, v: int):
    return read(module(f"sect1_{VISIT_NAME[v]}w{w}"))


def heads(w: int, v: int):
    """Roster line of the household head (relationship code 1)."""
    n, lab = roster(w, v)
    h = n[n.s1q3 == 1]
    h = h[~h.hhid.duplicated()]
    return h, lab.loc[h.index]


def demographics() -> pd.DataFrame:
    """Household size and age-sex shares from the roster of each visit.

    hh_size counts every roster line; each share is (members in the group) / hh_size,
    rounded half up to 2 decimals. prop_all_age_0_4 is the residual 1 - (sum of the
    six rounded shares of ages 5 and over), floored at 0.
    """
    out = []
    for (w, v), age in ROSTER_AGE.items():
        n, _ = roster(w, v)
        size = n.groupby("hhid").size()
        d = pd.DataFrame({"hh_size": size})
        for sex, code in (("female", 2), ("male", 1)):
            for g, (lo, hi) in AGE_GROUPS.items():
                member = n[age].between(lo, hi) & (n.s1q2 == code)
                d[f"prop_{sex}_age_{g}"] = round2(member.groupby(n.hhid).sum() / size)
        older = d[[c for c in d if c.startswith("prop_") and not c.endswith("0_4")]].sum(axis=1)
        d["prop_all_age_0_4"] = round2(1 - older).clip(lower=0)
        out.append(d.reset_index().astype({"hhid": int}).assign(surveyprd=PRD[(w, v)]))
    return pd.concat(out, ignore_index=True)


def head() -> pd.DataFrame:
    """Sex, age and marital status of the head (each visit), religion (wave 1), education (each wave)."""
    n1, l1 = heads(1, 1)
    religion = pd.Series(
        l1.s1q12.map(lambda s: RELIGION.get(norm_label(s), np.nan) if pd.notna(s) else np.nan).values,
        index=n1.hhid.astype(int).values)
    # Education is that of person no. 1 on the roster, the household head in all but
    # about 10 households (where person no. 1 is the spouse or another member).
    educ = {}
    for w, (mod, q, v) in EDUC_SOURCE.items():
        rn, _ = roster(w, v)
        first = rn.loc[rn.indiv == 1, ["hhid", "indiv"]]
        _, el = read(module(mod))
        e = first.merge(el[["hhid", "indiv", q]], on=["hhid", "indiv"], how="left").drop_duplicates("hhid")
        educ[w] = pd.Series(
            e[q].map(lambda s: EDUCATION.get(norm_label(s), "others") if pd.notna(s) else "none").values,
            index=e.hhid.astype(int).values)
    out = []
    for (w, v), age in ROSTER_AGE.items():
        hn, hl = heads(w, v)
        d = pd.DataFrame({
            "hhid": hn.hhid.astype(int).values,
            "sex_head_hh": hl.s1q2.map(norm_label).values,
            "age_head_hh": hn[age].astype(float).values,
            "marital_status_head_hh": _map(hl[ROSTER_MARITAL[(w, v)]], MARITAL, "others").values})
        d["religion_head_hh"] = d.hhid.map(religion)
        d["education_level_head_hh"] = d.hhid.map(educ[w])
        out.append(d.assign(surveyprd=PRD[(w, v)]))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# 7. Dwelling, durable assets, savings and loans (one value per wave)
# ---------------------------------------------------------------------------

HOUSING = {1: ("sect8_harvestw1", "s8q1", "s8q9", "s8q11", "s8q17", "s8q33a", "s8q36a"),
           2: ("sect8_harvestw2", "s8q1", "s8q9", "s8q11", "s8q17", "s8q33a", "s8q36"),
           3: ("sect11_plantingw3", "s11q1", "s11q9", "s11q11", "s11q17b", "s11q33a", "s11q36")}
COOK_FUEL = {"collected firewood": "firewood", "purchased firewood": "firewood", "kerosene": "kerosene"}
WATER = {"bore hole/hand pump": "borehole", "pipe borne water treated": "pipe borne",
         "pipe borne water untreated": "pipe borne", "river/spring": "river",
         "well/spring protected": "well", "well/spring unprotected": "well"}
TOILET = {"flush to septic tank": "flush to septic tank", "covered pit latrine": "pit latrine",
          "uncovered pit latrine": "pit latrine", "v.i.p latrine": "v.i.p latrine", "none": "none"}
ASSET_MODULE = {1: "sect5_plantingw1", 2: "sect5a_plantingw2", 3: "sect5_plantingw3"}
ASSET_ITEMS = {"own_car": 319, "own_computer": 328, "own_motorcycle": 318, "own_tv": 327}
# Any member: bank account, savings institution or informal savings group (saving);
# borrowed from an institution, an informal group, friends or relatives (loan).
FINANCE = {1: ("sect4_plantingw1", ["s4q1", "s4q4", "s4q6"], ["s4q7", "s4q9", "s4q10"]),
           2: ("sect4a_plantingw2", ["s4aq1", "s4aq8", "s4aq10"], ["s4aq11", "s4aq13", "s4aq14"]),
           3: ("sect4a_plantingw3", ["s4aq1", "s4aq8", "s4aq10"], None)}


def assets() -> pd.DataFrame:
    out = []
    for w in (1, 2, 3):
        mod, own, rooms, fuel, elec, water, toilet = HOUSING[w]
        n, lab = read(module(mod))
        keep = ~n.hhid.duplicated()
        n, lab = n[keep], lab[keep]
        d = pd.DataFrame({"hhid": n.hhid.astype(int).values,
                          "own_house": (n[own] == 1).astype(int).values,
                          "number_rooms": n[rooms].astype(float).values,
                          "main_cook_fuel": _map(lab[fuel], COOK_FUEL, "others").values,
                          "connection_electricity": (n[elec] == 1).astype(int).values,
                          "main_source_water": _map(lab[water], WATER, "others").values,
                          "main_toilet_facility": _map(lab[toilet], TOILET, "other").values})
        if w == 3:   # a missing number of rooms is coded 0 in wave 3
            d["number_rooms"] = d.number_rooms.fillna(0)
        a, _ = read(module(ASSET_MODULE[w]))
        for var, code in ASSET_ITEMS.items():
            owners = a.loc[(a.item_cd == code) & (a.s5q1 > 0), "hhid"].astype(int)
            d[var] = d.hhid.isin(set(owners)).astype(int)
        mod, sav, loan = FINANCE[w]
        f, _ = read(module(mod))
        d["saving"] = d.hhid.isin(set(f.loc[(f[sav] == 1).any(axis=1), "hhid"].astype(int))).astype(int)
        if loan:
            d["loan"] = d.hhid.isin(set(f.loc[(f[loan] == 1).any(axis=1), "hhid"].astype(int))).astype(int)
        else:        # wave 3 asks the household directly
            c, _ = read(module("sect4c1_plantingw3"))
            d["loan"] = d.hhid.isin(set(c.loc[c.s4cq1 == 1, "hhid"].astype(int))).astype(int)
        out.append(both_visits(d, w))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

GROUPS = (geovariables, epar, shocks, community, demographics, head, assets)


def all_household_rounds() -> pd.DataFrame:
    """Every household-round with a consumption aggregate, with all extract columns."""
    d = consumption()
    for group in GROUPS:
        d = d.merge(group(), on=KEY, how="left")
    return d[EXTRACT_COLUMNS]


# ---------------------------------------------------------------------------
# The estimation sample (paper Section 2.1)
# ---------------------------------------------------------------------------
# Household-rounds are screened first; a household is kept only if all six rounds
# survive (a balanced panel). The rules, in order:
#
#   1. missing data: drop a household-round whose household is absent from the
#      dwelling, assets, savings/loans or education module of that wave, or whose
#      geovariables (agro-ecological zone sq7, mean temperature) or EPAR
#      female-head indicator are missing;
#   2. the same household: drop a household whose person no. 1 on the roster has
#      left (wave 2 post-planting, wave 3 post-harvest) or is missing from the
#      roster, or that changes EA, LGA or urban/rural sector between rounds
#      (sector = the GHS "rururb"/cover-page sector, not EPAR's "rural");
#   3. outliers: drop a household-round whose log per-capita consumption is more
#      than 3 standard deviations from the mean of its survey round and sector;
#      the mean and s.d. are those of the households observed in all six rounds
#      after steps 1-2;
#   4. head's age: drop the wave-1 rounds of the two households whose head's age
#      in wave 1 is inconsistent with waves 2 and 3;
#   5. balanced panel: keep the households with all six rounds.
#
# The result is the 3,441 households of the paper, listed in
# data/sample/extract_households.csv.

ROSTER_MEMBER = {(1, 1): "s1q7", (1, 2): "s1q19", (2, 1): "s1q4", (2, 2): "s1q14",
                 (3, 1): "s1q4", (3, 2): "s1q4a"}            # "still a member?" (2 = no)
LINE1_LEFT_ROUNDS = [(2, 1), (3, 2)]                        # rounds whose answer is used
MODULES_REQUIRED = {"dwelling": HOUSING, "assets": ASSET_MODULE, "savings/loans": FINANCE,
                    "education": EDUC_SOURCE}
HEAD_AGE_CORRECTION = {80022: (2010, 2011), 180071: (2010, 2011)}
OUTLIER_SD = 3.0


def _module_hhids(name: str) -> set:
    return set(pd.read_stata(module(name), columns=["hhid"]).hhid.dropna().astype(int))


def missing_rounds() -> set:
    """(hhid, surveyprd) of household-rounds with missing data (rule 1)."""
    out = set()
    for w in (1, 2, 3):
        hh = set(households(w).hhid)
        bad = set()
        for spec in MODULES_REQUIRED.values():
            m = spec[w][0] if isinstance(spec[w], tuple) else spec[w]
            bad |= hh - _module_hhids(m)
        geo = pd.read_stata(module(GEO_FILE[w]), convert_categoricals=False)
        bad |= hh - set(geo.hhid.astype(int))
        bad |= set(geo.loc[geo.sq7.isna() | geo.af_bio_1.isna(), "hhid"].astype(int))
        ep = pd.read_stata(EPAR_DIR / f"Nigeria_GHS_W{w}_household_variables.dta",
                           convert_categoricals=False, columns=["hhid", "fhh"])
        bad |= set(ep.loc[ep.fhh.isna(), "hhid"].astype(int))
        out |= {(h, PRD[(w, v)]) for h in bad for v in (1, 2)}
    return out


def person1_gone() -> set:
    """Households whose person no. 1 has left or is missing from a roster (rule 2)."""
    out = set()
    for (w, v), q in ROSTER_MEMBER.items():
        n, _ = roster(w, v)
        first = n[n.indiv == 1].drop_duplicates("hhid")
        out |= set(households(w).hhid) - set(first.hhid.astype(int))
        if (w, v) in LINE1_LEFT_ROUNDS:
            out |= set(first.loc[first[q] == 2, "hhid"].astype(int))
    return out


def _balanced(d: pd.DataFrame) -> pd.DataFrame:
    n = d.groupby("hhid").surveyprd.nunique()
    return d[d.hhid.isin(n[n == len(PRD)].index)]


def select_sample(d: pd.DataFrame | None = None, verbose: bool = True) -> tuple[set, pd.DataFrame]:
    """Households of the estimation sample, and the count after each rule."""
    d = all_household_rounds() if d is None else d
    steps = [("household-rounds with a consumption aggregate", d)]
    key = pd.MultiIndex.from_frame(d[KEY])
    d1 = d[~key.isin(list(missing_rounds()))]
    steps.append(("1. missing data removed", d1))
    place = d.groupby("hhid").agg(e=("ea", "nunique"), l=("lga_norm", "nunique"), s=("sector", "nunique"))
    same_place = set(place.index[(place.e == 1) & (place.l == 1) & (place.s == 1)])
    d2 = d1[~d1.hhid.isin(person1_gone()) & d1.hhid.isin(same_place)]
    steps.append(("2. person no. 1 still a member; same EA, LGA and sector", d2))
    y = "ln_totcons_per_capita_per_day"
    base = _balanced(d2)          # mean and s.d.: households observed in all six rounds
    st = base.groupby(["surveyprd", "sector"])[y].agg(["mean", "std"])
    s = d2.join(st, on=["surveyprd", "sector"])
    d3 = d2[((d2[y] - s["mean"]) / s["std"]).abs() < OUTLIER_SD]
    steps.append((f"3. outliers: rounds beyond {OUTLIER_SD:g} s.d. of log consumption removed", d3))
    drop = {(h, p) for h, ps in HEAD_AGE_CORRECTION.items() for p in ps}
    d4 = d3[[(h, p) not in drop for h, p in zip(d3.hhid, d3.surveyprd)]]
    steps.append(("4. wave-1 rounds with inconsistent head age removed (2 households)", d4))
    d5 = _balanced(d4)
    steps.append(("5. balanced panel: households with all six rounds", d5))
    report = pd.DataFrame([(name, len(x), x.hhid.nunique()) for name, x in steps],
                          columns=["step", "household-rounds", "households"])
    if verbose:
        print(report.to_string(index=False))
    return set(d5.hhid), report


def build(panel: pd.DataFrame | None = None):
    """(extract, attrition sample): the sample is selected by select_sample()."""
    d = all_household_rounds() if panel is None else panel
    hh, _ = select_sample(d)
    published = set(pd.read_csv(SAMPLE_DIR / "extract_households.csv").hhid)
    if hh != published:
        print(f"WARNING: the selection gives {len(hh)} households; {len(hh - published)} not in "
              f"data/sample/extract_households.csv, {len(published - hh)} of it missing.")
    else:
        print(f"  {len(hh)} households selected, as listed in data/sample/extract_households.csv")
    extract = (d[d.hhid.isin(hh)].sort_values(KEY).reset_index(drop=True))
    rows = pd.read_csv(SAMPLE_DIR / "attrition_rows.csv")
    attrition = rows.merge(d, on=KEY, how="left")[EXTRACT_COLUMNS]
    return extract, attrition
