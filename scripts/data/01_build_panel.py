"""
Build the household panel used by every model (paper Section 2).

Pipeline stages:
  1. Load and combine resilience, PPP, and CPI data
  2. Convert consumption to 2016 PPP constant, compute poverty dummies
  3. Compute log consumption, leads, and dummies for panel analysis
  4. Attach EA GPS coordinates and survey timing variables
  5. Compute household-level climate anomalies (SPI, NDVI z-scores)
  6. Compute household-level conflict distance/timing indicators
  7. Merge all shocks with panel data, select final columns, save

Usage:
    python scripts/data/01_build_panel.py
    python scripts/data/01_build_panel.py --output outputs/data/analysis_panel.xlsx

Reads   outputs/data/ghs_extract.xlsx (GHS-Panel extract, from 00_build_extract.py)
        data/raw/deflators/{nigeria_ppp,CPI_Nigeria_BS_25022022}.csv
        data/derived/climate_monthly.parquet + GHS geovariables (CHIRPS / MODIS)
        data/derived/conflict_measures.csv (ACLED distances and timing)
        Authors' copies, used instead when present: data/raw/lsms/ (extract, EA
        coordinates), data/raw/climate/ (extraction with coordinates), data/raw/conflict/
        (ACLED events).
Writes  outputs/data/analysis_panel.xlsx (default): the analysis panel read by every model
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from climate_indices import compute, indices
from climate_indices.indices import Distribution

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


# ===================================================================
# 1. Data loading and consumption conversion helpers
# ===================================================================

def load_resilience_data(
    resilience_path: Path, ppp_path: Path, cpi_path: Path,
    ppp_sep: str = ";", cpi_sep: str = ";",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load resilience, PPP, and CPI data from specified file paths."""
    resilience_data = pd.read_excel(resilience_path)
    ppp_data = pd.read_csv(ppp_path, sep=ppp_sep)
    cpi_data = pd.read_csv(cpi_path, sep=cpi_sep)
    return resilience_data, ppp_data, cpi_data


def combine_data(
    resilience_data: pd.DataFrame, ppp_data: pd.DataFrame,
    cpi_data: pd.DataFrame, merge_column: str = "surveyprd",
) -> pd.DataFrame:
    """Merge resilience, PPP, and CPI data on *merge_column*."""
    combined = pd.merge(resilience_data, ppp_data, on=merge_column, how="inner")
    combined = pd.merge(combined, cpi_data, on=merge_column, how="inner")
    return combined


def convert_consumption_to_constant_naira(data: pd.DataFrame) -> pd.DataFrame:
    """Convert consumption to 2016 Naira constant and adjust for rural/urban CPI."""
    data["totcons_per_capita_per_day_2016cst"] = (
        data.totcons_per_capita_per_day * (1 + data.inflation_rate)
    )

    cpi_rural_2016 = 199.53
    cpi_urban_2016 = 201.47
    inflation_rate_urban_rural_diff = round(
        (cpi_rural_2016 - cpi_urban_2016) / cpi_urban_2016, 4
    )

    data["inflation_rate_urban_rural_2016"] = np.where(
        data.rural == 0, inflation_rate_urban_rural_diff, 0
    )
    data["totcons_per_capita_per_day_2016cst"] = (
        data.totcons_per_capita_per_day_2016cst
        * (1 + data.inflation_rate_urban_rural_2016)
    )
    data["totcons_per_capita_per_day_2016cst_naira"] = data[
        "totcons_per_capita_per_day_2016cst"
    ]
    data["ltotcons_per_capita_per_day_2016cst_naira"] = np.log(
        data["totcons_per_capita_per_day_2016cst_naira"]
    )
    return data


def convert_to_ppp_and_calculate_poverty_dummies(
    data: pd.DataFrame,
    ppp_conversion_factor: float = 101.91,
    national_poverty_line_2019_daily: float = 137430 / 365,
) -> pd.DataFrame:
    """Convert consumption to $PPP and compute poverty dummies."""
    data["totcons_per_capita_per_day_2016cst"] = (
        data["totcons_per_capita_per_day_2016cst"] / ppp_conversion_factor
    )

    inflation_rate_2019_2016 = (200.30 - 291.4) / 291.4
    national_poverty_line_2016 = national_poverty_line_2019_daily * (
        1 + inflation_rate_2019_2016
    )
    national_poverty_line_2016 = round(
        national_poverty_line_2016 / ppp_conversion_factor, 2
    )

    international_poverty_line_2016 = 1.9
    data["poverty_dummy_wb"] = np.where(
        data.totcons_per_capita_per_day_2016cst < international_poverty_line_2016, 1, 0
    )
    data["poverty_dummy_nat"] = np.where(
        data.totcons_per_capita_per_day_2016cst < national_poverty_line_2016, 1, 0
    )
    return data


def compute_log_consumption_and_prepare_for_analysis(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute log consumption, lead values for panel analysis, squared/cubed
    terms, and convert categorical variables into dummies.
    """
    data["ltotcons_per_capita_per_day_2016cst"] = np.log(
        data.totcons_per_capita_per_day_2016cst
    )

    data = data.sort_values(by=["hhid", "surveyprd"])
    data_panel = data.set_index(["hhid", "surveyprd"])

    # Lead values (shift -1 within each household)
    data_panel["lead_ltotcons_per_capita_per_day_2016cst"] = (
        data_panel.groupby(level="hhid")["ltotcons_per_capita_per_day_2016cst"].shift(-1)
    )
    data_panel["lead_poverty_dummy_wb"] = (
        data_panel.groupby(level="hhid")["poverty_dummy_wb"].shift(-1)
    )
    data_panel["lead_poverty_dummy_nat"] = (
        data_panel.groupby(level="hhid")["poverty_dummy_nat"].shift(-1)
    )

    data_panel = data_panel.reset_index()

    # Polynomial terms for mean equation
    data_panel["ltotcons_per_capita_per_day_2016cst_sqr"] = (
        data_panel.ltotcons_per_capita_per_day_2016cst ** 2
    )
    data_panel["ltotcons_per_capita_per_day_2016cst_cub"] = (
        data_panel.ltotcons_per_capita_per_day_2016cst ** 3
    )

    data_panel["survey_year"] = data_panel.surveyprd

    # Dummies for categorical variables
    data_panel = pd.get_dummies(
        data_panel,
        columns=[
            "survey_year",
            "marital_status_head_hh",
            "crop_livestock_hh",
            "religion_head_hh",
            "srtm_nga_5_15",
            "sq7",
            "main_cook_fuel",
            "main_source_water",
            "main_toilet_facility",
            "education_level_head_hh",
        ],
        drop_first=False,
        dtype="int",
    )
    return data_panel


# ===================================================================
# 2. Survey timing helpers
# ===================================================================

SURVEY_DATES = {
    2010.0: pd.Timestamp("2010-09-01"),
    2011.0: pd.Timestamp("2011-03-01"),
    2012.0: pd.Timestamp("2012-10-01"),
    2013.0: pd.Timestamp("2013-03-01"),
    2015.0: pd.Timestamp("2015-10-01"),
    2016.0: pd.Timestamp("2016-03-01"),
}

MAPPING_VISIT = {
    2010.0: "August-October",
    2011.0: "February-April",
    2012.0: "September-November",
    2013.0: "February-April",
    2015.0: "September-November",
    2016.0: "February-April",
}


def attach_survey_timing(final_data: pd.DataFrame) -> pd.DataFrame:
    """Attach periods_month, survey_date, months_since_prev, months_since_first."""
    final_data["periods_month"] = final_data["surveyprd"].map(MAPPING_VISIT)

    sp = pd.to_numeric(final_data["surveyprd"], errors="coerce").astype("float64")
    final_data["survey_date"] = sp.map(SURVEY_DATES)

    date_map = pd.Series(SURVEY_DATES)
    present_unique = pd.Index(sp.dropna().unique())
    defined_keys = pd.Index(date_map.index)
    waves_present = pd.Index(sorted(present_unique.intersection(defined_keys)))

    months_map = {}
    if len(waves_present) > 0:
        months_map[waves_present[0]] = pd.NA
        for i in range(1, len(waves_present)):
            curr = waves_present[i]
            prev = waves_present[i - 1]
            a = SURVEY_DATES.get(curr)
            b = SURVEY_DATES.get(prev)
            if a is None or b is None:
                months_map[curr] = pd.NA
            else:
                months_map[curr] = (a.year - b.year) * 12 + (a.month - b.month)

    final_data["months_since_prev"] = sp.map(months_map).astype("Int64")

    if len(waves_present) > 0:
        first_date = SURVEY_DATES[waves_present[0]]
        months_since_first_map = {
            w: (SURVEY_DATES[w].year - first_date.year) * 12
            + (SURVEY_DATES[w].month - first_date.month)
            for w in waves_present
        }
        final_data["months_since_first"] = sp.map(months_since_first_map).astype("Int64")

    return final_data


# ===================================================================
# 3. GPS merge
# ===================================================================

def attach_gps_coordinates(
    final_data: pd.DataFrame, gps_path: Path
) -> pd.DataFrame:
    """Merge EA GPS coordinates onto the panel data."""
    ea_gps = pd.read_stata(str(gps_path))
    ea_gps = ea_gps.drop_duplicates()
    ea_gps = ea_gps[["wave", "hh_id_merge", "lat_modified", "lon_modified"]].drop_duplicates()

    wave_map = {1.0: "wave1", 2.0: "wave2", 3.0: "wave3"}
    ea_gps["wave_text"] = ea_gps["wave"].map(wave_map)
    ea_gps = ea_gps.drop(columns=["wave"])
    ea_gps.rename(columns={"hh_id_merge": "hhid", "wave_text": "wave"}, inplace=True)
    ea_gps["hhid"] = ea_gps["hhid"].astype(int)

    merged = pd.merge(final_data, ea_gps, on=["hhid", "wave"], how="left")
    return merged


# ===================================================================
# 4. Climate anomaly helpers (SPI + NDVI z-scores)
# ===================================================================

MONTHMAP = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        1,
    )
}


def month_range_to_list(s: str) -> List[int]:
    """Parse 'August-October' or 'September-November' -> [8,9,10] or [9,10,11]."""
    a, b = re.split(r"\s*[--]\s*", s.strip())
    start, end = MONTHMAP[a.lower()], MONTHMAP[b.lower()]
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, 13)) + list(range(1, end + 1))


def prep_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Create a monthly datetime column and sort."""
    out = df.copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    return out.sort_values(["hhid", "date"]).reset_index(drop=True)


def compute_spi_ci(monthly_df: pd.DataFrame, scale: int) -> pd.DataFrame:
    """
    SPI-{scale} for each hhid using climate_indices.indices.spi
    (Gamma, zero-adjusted, monthly), calibration 2000-2021.
    """
    full_idx = pd.date_range("2000-01-01", "2021-12-01", freq="MS")
    wide = (
        monthly_df.pivot_table(
            index="date", columns="hhid", values="rain_mm", aggfunc="first"
        ).reindex(full_idx)
    )

    def _spi_series(col: pd.Series) -> pd.Series:
        arr = col.to_numpy(dtype=float)
        vals = indices.spi(
            arr,
            scale=scale,
            distribution=Distribution.gamma,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2021,
            periodicity=compute.Periodicity.monthly,
        )
        return pd.Series(vals, index=col.index)

    spi = wide.apply(_spi_series, axis=0)
    long = spi.stack().reset_index()
    long.columns = ["date", "hhid", f"SPI_{scale}"]
    return long


def compute_ndvi_z(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (hhid, calendar month) z-score using fixed 2000-2021 baseline.
    """
    d = monthly_df[["hhid", "date", "ndvi"]].copy()
    d["month"] = d["date"].dt.month
    base = d[(d["date"] >= "2000-01-01") & (d["date"] <= "2021-12-31")]

    stats = (
        base.groupby(["hhid", "month"], dropna=False)["ndvi"]
        .agg(mu="mean", sd=lambda s: s.std(ddof=1))
        .reset_index()
    )
    out = d.merge(stats, on=["hhid", "month"], how="left")
    out.loc[out["sd"] <= 1e-6, "sd"] = np.nan
    out["NDVI_z"] = (out["ndvi"] - out["mu"]) / out["sd"]
    return out[["hhid", "date", "NDVI_z"]]


def build_schedule() -> pd.DataFrame:
    """Return the survey wave/season schedule."""
    rows = [
        {"season": "post_planting", "wave": "wave1", "surveyprd": 2010, "periods_month": "August-October"},
        {"season": "post_harvest", "wave": "wave1", "surveyprd": 2011, "periods_month": "February-April"},
        {"season": "post_planting", "wave": "wave2", "surveyprd": 2012, "periods_month": "September-November"},
        {"season": "post_harvest", "wave": "wave2", "surveyprd": 2013, "periods_month": "February-April"},
        {"season": "post_planting", "wave": "wave3", "surveyprd": 2015, "periods_month": "September-November"},
        {"season": "post_harvest", "wave": "wave3", "surveyprd": 2016, "periods_month": "February-April"},
    ]
    return pd.DataFrame(rows)


def schedule_to_months(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Expand schedule into (year, month) rows."""
    rows = []
    for r in sched_df.itertuples(index=False):
        months = month_range_to_list(str(r.periods_month))
        for m in months:
            y = int(r.surveyprd)
            if months[0] > months[-1] and m < months[0]:
                y += 1
            rows.append({
                "wave": r.wave, "season": r.season,
                "surveyprd": int(r.surveyprd), "year": y, "month": m,
            })
    return pd.DataFrame(rows)


def seasonal_aggregate(
    monthly: pd.DataFrame, sched_months: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate monthly SPI/NDVI to seasonal level.
    Output: one row per (hhid, wave, season, surveyprd).
    """
    base = monthly[["hhid", "year", "month", "SPI_1", "SPI_3", "NDVI_z", "lat", "lon"]]

    joined = sched_months.merge(base, on=["year", "month"], how="left")

    agg = (
        joined.groupby(["hhid", "wave", "season", "surveyprd"], as_index=False)
        .agg(NDVI_z_mean=("NDVI_z", "mean"), SPI_1_mean=("SPI_1", "mean"))
    )

    ends = (
        sched_months.sort_values(["wave", "season", "surveyprd", "year", "month"])
        .groupby(["wave", "season", "surveyprd"], as_index=False)
        .tail(1)
        .rename(columns={"year": "end_year", "month": "end_month"})
    )

    end_rows = (
        ends.merge(
            base, left_on=["end_year", "end_month"], right_on=["year", "month"], how="left"
        )
        .rename(columns={"SPI_3": "SPI_3_end"})
        [["hhid", "wave", "season", "surveyprd", "SPI_3_end", "lat", "lon"]]
    )

    out = agg.merge(end_rows, on=["hhid", "wave", "season", "surveyprd"], how="left")
    return out.sort_values(["hhid", "surveyprd", "season"]).reset_index(drop=True)


def build_wave_season_anomalies(rain_ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate full SPI + NDVI anomaly computation."""
    df = prep_monthly(rain_ndvi_df)

    print("  Computing SPI-1...")
    spi1 = compute_spi_ci(df, scale=1)
    print("  Computing SPI-3...")
    spi3 = compute_spi_ci(df, scale=3)
    print("  Computing NDVI z-scores...")
    ndviz = compute_ndvi_z(df)

    monthly = (
        df[["hhid", "lat", "lon", "date"]]
        .merge(spi1, on=["hhid", "date"], how="left")
        .merge(spi3, on=["hhid", "date"], how="left")
        .merge(ndviz, on=["hhid", "date"], how="left")
    )
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month

    sched = build_schedule()
    sched_months = schedule_to_months(sched)
    return seasonal_aggregate(monthly, sched_months)


# ===================================================================
# 5. Conflict distance/timing helpers
# ===================================================================

EARTH_RADIUS_KM = 6371.0088


def households_x_events_distances(
    households_df: pd.DataFrame, events_df: pd.DataFrame, chunk_events: int = 2000
) -> pd.DataFrame:
    """
    Compute Haversine distances between every household and every conflict event.
    Returns one row per (household x event).
    """
    hh = households_df.dropna(subset=["lat_modified", "lon_modified"]).copy()
    ev = events_df.dropna(subset=["latitude", "longitude", "event_date"]).copy()

    hh_lat = np.radians(hh["lat_modified"].to_numpy(float))
    hh_lon = np.radians(hh["lon_modified"].to_numpy(float))
    H = len(hh)

    parts = []
    for start in range(0, len(ev), chunk_events):
        sub = ev.iloc[start : start + chunk_events].reset_index(drop=True)
        ev_lat = np.radians(sub["latitude"].to_numpy(float))
        ev_lon = np.radians(sub["longitude"].to_numpy(float))

        dlat = ev_lat[None, :] - hh_lat[:, None]
        dlon = ev_lon[None, :] - hh_lon[:, None]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(hh_lat)[:, None] * np.cos(ev_lat)[None, :] * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist_km = EARTH_RADIUS_KM * c

        E = len(sub)
        rep_ev = (
            sub[["event_id_no_cnty", "event_type", "event_date", "latitude", "longitude"]]
            .iloc[np.repeat(np.arange(E), H)]
            .reset_index(drop=True)
        )
        rep_hh = pd.concat(
            [hh[["hhid", "lat_modified", "lon_modified"]]] * E, ignore_index=True
        )
        parts.append(
            pd.concat(
                [rep_hh, rep_ev, pd.DataFrame({"distance_km": dist_km.ravel(order="F")})],
                axis=1,
            )
        )

    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "hhid", "lat_modified", "lon_modified", "event_id_no_cnty",
            "event_type", "event_date", "latitude", "longitude", "distance_km",
        ]
    )


def _parse_month_range(s: str) -> Tuple[int, int]:
    a, b = re.split(r"\s*[--]\s*", s.strip())
    return MONTHMAP[a.lower()], MONTHMAP[b.lower()]


def build_windows(sched_df: pd.DataFrame) -> pd.DataFrame:
    """Build survey windows with pre-6-month and pre-3-month dates."""
    rows = []
    for r in sched_df.itertuples(index=False):
        m0, m1 = _parse_month_range(str(r.periods_month))
        y0 = int(r.surveyprd)
        y1 = y0 if m0 <= m1 else y0 + 1
        start = pd.Timestamp(year=y0, month=m0, day=1)
        end = (
            pd.Timestamp(year=y1, month=m1, day=1)
            + pd.DateOffset(months=1)
            - pd.Timedelta(days=1)
        )
        rows.append({
            "wave": r.wave, "season": r.season, "surveyprd": y0,
            "start_date": start.normalize(), "end_date": end.normalize(),
        })
    win = pd.DataFrame(rows)
    win["pre6_start"] = (win["start_date"] - pd.DateOffset(months=6)).dt.normalize()
    win["pre3_start"] = (win["start_date"] - pd.DateOffset(months=3)).dt.normalize()
    win["combo_start"] = win["pre6_start"]
    win["combo_end"] = win["end_date"]
    return win


def min_distances_with_timing(
    pairs_df: pd.DataFrame,
    sched_df: pd.DataFrame,
    event_types: tuple = ("Battles", "Protests", "Riots", "Violence against civilians"),
) -> pd.DataFrame:
    """
    For each (hhid, wave/season, event_type), find the closest event within
    the combined 6-month-pre-to-end window and assign a timing bin.
    """
    df = pairs_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["event_date", "distance_km"])
    df = df[df["event_type"].isin(event_types)]

    windows = build_windows(sched_df)
    out = []

    for w in windows.itertuples(index=False):
        sub = df[(df["event_date"] >= w.combo_start) & (df["event_date"] <= w.combo_end)].copy()
        if sub.empty:
            continue
        best = (
            sub.sort_values(["hhid", "event_type", "distance_km", "event_date"])
            .groupby(["hhid", "event_type"], as_index=False)
            .first()
        )

        within = (best["event_date"] >= w.start_date) & (best["event_date"] <= w.end_date)
        pre13 = (best["event_date"] >= w.pre3_start) & (best["event_date"] < w.start_date)
        pre46 = (best["event_date"] >= w.pre6_start) & (best["event_date"] < w.pre3_start)
        best["timing_bin"] = np.select(
            [within, pre13, pre46], ["within", "pre_1_3", "pre_4_6"], default=np.nan
        )

        best["event_year"] = best["event_date"].dt.year
        best["event_month"] = best["event_date"].dt.month
        best["wave"] = w.wave
        best["season"] = w.season
        best["surveyprd"] = w.surveyprd

        out.append(
            best[
                [
                    "hhid", "wave", "season", "surveyprd", "event_type",
                    "event_id_no_cnty", "event_date", "event_year", "event_month",
                    "latitude", "longitude", "distance_km", "timing_bin",
                ]
            ]
        )

    if not out:
        return pd.DataFrame(
            columns=[
                "hhid", "wave", "season", "surveyprd", "event_type",
                "event_id_no_cnty", "event_date", "event_year", "event_month",
                "latitude", "longitude", "distance_km", "timing_bin",
            ]
        )
    return pd.concat(out, ignore_index=True)


def make_dist_time_wide(
    result_long: pd.DataFrame, tau_km: float | None = 50.0
) -> pd.DataFrame:
    """Pivot conflict results to wide format with *_dist and *_time columns."""
    name_map = {
        "Battles": "battle",
        "Protests": "protest",
        "Riots": "riot",
        "Violence against civilians": "vac",
    }
    keys = ["hhid", "wave", "season", "surveyprd"]

    dist_wide = (
        result_long.pivot_table(
            index=keys, columns="event_type", values="distance_km", aggfunc="min"
        )
        .rename(columns=name_map)
        .add_suffix("_dist")
        .reset_index()
    )

    time_wide = (
        result_long.pivot_table(
            index=keys, columns="event_type", values="timing_bin", aggfunc="first"
        )
        .rename(columns=name_map)
        .add_suffix("_time")
        .reset_index()
    )

    out = dist_wide.merge(time_wide, on=keys, how="outer")

    if tau_km is not None:
        for col in [c for c in out.columns if c.endswith("_dist")]:
            base_name = col[:-5]
            out[f"{base_name}_decay"] = np.exp(-out[col] / tau_km)

    return out


# ===================================================================
# 6. Final column selection lists
# ===================================================================

LIST_EXP = [
    "ltotcons_per_capita_per_day_2016cst",
    "ltotcons_per_capita_per_day_2016cst_sqr",
    "ltotcons_per_capita_per_day_2016cst_cub",
]

LIST_HH_CHARACTERISTICS = [
    "fhh", "rural",
    "marital_status_head_hh_married_monogamous",
    "marital_status_head_hh_married_polygamous",
    "marital_status_head_hh_widowed",
    "age_head_hh", "agactivities_hh",
    "prop_female_age_sup_66",
    "prop_female_age_16_65",
    "prop_male_age_16_65",
    "prop_female_age_5_15",
    "prop_male_age_5_15",
    "prop_all_age_0_4",
    "education_level_head_hh_higher", "education_level_head_hh_others",
    "education_level_head_hh_primary", "education_level_head_hh_secondary",
    "own_car", "own_computer", "own_motorcycle", "own_tv",
    "own_house", "number_rooms", "connection_electricity",
    "saving", "loan",
    "main_cook_fuel_firewood", "main_cook_fuel_kerosene",
    "main_source_water_borehole", "main_source_water_pipe borne",
    "main_source_water_river", "main_source_water_well",
    "main_toilet_facility_flush to septic tank",
    "main_toilet_facility_none",
    "main_toilet_facility_pit latrine",
    "main_toilet_facility_v.i.p latrine",
    "farm_area",
]

LIST_GEO_VARIABLES = [
    "dist_road", "dist_market",
    "srtm_nga_5_15_high_altitude_plains",
    "srtm_nga_5_15_hills",
    "srtm_nga_5_15_low_plateaus",
    "srtm_nga_5_15_lowlands",
    "srtm_nga_5_15_mid_altitude_plains",
    "srtm_nga_5_15_plains",
    "srtm_nga_5_15_platform_very_low_plateaus",
    "af_bio_1", "af_bio_12",
]

LIST_SHOCKS_HH = [
    "death_adultworking", "illness_incomearningsmbers",
    "increase_price_input", "increase_price_food",
]

LIST_SHOCKS_COMM = [
    "event_drought", "event_flood", "event_crop_disease", "event_livestock_disease",
]

LIST_EXOG_SHOCKS = [
    "SPI_3_end", "NDVI_z_mean", "battle_dist",
    "protest_dist", "riot_dist", "vac_dist",
    "battle_time_pre_1_3", "battle_time_pre_4_6",
    "protest_time_pre_1_3", "protest_time_pre_4_6",
    "riot_time_pre_1_3", "riot_time_pre_4_6",
    "vac_time_pre_1_3", "vac_time_pre_4_6",
]

TARGET = ["lead_ltotcons_per_capita_per_day_2016cst"]

ID_COLS = [
    "hhid", "surveyprd", "ag_hh", "zone_norm", "state_norm", "lga_norm",
    "ea", "hhweight", "year", "lat", "lon", "wave",
    "months_since_prev", "season_post_harvest",
]

LIST_ALL = (
    LIST_EXP + LIST_HH_CHARACTERISTICS + LIST_GEO_VARIABLES
    + LIST_SHOCKS_HH + LIST_SHOCKS_COMM + LIST_EXOG_SHOCKS
)

LIST_ALL_AND_TARGET = ID_COLS + TARGET + LIST_ALL


# ===================================================================
# Main pipeline
# ===================================================================

def main(
    resilience_path: Path,
    ppp_path: Path,
    cpi_path: Path,
    gps_path: Path,
    rain_ndvi_path: Path,
    conflict_path: Path,
    output_path: Path,
) -> None:
    """Orchestrate the full data-preparation pipeline."""

    # ------------------------------------------------------------------
    # Stage 1: Load, merge, convert consumption
    # ------------------------------------------------------------------
    print("Stage 1: Loading and combining resilience, PPP, and CPI data...")
    resilience_data, ppp_data, cpi_data = load_resilience_data(
        resilience_path, ppp_path, cpi_path
    )
    combined = combine_data(resilience_data, ppp_data, cpi_data)

    print("  Converting consumption to 2016 constant Naira...")
    processed = convert_consumption_to_constant_naira(combined)

    print("  Converting to PPP and computing poverty dummies...")
    data = convert_to_ppp_and_calculate_poverty_dummies(processed)

    print("  Computing log consumption, leads, and dummies...")
    final_data = compute_log_consumption_and_prepare_for_analysis(data)

    # Back-fill age and number_rooms within household
    final_data["age_head_hh"] = final_data.groupby(["hhid", "fhh"])["age_head_hh"].bfill()
    final_data["number_rooms"] = final_data.groupby(["hhid"])["number_rooms"].bfill()

    # ------------------------------------------------------------------
    # Stage 2: Attach survey timing variables
    # ------------------------------------------------------------------
    print("Stage 2: Attaching survey timing variables...")
    final_data = attach_survey_timing(final_data)

    # ------------------------------------------------------------------
    # Stage 3: Attach GPS coordinates
    # ------------------------------------------------------------------
    # The conflict measures need the EA coordinates and the raw ACLED events, which
    # cannot be redistributed; without them the shipped derived measures are used.
    raw_conflict = Path(gps_path).exists() and Path(conflict_path).exists()
    if raw_conflict:
        print("Stage 3: Attaching EA GPS coordinates...")
        final_data_gps = attach_gps_coordinates(final_data, gps_path)
        print(f"  Panel data with GPS: {final_data_gps.shape}")
        # Unique household-location pairs for shock computation
        hh_locations = final_data_gps[["hhid", "lat_modified", "lon_modified"]].drop_duplicates()

    # ------------------------------------------------------------------
    # Stage 4: Compute climate anomalies (SPI + NDVI z)
    # ------------------------------------------------------------------
    print("Stage 4: Computing climate anomalies (SPI, NDVI z-scores)...")
    print("  Loading pre-computed monthly rainfall/NDVI data...")
    rain_ndvi = load_climate(rain_ndvi_path)
    anomalies = build_wave_season_anomalies(rain_ndvi)
    print(f"  Climate anomalies: {anomalies.shape}")

    # ------------------------------------------------------------------
    # Stage 5: Compute conflict distance/timing indicators
    # ------------------------------------------------------------------
    if raw_conflict:
        print("Stage 5: Computing conflict distance and timing indicators...")
        print("  Loading ACLED conflict data...")
        conflict_raw = pd.read_csv(conflict_path)

        col_keep = [
            "event_id_cnty", "event_id_no_cnty", "event_date", "event_type",
            "sub_event_type", "region", "country", "admin1", "admin2", "admin3",
            "location", "latitude", "longitude", "source", "notes", "fatalities", "iso3",
        ]
        events = conflict_raw[col_keep].copy()
        events["event_date"] = pd.to_datetime(events["event_date"], dayfirst=True, errors="coerce")
        events = events.dropna(subset=["event_date", "latitude", "longitude"])
        events = events[events["event_date"].dt.year.between(2009, 2017)]

        print(f"  Computing pairwise distances ({len(hh_locations)} households x {len(events)} events)...")
        pairs = households_x_events_distances(hh_locations, events)

        sched_df = build_schedule()
        print("  Finding closest events per household/wave/season...")
        result_long = min_distances_with_timing(pairs, sched_df)
        conflict_wide = make_dist_time_wide(result_long, tau_km=None)
        conflict_wide[["battle_dist", "protest_dist", "riot_dist", "vac_dist"]] = (
            conflict_wide[["battle_dist", "protest_dist", "riot_dist", "vac_dist"]].round(2)
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        conflict_wide[CONFLICT_COLUMNS].to_csv(Path(output_path).parent / "conflict_measures.csv", index=False)
    else:
        print("Stage 5: ACLED events or EA coordinates not present: using the derived conflict "
              f"measures {CONFLICT_DERIVED.relative_to(PROJECT_ROOT)}")
        conflict_wide = pd.read_csv(CONFLICT_DERIVED)

    # ------------------------------------------------------------------
    # Stage 6: Merge all shocks
    # ------------------------------------------------------------------
    print("Stage 6: Merging climate and conflict shocks...")
    hh_shocks = pd.merge(
        anomalies, conflict_wide,
        on=["hhid", "wave", "season", "surveyprd"], how="left",
    )
    hh_shocks.drop_duplicates(subset=["hhid", "wave", "season", "surveyprd"], inplace=True)

    # ------------------------------------------------------------------
    # Stage 7: Merge shocks with panel data
    # ------------------------------------------------------------------
    print("Stage 7: Merging shocks with panel data...")
    final_merged = pd.merge(
        final_data, hh_shocks,
        on=["hhid", "wave", "season", "surveyprd"], how="left",
    )

    # Create dummies for season and conflict timing
    print("  Creating dummies for season and conflict timing...")
    final_merged = pd.get_dummies(
        final_merged,
        columns=["season", "battle_time", "protest_time", "riot_time", "vac_time"],
        dtype="int",
        drop_first=False,
    )

    # Select final columns
    print("  Selecting final columns...")
    final_vf = final_merged[LIST_ALL_AND_TARGET].copy()

    # Back-fill SPI_3_end within household
    final_vf["SPI_3_end"] = final_vf.groupby(["hhid"])["SPI_3_end"].bfill()

    # ------------------------------------------------------------------
    # Stage 8: Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving output ({final_vf.shape}) to {output_path}...")
    final_vf.to_excel(str(output_path), index=False)
    print("Done.")



# ===================================================================
# Derived (shareable) inputs
# ===================================================================

CONFLICT_DERIVED = PROJECT_ROOT / "data" / "derived" / "conflict_measures.csv"
CLIMATE_DERIVED = PROJECT_ROOT / "data" / "derived" / "climate_monthly.parquet"
CONFLICT_COLUMNS = ["hhid", "wave", "season", "surveyprd",
                    "battle_dist", "protest_dist", "riot_dist", "vac_dist",
                    "battle_time", "protest_time", "riot_time", "vac_time"]


def load_climate(path: Path) -> pd.DataFrame:
    """Monthly CHIRPS rainfall and MODIS NDVI at each household location.

    The authors' extraction (path) carries the EA coordinates. The shipped file
    data/derived/climate_monthly.parquet has the same rows in the same order without
    coordinates: site 0 is the household's wave-1 location and site 1, for the 14
    households the survey relocated, its later location. The coordinates are then
    taken from the GHS-Panel household geovariables (data/raw/ghs/).
    """
    if Path(path).exists():
        return pd.read_excel(str(path))
    c = pd.read_parquet(CLIMATE_DERIVED)
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from nigeria_resilience.data import ghs_extract as gx
    sites = []
    for w in (1, 2, 3):
        n, _ = gx.read(gx.module(gx.GEO_FILE[w]))
        lat = next(x for x in n.columns if x.lower() == "lat_dd_mod")
        lon = next(x for x in n.columns if x.lower() == "lon_dd_mod")
        sites.append(pd.DataFrame({"hhid": n.hhid.astype(int), "wave": w,
                                   "lat": n[lat].astype(float), "lon": n[lon].astype(float)}))
    g = pd.concat(sites).dropna(subset=["lat", "lon"]).sort_values(["hhid", "wave"])
    g = g.drop_duplicates(["hhid", "lat", "lon"])
    g["site"] = g.groupby("hhid").cumcount()
    out = c.merge(g[["hhid", "site", "lat", "lon"]], on=["hhid", "site"], how="left", sort=False)
    return out[["hhid", "lat", "lon", "year", "month", "rain_mm", "ndvi"]]

# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the final analysis-ready panel dataset for Nigeria resilience study."
    )
    parser.add_argument(
        "--resilience",
        type=Path,
        default=None,
        help="Path to the GHS-Panel extract (default: the authors' copy in data/raw/lsms/ "
             "if present, else outputs/data/ghs_extract.xlsx from 00_build_extract.py)",
    )
    parser.add_argument(
        "--ppp",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "deflators" / "nigeria_ppp.csv",
        help="Path to PPP CSV file",
    )
    parser.add_argument(
        "--cpi",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "deflators" / "CPI_Nigeria_BS_25022022.csv",
        help="Path to CPI CSV file",
    )
    parser.add_argument(
        "--gps",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "lsms" / "Nigeria_ea_gps.dta",
        help="Path to EA GPS Stata file",
    )
    parser.add_argument(
        "--rain-ndvi",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "climate" / "gee_chirps_modis_monthly_with_progress.xlsx",
        help="Path to pre-computed monthly rainfall/NDVI Excel file",
    )
    parser.add_argument(
        "--conflict",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "conflict" / "1997-01-01-2021-12-31-Nigeria-conflict.csv",
        help="Path to ACLED conflict CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "data" / "analysis_panel.xlsx",
        help="Path to output Excel file",
    )

    args = parser.parse_args()
    if args.resilience is None:
        authors_extract = PROJECT_ROOT / "data" / "raw" / "lsms" / "dev_resilience_cst_outiliers_prepared.xlsx"
        args.resilience = authors_extract if authors_extract.exists() else (
            PROJECT_ROOT / "outputs" / "data" / "ghs_extract.xlsx")
        print(f"GHS-Panel extract: {args.resilience}")

    main(
        resilience_path=args.resilience,
        ppp_path=args.ppp,
        cpi_path=args.cpi,
        gps_path=args.gps,
        rain_ndvi_path=args.rain_ndvi,
        conflict_path=args.conflict,
        output_path=args.output,
    )
