"""Optional: extract monthly CHIRPS rainfall and MODIS NDVI at each household location.

This is how data/derived/climate_monthly.parquet was produced (Google Earth Engine).
It is not part of `python reproduce.py`: it needs an Earth Engine account and takes
several hours, and the derived file is shipped. Run it to check or update the
climate inputs:

    pip install earthengine-api tqdm        # not in uv.lock
    earthengine authenticate
    python scripts/data/optional_gee_climate.py

Inputs   household locations from the GHS-Panel household geovariables (data/raw/ghs/):
         each household's wave-1 location and, for the 14 households the survey
         relocated, the later one (as in scripts/data/01_build_panel.py:load_climate);
         households of data/sample/extract_households.csv.
Output   outputs/data/gee_chirps_modis_monthly.xlsx with columns
         hhid, lat, lon, year, month, rain_mm, ndvi (one row per location and month,
         2000-01 to 2021-12). Extraction resumes where a previous run stopped.

Rainfall: CHIRPS daily (UCSB-CHG/CHIRPS/DAILY) summed by month. NDVI: MODIS MOD13A3
monthly (MODIS/061/MOD13A3), fill value -3000 masked, scaled by 1e-4. Both averaged
over a 2.5 km buffer around the location (reduceRegion, mean, 1 km scale).
CHIRPS and MODIS are revised from time to time, so a new extraction can differ
slightly from the shipped values.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nigeria_resilience.data import ghs_extract as gx  # noqa: E402

OUT_XLSX = ROOT / "outputs" / "data" / "gee_chirps_modis_monthly.xlsx"
RAIN_START_YM, RAIN_END_YM = "2000-01", "2021-12"
NDVI_START_YM, NDVI_END_YM = "2000-02", "2021-12"
BUFFER_M = 2500
KEY_DECIMALS = 6
SHEET_NAME = "data"


def household_locations() -> pd.DataFrame:
    """hhid, lat, lon: wave-1 location and any different later location, sample households."""
    sites = []
    for w in (1, 2, 3):
        n, _ = gx.read(gx.module(gx.GEO_FILE[w]))
        lat = next(x for x in n.columns if x.lower() == "lat_dd_mod")
        lon = next(x for x in n.columns if x.lower() == "lon_dd_mod")
        sites.append(pd.DataFrame({"hhid": n.hhid.astype(int), "wave": w,
                                   "lat": n[lat].astype(float), "lon": n[lon].astype(float)}))
    g = pd.concat(sites).dropna(subset=["lat", "lon"]).sort_values(["hhid", "wave"])
    g = g.drop_duplicates(["hhid", "lat", "lon"])
    sample = set(pd.read_csv(gx.SAMPLE_DIR / "extract_households.csv").hhid)
    return g[g.hhid.isin(sample)][["hhid", "lat", "lon"]].reset_index(drop=True)


def ym_range(start_ym: str, end_ym: str) -> List[Tuple[int, int]]:
    ys, ms = map(int, start_ym.split("-"))
    ye, me = map(int, end_ym.split("-"))
    out, y, m = [], ys, ms
    while (y < ye) or (y == ye and m <= me):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


MONTHS_RAIN = ym_range(RAIN_START_YM, RAIN_END_YM)
MONTHS_NDVI = set(ym_range(NDVI_START_YM, NDVI_END_YM))


def key_for(hhid, lat, lon) -> str:
    return f"{hhid}|{round(float(lat), KEY_DECIMALS)}|{round(float(lon), KEY_DECIMALS)}"


def completed_keys(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    df = pd.read_excel(path, sheet_name=SHEET_NAME).dropna(subset=["year", "month"])
    counts = df.groupby([df.hhid, df.lat.round(KEY_DECIMALS), df.lon.round(KEY_DECIMALS)]).size()
    return {key_for(h, la, lo) for (h, la, lo), n in counts.items() if n >= len(MONTHS_RAIN)}


def append_rows(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_excel(path, sheet_name=SHEET_NAME, index=False)
        return
    from openpyxl import load_workbook
    start = load_workbook(path)[SHEET_NAME].max_row
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as w:
        df.to_excel(w, sheet_name=SHEET_NAME, index=False, header=False, startrow=start)


def main(batch_size: int = 100) -> None:
    import ee
    from tqdm.auto import tqdm

    ee.Initialize()
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
    mod13a3 = ee.ImageCollection("MODIS/061/MOD13A3").select("NDVI")

    def masked(name):
        return ee.Image.constant(0).rename(name).updateMask(ee.Image.constant(0))

    def rain_month(start, end):
        coll = chirps.filterDate(start, end)
        return ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.sum().rename("rain_mm"), masked("rain_mm")))

    def ndvi_month(start, end):
        coll = mod13a3.filterDate(start, end).map(
            lambda img: img.select("NDVI").updateMask(img.select("NDVI").neq(-3000)))
        return ee.Image(ee.Algorithms.If(coll.size().gt(0),
                                         coll.mean().multiply(0.0001).rename("ndvi"), masked("ndvi")))

    def monthly_at_point(lat, lon):
        geom = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)
        feats = []
        for yy, mm in MONTHS_RAIN:
            start = ee.Date.fromYMD(yy, mm, 1)
            end = start.advance(1, "month")
            nd = ndvi_month(start, end) if (yy, mm) in MONTHS_NDVI else masked("ndvi")
            vals = ee.Dictionary(rain_month(start, end).addBands(nd).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom, scale=1000, bestEffort=True,
                maxPixels=1_000_000_000)).combine(ee.Dictionary({"year": yy, "month": mm}), True)
            feats.append(ee.Feature(None, vals))
        return ee.FeatureCollection(feats)

    points = household_locations()
    done = completed_keys(OUT_XLSX)
    todo = [r for r in points.itertuples(index=False) if key_for(r.hhid, r.lat, r.lon) not in done]
    buf: List[dict] = []
    for i, r in enumerate(tqdm(todo, desc="Household locations"), start=1):
        for f in monthly_at_point(r.lat, r.lon).getInfo().get("features", []):
            p = f.get("properties", {})
            buf.append({"hhid": r.hhid, "lat": r.lat, "lon": r.lon, "year": p.get("year"),
                        "month": p.get("month"), "rain_mm": p.get("rain_mm"), "ndvi": p.get("ndvi")})
        if i % batch_size == 0:
            append_rows(buf, OUT_XLSX)
            buf.clear()
    append_rows(buf, OUT_XLSX)
    print(f"Written {OUT_XLSX}")


if __name__ == "__main__":
    main()
