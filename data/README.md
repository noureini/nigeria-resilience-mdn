# Data

## Data availability statement

The paper uses five sources. Two of them (the GHS-Panel survey and the ACLED event
data) are free to obtain but their terms do not allow redistribution, so the package
ships the code that builds every analysis file from the original downloads, plus the
data that may be shared: open data, derived measures and identifiers.

| Source | Used for | In the package | How to obtain the rest |
|---|---|---|---|
| World Bank, *Nigeria General Household Survey-Panel* (GHS-Panel), waves 1-3 | households, consumption, covariates | no (terms of use) | free registration, see 1 below |
| EPAR (Evans School, University of Washington), agricultural indicators for the GHS-Panel | farm, livestock and income variables | code (`third_party/epar/`, BSD-3) | run on the GHS-Panel files, see 2 |
| ACLED, conflict events, Nigeria | distance to and timing of conflict events | derived measures (`derived/conflict_measures.csv`) | optional, see 3 |
| CHIRPS rainfall, MODIS NDVI (Google Earth Engine) | rainfall (SPI) and vegetation shocks | monthly values by household (`derived/climate_monthly.parquet`) | optional, see 4 |
| National Bureau of Statistics (CPI); World Bank WDI (PPP) | deflators | yes (`raw/deflators/`) | — |

### 1. GHS-Panel (World Bank Microdata Library)

Nigeria National Bureau of Statistics and World Bank, *General Household Survey,
Panel*, waves 1 (2010/11), 2 (2012/13) and 3 (2015/16), public use files:

* Wave 1: <https://microdata.worldbank.org/index.php/catalog/1002>, data release
  NGA_2010_GHSP-W1_v03_M, DOI [10.48529/y9e2-b753](https://doi.org/10.48529/y9e2-b753)
* Wave 2: <https://microdata.worldbank.org/index.php/catalog/1952>, data release
  NGA_2012_GHSP-W2_v02_M, DOI [10.48529/kxpy-aa72](https://doi.org/10.48529/kxpy-aa72)
* Wave 3: <https://microdata.worldbank.org/index.php/catalog/2734>, data release
  NGA_2015_GHSP-W3_v02_M, DOI [10.48529/7xmj-q133](https://doi.org/10.48529/7xmj-q133)

These are the data files served by the Microdata Library in November 2026. The
catalog's later version numbers (wave 1 version 04, wave 2 version 03) added
documentation only (food item photos; the survey report), not data.

Access: free registration and acceptance of the Microdata Library's terms of use;
the files are available immediately. Cost: none. The terms do not allow
redistribution, so the files are not in the package.

Download the **Stata** version of each wave and unzip it, keeping its folders, into

```
data/raw/ghs/wave1/   data/raw/ghs/wave2/   data/raw/ghs/wave3/
```

`scripts/data/00_build_extract.py` finds the files it needs by their World Bank file
names anywhere under `data/raw/ghs/` and checks them against `checksums.csv`
(a warning names any file that differs, e.g. from a later release).

### 2. EPAR household indicators

Farm area, crop, livestock and non-farm income, livestock holdings and the value of
assets come from the indicators of the Evans School Policy Analysis and Research
Group (EPAR), *Agricultural Development Indicator Curation*, Technical Report #335.
Their Stata code, version of 24 July 2019, is in `third_party/epar/` (BSD 3-Clause
license, © University of Washington; unchanged except for the folder paths).

With the GHS-Panel files in place, set the package folder at the top of
`third_party/epar/run_epar.do` and run it in Stata 14 or later. It writes
`data/raw/epar/Nigeria_GHS_W{1,2,3}_household_variables.dta`.
The build compares the variables it uses with `checksums.csv` (column
`content_sha256`).

### 3. ACLED

Armed Conflict Location & Event Data Project (ACLED), Nigeria, events from 1997-01-01
to 2021-12-31 (the analysis uses 2009-2017), exported after free registration at
<https://acleddata.com>.
ACLED's terms allow sharing the manipulated data used in a publication but not the
event file itself. The package ships the measures computed from it:
`derived/conflict_measures.csv`, one row per household and survey round, with the
distance (km) to the nearest battle, protest, riot and violence against civilians in
the survey window and its timing. They were computed by
`scripts/data/01_build_panel.py` from the events and the EA coordinates; with your
own export in `data/raw/conflict/` and the coordinates, the script recomputes them
(ACLED revises past events, so a new export can differ slightly).

### 4. CHIRPS and MODIS

Monthly rainfall (CHIRPS daily, Funk et al. 2015) and vegetation (MODIS MOD13A3 NDVI,
collection 6.1, Didan 2021), 2000-2021, averaged within 2.5 km of each household location in Google
Earth Engine. Both products are public. `derived/climate_monthly.parquet` holds the
monthly values by household, without coordinates; the build attaches the GHS-Panel
coordinates. `scripts/data/optional_gee_climate.py` is the extraction code (needs an
Earth Engine account; not run by `reproduce.py`).

### 5. Deflators

* `raw/deflators/CPI_Nigeria_BS_25022022.csv`: consumer price index (December 2009 =
  100) by survey round, National Bureau of Statistics (2021).
* `raw/deflators/nigeria_ppp.csv`: 2016 PPP conversion factor, private consumption,
  World Bank WDI (retrieved 5 January 2022), CC BY 4.0.

The national poverty line (2.54 $/day, 2016 PPP) is computed in
`nigeria_resilience.models.xgboost_mixture.compute_national_poverty_line` from the 2019
national line (137,430 Naira per person per year; National Bureau of Statistics 2020).

### Preservation and later releases of the survey

If the World Bank revises the data files, a new download can differ from the files
used here; `scripts/data/00_build_extract.py` then names each file whose SHA-256
differs from `checksums.csv`. The authors keep the exact files used (World Bank and
EPAR, with these checksums) in private storage for at least ten years after
publication, and will help a replicator compare a different release with them.
Contact: Edouard Romeo Mensah (corresponding author).

## The estimation sample

The 3,441 households of the analysis are selected from the GHS-Panel files by
`select_sample()` in `src/nigeria_resilience/data/ghs_extract.py`, which
`scripts/data/00_build_extract.py` runs and reports step by step:

| Step | Rule | Household-rounds | Households |
|---|---|---|---|
| | every household-round with a consumption aggregate | 28,279 | 4,997 |
| 1 | missing data: drop a household-round whose household is absent from that wave's dwelling, assets, savings/loans or education module, or whose geovariables or EPAR female-head indicator are missing | 28,075 | 4,969 |
| 2 | the same household: drop households whose person no. 1 on the roster has left (wave 2 post-planting, wave 3 post-harvest) or is missing from a roster, or that change EA, LGA or urban/rural sector (the GHS sector, `rururb`) | 22,897 | 4,001 |
| 3 | outliers: drop household-rounds whose log per-capita consumption is more than 3 standard deviations from the mean of their survey round and sector (mean and s.d. over the households observed in all six rounds after steps 1-2) | 22,733 | 4,001 |
| 4 | head's age: drop the wave-1 rounds of households 80022 and 180071, whose head's age in wave 1 is inconsistent with waves 2 and 3 | 22,729 | 4,001 |
| 5 | balanced panel: keep the households with all six rounds | 20,646 | 3,441 |

`sample/extract_households.csv` lists the 3,441 households, and
`sample/attrition_rows.csv` the household-rounds of the attrition comparison
(Tables A.4-A.5). Both hold identifiers only.

## What the data stage builds

```
data/raw/ghs/ + data/raw/epar/ ──00_build_extract.py──► outputs/data/ghs_extract.xlsx
                                                         outputs/data/attrition_sample.xlsx
extract + deflators + derived/ ──01_build_panel.py────► outputs/data/analysis_panel.xlsx
```

`analysis_panel.xlsx` (20,646 household-rounds, 88 columns) is the analysis panel read
by the models and the exhibits. Built in the reference environment it is identical to
the panel the shipped models were fitted on; elsewhere it agrees to about 1e-11
(last-digit differences in `log` and in the SPI's gamma fit). The verification step
checks it through the predictions of the shipped models.

## Files in this folder

```
data/
├── checksums.csv                   SHA-256 of every input (World Bank, EPAR, shipped files)
├── sample/                         household identifiers of the estimation and attrition samples
├── derived/                        conflict measures (ACLED) and climate values (CHIRPS, MODIS)
├── raw/deflators/                  CPI and PPP
├── raw/ghs/      (you add)         GHS-Panel waves 1-3, Stata files
└── raw/epar/     (you build)       EPAR household indicators
```

## Citations for the data

- Nigeria National Bureau of Statistics and World Bank. *Nigeria General Household Survey-Panel*, waves 1-3 (2010-2016). World Bank Microdata Library, NGA_2010_GHSP-W1_v03_M, NGA_2012_GHSP-W2_v02_M, NGA_2015_GHSP-W3_v02_M.
- Evans School Policy Analysis and Research Group (2019). *Agricultural Development Indicator Curation* (EPAR Technical Report #335), Stata code, version of 24 July 2019. University of Washington.
- Raleigh, C., Linke, A., Hegre, H., Karlsen, J. (2010). Introducing ACLED. *Journal of Peace Research* 47, 651-660. Data: ACLED, <https://acleddata.com>.
- Funk, C. et al. (2015). The climate hazards infrared precipitation with stations. *Scientific Data* 2, 150066.
- Didan, K. (2021). MODIS/Terra Vegetation Indices Monthly L3 Global 1km SIN Grid V061 (MOD13A3). NASA EOSDIS LP DAAC.
- National Bureau of Statistics (2020). *2019 Poverty and Inequality in Nigeria.* Abuja.
- National Bureau of Statistics (2021). *Consumer Price Index.* Abuja.
- World Bank. *World Development Indicators: PPP conversion factor, private consumption.*
- McKee, T. B., Doesken, N. J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *8th Conference on Applied Climatology*, 179-184.
