"""Load the Nigeria GHS-Panel analysis file used by every model and exhibit."""
from pathlib import Path

import pandas as pd

from nigeria_resilience.config import PANEL_FILE


def load_nigeria_data(path: Path = PANEL_FILE) -> pd.DataFrame:
    """Read the household panel and add the lead timing variables.

    ``lead_months_since_prev`` and ``lead_season_post_harvest`` describe the
    *next* survey round (t+1) and are shifted within household.
    """
    data = pd.read_excel(path)
    data = data.sort_values(by=["hhid", "surveyprd"]).set_index(["hhid", "surveyprd"])

    cols_to_lead = ["months_since_prev", "season_post_harvest"]
    lead_names = {c: f"lead_{c}" for c in cols_to_lead}
    data[list(lead_names.values())] = data.groupby(level="hhid")[cols_to_lead].shift(-1)

    return data.reset_index()
