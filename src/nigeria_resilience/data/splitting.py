from typing import Tuple

import numpy as np
import pandas as pd


def _split_counts_temp(n: int, fracs=(0.6, 0.1, 0.3)) -> Tuple[int, int, int]:
    """Return integer counts close to fracs that sum to n; keep splits non-empty when n>=3."""
    f_tr, f_va, f_te = fracs
    n_tr = int(np.floor(n * f_tr))
    n_va = int(np.floor(n * f_va))
    n_te = n - n_tr - n_va
    n_tr = max(0, n_tr); n_va = max(0, n_va); n_te = max(0, n_te)
    if n >= 3:
        counts = [n_tr, n_va, n_te]
        for i in range(3):
            if counts[i] == 0:
                j = int(np.argmax(counts))
                if counts[j] > 1:
                    counts[i] += 1; counts[j] -= 1
        n_tr, n_va, n_te = counts
    diff = n - (n_tr + n_va + n_te)
    order = np.argsort([-f_tr, -f_va, -f_te])
    for idx in order:
        if diff == 0: break
        if diff > 0:
            if idx == 0: n_tr += 1
            elif idx == 1: n_va += 1
            else: n_te += 1
            diff -= 1
        else:
            if idx == 0 and n_tr > 0: n_tr -= 1; diff += 1
            elif idx == 1 and n_va > 0: n_va -= 1; diff += 1
            elif idx == 2 and n_te > 0: n_te -= 1; diff += 1
    return n_tr, n_va, n_te


def splitting_data(
    data        : pd.DataFrame,
    split_type  : str = "temporal",
    time_col    : str = "surveyprd",
    lga_col     : str = "lga_norm",
    train_start : int = 2010,
    train_end   : int = 2012,
    val_year    : int = 2013,
    test_year   : int = 2015,
    hold_frac   : float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split with optional LGA partition.

    If split_type == "lga_temporal":
      - Partition LGAs into disjoint groups: 60% TRAIN, 10% VAL, 30% TEST.
      - TRAIN:       [train_start, train_end] x TRAIN_LGAs
      - VAL (HPO):   val_year x VAL_LGAs (unseen during training)
      - FULL_TRAIN:  [train_start, val_year] x (TRAIN_LGAs | VAL_LGAs)
      - TEST:        test_year x TEST_LGAs (unseen)

    If split_type == "temporal":
      - No LGA partition; use all LGAs per time window; FULL_TRAIN = Train|Val years.

    Returns:
      full_train, train_df, val_df, test_df
    """
    df = data.copy()

    if split_type == "lga_temporal":
        rng = np.random.RandomState(random_state)
        lgas = df[lga_col].dropna().unique()
        rng.shuffle(lgas)
        n_tr, n_va, n_te = _split_counts_temp(len(lgas), fracs=(0.4, 0.4, 0.2))
        train_lgas = set(lgas[:n_tr])
        val_lgas   = set(lgas[n_tr:n_tr+n_va])
        test_lgas  = set(lgas[n_tr+n_va:])

        train_mask = df[lga_col].isin(train_lgas)
        val_mask   = df[lga_col].isin(val_lgas)
        test_mask  = df[lga_col].isin(test_lgas)

        train_df = df[(df[time_col] >= train_start) & (df[time_col] <= train_end) & train_mask].copy()
        val_df   = df[(df[time_col] == val_year)    & val_mask  ].copy()
        test_df  = df[(df[time_col] == test_year)   & test_mask ].copy()

        full_train_mask = df[lga_col].isin(train_lgas.union(val_lgas)) & \
                          (df[time_col] >= train_start) & (df[time_col] <= val_year)
        full_train = df[full_train_mask].copy()
        return full_train, train_df, val_df, test_df

    # ----- Plain temporal (no LGA partition) -----
    train_df   = df[(df[time_col] >= train_start) & (df[time_col] <= train_end)].copy()
    val_df     = df[(df[time_col] == val_year)].copy()
    full_train = df[(df[time_col] >= train_start) & (df[time_col] <= val_year)].copy()
    test_df    = df[(df[time_col] == test_year)].copy()
    return full_train, train_df, val_df, test_df
