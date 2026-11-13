"""Attrition analysis: data processing pipeline and statistical comparison
of attrition vs non-attrition samples."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# =====================================================================
# Data processing pipeline
# =====================================================================

def combine_data(resilience_data, ppp_data, cpi_data, merge_column='surveyprd'):
    """Merge resilience, PPP, and CPI data on a common column.

    Parameters
    ----------
    resilience_data : DataFrame
    ppp_data : DataFrame
    cpi_data : DataFrame
    merge_column : str

    Returns
    -------
    DataFrame
    """
    combined_data = pd.merge(resilience_data, ppp_data, on=merge_column, how='inner')
    combined_data = pd.merge(combined_data, cpi_data, on=merge_column, how='inner')
    return combined_data


def convert_consumption_to_constant_naira(data):
    """Convert consumption to 2016 constant Naira (urban-rural adjusted).

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame
    """
    data = data.copy()
    data['totcons_per_capita_per_day_2016cst'] = (
        data.totcons_per_capita_per_day * (1 + data.inflation_rate)
    )

    cpi_rural_2016 = 199.53
    cpi_urban_2016 = 201.47
    inflation_rate_urban_rural_diff = round(
        (cpi_rural_2016 - cpi_urban_2016) / cpi_urban_2016, 4
    )

    data['inflation_rate_urban_rural_2016'] = np.where(
        data.rural == 0, inflation_rate_urban_rural_diff, 0
    )
    data['totcons_per_capita_per_day_2016cst'] = (
        data['totcons_per_capita_per_day_2016cst']
        * (1 + data['inflation_rate_urban_rural_2016'])
    )
    data['totcons_per_capita_per_day_2016cst_naira'] = data['totcons_per_capita_per_day_2016cst']
    data['ltotcons_per_capita_per_day_2016cst_naira'] = np.log(
        data['totcons_per_capita_per_day_2016cst_naira']
    )
    return data


def convert_to_ppp_and_calculate_poverty_dummies(
    data,
    ppp_conversion_factor=101.91,
    national_poverty_line_2019_daily=137430 / 365,
):
    """Convert consumption to $PPP and compute poverty dummies.

    Parameters
    ----------
    data : DataFrame
    ppp_conversion_factor : float
    national_poverty_line_2019_daily : float

    Returns
    -------
    DataFrame
    """
    data = data.copy()
    data['totcons_per_capita_per_day_2016cst'] = (
        data['totcons_per_capita_per_day_2016cst'] / ppp_conversion_factor
    )

    inflation_rate_2019_2016 = (200.30 - 291.4) / 291.4
    national_poverty_line_2016 = national_poverty_line_2019_daily * (1 + inflation_rate_2019_2016)
    national_poverty_line_2016 = round(national_poverty_line_2016 / ppp_conversion_factor, 2)

    international_poverty_line_2016 = 1.9
    data['poverty_dummy_wb'] = np.where(
        data.totcons_per_capita_per_day_2016cst < international_poverty_line_2016, 1, 0
    )
    data['poverty_dummy_nat'] = np.where(
        data.totcons_per_capita_per_day_2016cst < national_poverty_line_2016, 1, 0
    )
    return data


def compute_national_poverty_line(
    ppp_conversion_factor=101.91,
    national_poverty_line_2019_daily=137430 / 365,
):
    """Compute the national poverty line in 2016 $PPP terms.

    Returns
    -------
    float
    """
    inflation_rate_2019_2016 = (200.30 - 291.4) / 291.4
    national_poverty_line_2016 = national_poverty_line_2019_daily * (1 + inflation_rate_2019_2016)
    national_poverty_line_2016 = round(national_poverty_line_2016 / ppp_conversion_factor, 2)
    return national_poverty_line_2016


def compute_log_consumption_and_prepare_for_analysis(data):
    """Compute log consumption, lagged values, and dummies for panel analysis.

    Parameters
    ----------
    data : DataFrame

    Returns
    -------
    DataFrame
    """
    data = data.copy()
    data['ltotcons_per_capita_per_day_2016cst'] = np.log(
        data.totcons_per_capita_per_day_2016cst
    )
    data = data.sort_values(by=['hhid', 'surveyprd'])
    data_panel = data.set_index(['hhid', 'surveyprd'])
    data_panel['lag_ltotcons_per_capita_per_day_2016cst'] = (
        data_panel.groupby(['hhid']).ltotcons_per_capita_per_day_2016cst.shift()
    )
    data_panel = data_panel.reset_index()
    data_panel['lag_ltotcons_per_capita_per_day_2016cst_sqr'] = (
        data_panel.lag_ltotcons_per_capita_per_day_2016cst ** 2
    )
    data_panel['lag_ltotcons_per_capita_per_day_2016cst_cub'] = (
        data_panel.lag_ltotcons_per_capita_per_day_2016cst ** 3
    )
    data_panel['survey_year'] = data_panel.surveyprd

    data_panel = pd.get_dummies(
        data_panel,
        columns=[
            'survey_year', 'marital_status_head_hh', 'crop_livestock_hh',
            'religion_head_hh', 'srtm_nga_5_15', 'sq7',
            'main_cook_fuel', 'main_source_water', 'main_toilet_facility',
            'education_level_head_hh',
        ],
        drop_first=False,
    )
    return data_panel


def final_process_data(data, list_all):
    """Handle missing values and drop rows with NaN in required columns.

    Parameters
    ----------
    data : DataFrame
    list_all : list
        List of column names that must be non-null.

    Returns
    -------
    DataFrame
    """
    data = data.copy()
    data['age_head_hh'] = data['age_head_hh'].replace(999, np.nan)
    data['age_head_hh'] = data.groupby(['hhid', 'fhh'])['age_head_hh'].bfill()
    data['number_rooms'] = data.groupby(['hhid'])['number_rooms'].bfill()
    data.dropna(subset=list_all, inplace=True)
    return data


# =====================================================================
# Significance helper
# =====================================================================

def significance_asterisks(p_value):
    """Return significance stars for a p-value."""
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    else:
        return ''


# =====================================================================
# Attrition comparison pipeline
# =====================================================================

def compare_attrition_samples(
    sample_without_attrition,
    sample_with_attrition,
    columns,
    label_without="without_attrition",
    label_with="with_attrition",
):
    """T-test comparison of two samples across a list of variables.

    Parameters
    ----------
    sample_without_attrition : DataFrame
    sample_with_attrition : DataFrame
    columns : list
        Variable names to compare.
    label_without : str
        Label for the sample without attrition.
    label_with : str
        Label for the sample with attrition.

    Returns
    -------
    DataFrame
        Comparison results with means, SDs, differences, t-stats, and p-values.
    """
    results = []

    for column in columns:
        data_without = sample_without_attrition[column].dropna()
        data_with = sample_with_attrition[column].dropna()

        t_stat, p_value = ttest_ind(data_without, data_with, equal_var=True)

        mean1 = round(data_without.mean(), 2)
        mean2 = round(data_with.mean(), 2)
        std1 = round(data_without.std(), 2)
        std2 = round(data_with.std(), 2)
        n1 = len(data_without)
        n2 = len(data_with)

        mean_diff = mean1 - mean2
        std_diff = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
        std_diff = round(std_diff, 3)

        asterisks = significance_asterisks(p_value)

        results.append({
            'Variable': column,
            f'Mean_{label_without}': mean1,
            f'Mean_{label_with}': mean2,
            f'Std_{label_without}': std1,
            f'Std_{label_with}': std2,
            'Mean_Difference': mean_diff,
            'Std_Difference': std_diff,
            'T-Statistic': t_stat,
            'P-Value': p_value,
            'Significance': asterisks,
        })

    return pd.DataFrame(results)


def run_attrition_check(
    attrition_data_path,
    ppp_data_path,
    cpi_data_path,
    train_set,
    test_set,
    target,
    list_all,
):
    """Full attrition analysis pipeline.

    Parameters
    ----------
    attrition_data_path : str
        Path to Attrition_sample_for_testing.xlsx.
    ppp_data_path : str
        Path to nigeria_ppp.csv.
    cpi_data_path : str
        Path to CPI_Nigeria_BS_25022022.csv.
    train_set : DataFrame
        Training set without attrition.
    test_set : DataFrame
        Test set without attrition.
    target : str
        Target variable name.
    list_all : list
        All required variable names.

    Returns
    -------
    (results_df_train, results_df_test)
        Comparison DataFrames for train and test sets.
    """
    attrition_data = pd.read_excel(attrition_data_path)
    ppp_data = pd.read_csv(ppp_data_path, sep=';')
    cpi_data = pd.read_csv(cpi_data_path, sep=';')

    combined = combine_data(attrition_data, ppp_data, cpi_data)
    processed = convert_consumption_to_constant_naira(combined)
    with_ppp = convert_to_ppp_and_calculate_poverty_dummies(processed)
    prepared = compute_log_consumption_and_prepare_for_analysis(with_ppp)
    final = final_process_data(prepared, list_all)

    # Split attrition data into train and test
    # Train = all years except the last, test = the last year
    years = sorted(final['surveyprd'].unique()) if 'surveyprd' in final.columns else []
    if len(years) > 1:
        test_year = years[-1]
        training_with_attrition = final[final['surveyprd'] != test_year]
        test_with_attrition = final[final['surveyprd'] == test_year]
    else:
        training_with_attrition = final
        test_with_attrition = final

    columns_to_compare = [target] + list_all

    results_df_train = compare_attrition_samples(
        sample_without_attrition=train_set,
        sample_with_attrition=training_with_attrition,
        columns=columns_to_compare,
        label_without="Train_without_attrition",
        label_with="Train_with_attrition",
    )

    results_df_test = compare_attrition_samples(
        sample_without_attrition=test_set,
        sample_with_attrition=test_with_attrition,
        columns=columns_to_compare,
        label_without="Test_without_attrition",
        label_with="Test_with_attrition",
    )

    return results_df_train, results_df_test
