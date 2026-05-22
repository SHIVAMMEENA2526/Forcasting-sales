"""
utils.py — Helper functions for fan sales forecasting project
"""

import pandas as pd
import numpy as np


def calculate_mape(actual: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) per competition rules.

    Rules:
      - MAPE = |actual - forecast| / actual       if actual > 0
      - MAPE = 0%                                  if actual == 0 and forecast == 0
      - MAPE = 100%                                if actual == 0 and forecast != 0

    Returns:
        Mean MAPE across all rows (as a percentage, e.g. 46.55 for 46.55%)
    """
    errors = []
    for a, f in zip(actual, forecast):
        if a > 0:
            errors.append(abs(a - f) / a)
        elif a == 0 and f == 0:
            errors.append(0.0)
        else:  # actual == 0, forecast != 0
            errors.append(1.0)
    return float(np.mean(errors)) * 100


def load_train_data(path: str = "data/train_data.xlsx") -> pd.DataFrame:
    """Load and lightly clean the training data."""
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.replace("-01 00:00:00", "", regex=False)
    return df


def get_month_columns(df: pd.DataFrame) -> list:
    """Return list of month column names (excludes metadata columns)."""
    meta = {"Warehouse id", "Region", "SKU id"}
    return [c for c in df.columns if c not in meta]


def split_train_val(df: pd.DataFrame, val_month: str):
    """
    Split dataframe into train features and validation target.

    Args:
        df: full dataframe with month columns
        val_month: the month column to use as validation target (e.g. '2021-05')

    Returns:
        X_train (all months before val_month), y_val (val_month column)
    """
    month_cols = get_month_columns(df)
    val_idx = month_cols.index(val_month)
    train_months = month_cols[:val_idx]
    return df[train_months], df[val_month]


def min_of_last_two_months(df: pd.DataFrame) -> pd.Series:
    """
    Final model: predict next month as min(last month, second-to-last month).
    Exploits MAPE asymmetry — under-forecasting is penalized less.
    """
    month_cols = get_month_columns(df)
    last = df[month_cols[-1]]
    second_last = df[month_cols[-2]]
    return last.combine(second_last, min)
