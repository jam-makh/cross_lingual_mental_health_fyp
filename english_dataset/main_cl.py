#!/usr/bin/env python3

"""
Main orchestration module for mental health text classification pipeline.
"""

from typing import Optional
from pathlib import Path

import pandas as pd

from clean_eda.cleaning import Cleaning
from clean_eda.eda import EDA


def clean_dataset(
    csv_path: str,
    text_column: str = "text",
    cleaner: Optional[Cleaning] = None,
) -> pd.DataFrame:
    """
    Load a CSV file, apply :class:`Cleaning`, and return the result.
    All data-integrity and text-cleaning logic lives inside
    :class:`Cleaning`.

    :param csv_path: Path to the raw CSV file.
    :type csv_path: str
    :param text_column: Name of the column holding raw post text.
    :type text_column: str
    :param cleaner: Optional pre-instantiated :class:Cleaning object.
        Pass one in when calling this function repeatedly so the spaCy
        model is loaded only once.
    :type cleaner: Cleaning or None
    :returns: Cleaned DataFrame ready for feature extraction.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: If csv_path does not exist.
    :raises ValueError: If text_column is absent from the CSV.
    """
    print(f"LOADING DATASET : {csv_path}")

    df: pd.DataFrame = pd.read_csv(csv_path)
    print(f"\nShape before cleaning : {df.shape}")

    if "Unique_ID" in df.columns:
        df = df.drop(columns=["Unique_ID"])
        print("  Dropped column  : Unique_ID")

    print(f"\nRunning Cleaning pipeline on column '{text_column}'…")
    c: Cleaning = cleaner if cleaner is not None else Cleaning()
    df = c.clean_dataframe(df, text_column=text_column)

    print(f"Shape after  cleaning : {df.shape}")
    print(f"{'='*60}\n")
    return df


def save_features_to_csv(
    df: pd.DataFrame, output_dir: str = "plots"
) -> str:
    """
    Save the feature-engineered DataFrame to a CSV file.

    Drops the tokens column (which contains lists) before saving,
    as it does not serialize cleanly to CSV format.

    :param df: Feature-engineered DataFrame from :meth:EDA.extract_features.
    :type df: pandas.DataFrame
    :param output_dir: Directory where the CSV will be saved.
    :type output_dir: str
    :returns: Path to the saved CSV file.
    :rtype: str
    """
    output_path = Path(output_dir) / "processed_data_with_features.csv"
    df_to_save = df.drop(columns=["tokens"], errors="ignore")
    df_to_save.to_csv(output_path, index=False)
    print(f"Feature-engineered data saved to: {output_path}")
    return str(output_path)


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    cleaner: Optional[Cleaning] = None,
) -> pd.DataFrame:
    """
    Clean a preloaded DataFrame using the shared :class:`Cleaning` pipeline.

    This function can be imported and reused from other modules when you
    already have a DataFrame in memory.

    :param df: Input DataFrame containing raw text data.
    :type df: pandas.DataFrame
    :param text_column: Name of the column holding raw post text.
    :type text_column: str
    :param cleaner: Optional existing :class:`Cleaning` instance.
        Use this to reuse a loaded spaCy model across calls.
    :type cleaner: Cleaning or None
    :returns: Cleaned DataFrame with a new ``cleaned_text`` column.
    :rtype: pandas.DataFrame
    :raises ValueError: If text_column is absent from df.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. "
            f"Available: {df.columns.tolist()}"
        )

    df = df.copy()

    cleaner_instance = cleaner if cleaner is not None else Cleaning()
    return cleaner_instance.clean_dataframe(df, text_column=text_column)


def run_eda_pipeline(
    csv_path: str, output_dir: str = "plots"
) -> pd.DataFrame:
    """
    Orchestrate the complete EDA pipeline: cleaning, feature extraction,
    visualization, and CSV export.

    :param csv_path: Path to the raw CSV file.
    :type csv_path: str
    :param output_dir: Directory to save visualization plots and CSV.
    :type output_dir: str
    :returns: DataFrame with extracted features.
    :rtype: pandas.DataFrame
    """
    cleaner = Cleaning()
    df = clean_dataset(csv_path, cleaner=cleaner)
    eda = EDA(cleaner, output_dir=output_dir)
    df = eda.extract_features(df)
    eda.run_plots(df)
    eda.save_feature_summary(df)
    save_features_to_csv(df, output_dir=output_dir)
    return df


if __name__ == "__main__":
    run_eda_pipeline("mental_heath_unbanlanced.csv", output_dir="plots")
