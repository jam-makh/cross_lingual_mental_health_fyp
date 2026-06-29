"""
Mental Health Text Classification EDA Pipeline.

A comprehensive exploratory data analysis and text preprocessing pipeline
for mental health social media text classification.

Modules:
    - cleaning: Pydantic models and Cleaning class
    - main: Orchestration (clean_dataset, run_eda_pipeline)
    - eda: Visualization and analysis (EDA class)

Usage::

    from clean_eda import clean_dataset, run_eda_pipeline

    # Option 1: Clean data only
    df = clean_dataset("data.csv")

    # Option 2: Run full pipeline
    df = run_eda_pipeline("data.csv", output_dir="plots")
"""

from clean_eda.cleaning import Cleaning, MentalHealthPost, PostAnalysis
from clean_eda.eda import EDA
# package metadata
__version__ = "4.0"
__author__ = "Joseph Am-Makhlouf"

__all__ = [
    "Cleaning",
    "MentalHealthPost",
    "PostAnalysis",
    "EDA",
    "clean_dataset", # chouf kif tzid l function
]
