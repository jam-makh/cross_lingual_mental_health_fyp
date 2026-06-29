import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_french_dataset(path, text_col="text_nostop", label_col="mental_state",
                         test_size=0.2, random_state=42):

    print(f"[dataset] Loading: {path}")
    df = pd.read_csv(path)
    print(f"[dataset] Shape: {df.shape}")

    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    if len(df) < before:
        print(f"[dataset] Dropped {before - len(df)} null rows.")

    print(f"[dataset] Class distribution:\n{df[label_col].value_counts().to_string()}")

    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    print(f"[dataset] Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df[text_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[dataset] Train: {len(X_train)} | Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, le
