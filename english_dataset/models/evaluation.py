"""
evaluation.py

Centralized evaluation utilities.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


def evaluate_model(
    model,
    X,
    y_true,
    labels,
    model_name,
    vectorizer_name,
    split,
    save_csv: Optional[str] = None,
):
    """
    Evaluate predictions and build results table.
    """

    y_pred = model.predict(X)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    rows = []

    for i, class_name in enumerate(labels):

        item = report.get(class_name, {})

        rows.append({
            "Model": model_name,
            "Vectorizer": vectorizer_name,
            "Split": split,
            "Class": class_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(item.get("precision", 0.0), 4),
            "Recall": round(item.get("recall", 0.0), 4),
            "F1-score": round(item.get("f1-score", 0.0), 4),
            "Support": int(item.get("support", 0)),
            "Macro F1": round(macro_f1, 4),
        })

    results_df = pd.DataFrame(rows)

    if save_csv is not None:
        Path(save_csv).resolve().parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_csv, index=False)
        print(f"  Saved table → {save_csv}")

    print(f"\n--- {model_name} | {vectorizer_name} | {split} ---")
    print(results_df.to_string(index=False))

    return results_df, y_pred


# Confusion Matrix
def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    model_name,
    vectorizer_name,
    split,
    save_dir="results",
):
    """
    Save confusion matrix as PNG.
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(
        f"{model_name} | {vectorizer_name} | {split}"
    )

    filename = (
        f"{model_name}_{vectorizer_name}_{split}"
        .replace(" ", "_")
        .lower()
    )

    plt.tight_layout()

    plt.savefig(
        Path(save_dir) / f"{filename}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# Cross Validation
def run_cross_validation(
    estimator,
    X,
    y,
    cv_splits=5,
):
    """
    Run cross-validation on best estimator.
    """

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42,
    )

    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1,
    )

    print("\nCross Validation Results")
    # Print the results of the 5-fold cross-validation
    print("Test Accuracy scores:", results['test_accuracy'])
    print("Mean Test Accuracy:", results['test_accuracy'].mean())

    print("\nTest F1 Macro scores:", results['test_f1_macro'])
    print("Mean Test F1 Macro:", results['test_f1_macro'].mean())


    return results
