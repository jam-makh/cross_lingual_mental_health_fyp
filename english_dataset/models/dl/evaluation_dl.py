from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

def evaluate_predictions(
    y_true,
    y_pred,
    labels,
    model_name,
):
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

    for label in labels:

        metrics = report[label]

        rows.append({
            "Model": model_name,
            "Class": label,
            "Accuracy": accuracy,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1-score"],
            "Support": metrics["support"],
            "Macro_F1": macro_f1,
        })

    return pd.DataFrame(rows)

def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    model_name,
    save_dir="results_deep",
):
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

    plt.tight_layout()

    plt.savefig(
        Path(save_dir)
        / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

def save_training_curves(
    history,
    model_name,
    save_dir="results_deep",
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        Path(save_dir)
        / f"{model_name.lower()}_loss_curve.png",
        dpi=300,
    )

    plt.close()

def save_results_csv(
    results,
    save_path="results_deep/final_deep_results.csv",
):
    pd.DataFrame(results).to_csv(
        save_path,
        index=False,
    )
