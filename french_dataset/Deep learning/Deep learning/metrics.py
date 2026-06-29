import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)


def epoch_log(epoch, total, train_loss, val_loss, val_acc):
    print(f"Epoch [{epoch:>3}/{total}]  "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  "
          f"val_acc={val_acc:.4f}")


def evaluate(y_true, y_pred, label_encoder=None, model_name="Model"):
    labels      = list(label_encoder.classes_) if label_encoder else None
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print(f"Results — {model_name}")
    print("=" * 50)
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 macro    : {f1_macro:.4f}")
    print(f"F1 weighted : {f1_weighted:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def plot_confusion_matrix(y_true, y_pred, label_encoder=None,
                          title="Confusion Matrix", save_path=None):
    labels = list(label_encoder.classes_) if label_encoder else None
    cm     = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if labels:
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[metrics] Saved → {save_path}")
    else:
        plt.show()

    plt.close()
