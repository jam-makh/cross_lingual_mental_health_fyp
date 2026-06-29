import argparse
import os
import json
import csv
import sys
import uuid
from datetime import datetime

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from models import get_model
from trainer import Trainer
from metrics import evaluate, plot_confusion_matrix


# ───────────────────────── LOGGER ─────────────────────────
class Logger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


# ───────────────────────── PLOTS ─────────────────────────
def plot_training(history, save_path):

    plt.style.use("default")

    # LOSS
    plt.figure(figsize=(8,5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")

    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend()
    plt.savefig(save_path + "_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ACCURACY
    plt.figure(figsize=(8,5), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    plt.plot(history["val_acc"], label="Validation Accuracy")

    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend()
    plt.savefig(save_path + "_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


# ───────────────────────── SAVE RESULTS ─────────────────────────
def save_results(cfg, history, metrics, results_dir, run_id, timestamp):

    os.makedirs(results_dir, exist_ok=True)

    record = {
        "timestamp": timestamp,
        "run_name": run_id,
        "vectorizer": cfg["vectorizer"],
        "architecture": cfg["model"]["architecture"],
        "input_size": cfg["model"]["input_size"],
        "hidden_size": cfg["model"]["hidden_size"],
        "num_layers": cfg["model"]["num_layers"],
        "dropout": cfg["model"]["dropout"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "learning_rate": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "clip_grad_norm": cfg["training"]["clip_grad_norm"],
        "patience": cfg["training"]["patience"],
        "train_mean_accuracy": metrics["train_accuracy"],
        "test_mean_accuracy": metrics["test_accuracy"],
        "test_mean_f1_macro": metrics["test_f1_macro"],
    }

    # CSV
    csv_path = os.path.join(results_dir, "results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    # JSON
    json_path = os.path.join(results_dir, f"{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": record, "history": history}, f, indent=2)

    # TXT
    txt_path = os.path.join(results_dir, "epochs.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write(f"RUN: {run_id}\n")
        f.write("="*70 + "\n")

        for i in range(len(history["train_loss"])):
            f.write(
                f"Epoch {i+1} | "
                f"Train Loss: {history['train_loss'][i]:.4f} | "
                f"Val Loss: {history['val_loss'][i]:.4f} | "
                f"Val Acc: {history['val_acc'][i]:.4f}\n"
            )


# ───────────────────────── MAIN ─────────────────────────
def main(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vec = cfg["vectorizer"].lower()
    arch = cfg["model"]["architecture"].upper()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{arch}_{vec}_{timestamp}_{uuid.uuid4().hex[:6]}"

    results_dir = "results1"
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print(" French Mental Health Classifier")
    print(f" Model      : {arch}")
    print(f" Vectorizer : {vec}")
    print("="*60)

    # LOAD DATA
    df = pd.read_csv(cfg["paths"]["data"])
    df = df.dropna(subset=[cfg["dataset"]["text_col"], cfg["dataset"]["label_col"]])

    le = LabelEncoder()
    y = le.fit_transform(df[cfg["dataset"]["label_col"]].values)
    X = df[cfg["dataset"]["text_col"]].values

    print(f"[DATA] Samples: {len(X)}")

    # SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg["dataset"]["test_size"],
        random_state=cfg["dataset"]["random_state"],
        stratify=y
    )

    # VECTORIZE
    if vec == "camembert":
        from camembert import CamembertVectorizer
        v = CamembertVectorizer()
        X_train = v.encode_dataset(X_train.tolist())
        X_val = v.encode_dataset(X_val.tolist())

    else:
        from tfidf import TfidfVectorizerWrapper
        v = TfidfVectorizerWrapper(max_features=cfg["model"]["input_size"])

        X_train = v.fit_transform(X_train)
        X_val = v.transform(X_val)

        # 🔥 CRITICAL FIX (THIS WAS YOUR ERROR)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_val = X_val.toarray()

        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)

    # MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    trainer = Trainer(model, cfg, device)

    # TRAIN
    trainer.fit(X_train, y_train, X_val, y_val, label_encoder=le)

    # PREDICT
    y_pred = trainer.predict(X_val)
    y_train_pred = trainer.predict(X_train)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_val, y_pred),
        "test_f1_macro": f1_score(y_val, y_pred, average="macro")
    }

    print("[RESULTS]", metrics)

    # PLOTS
    plot_training(trainer.history, os.path.join(results_dir, run_id))

    plot_confusion_matrix(
        y_val, y_pred,
        label_encoder=le,
        title=f"Confusion Matrix - {run_id}",
        save_path=os.path.join(results_dir, f"{run_id}_cm.png")
    )

    evaluate(y_val, y_pred, label_encoder=le, model_name=run_id)

    save_results(cfg, trainer.history, metrics, results_dir, run_id, timestamp)

    print(f"\n[DONE] Saved in {results_dir}")


# ENTRY
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    log_dir = "results1"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    sys.stdout = Logger(log_path)

    try:
        main(args.config)
    finally:
        sys.stdout.close()
        print(f"[LOG SAVED] {log_path}")