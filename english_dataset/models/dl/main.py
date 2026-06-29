"""
main.py

Train deep-learning classifiers on cached vectorizer features.

Expected cache files inside CACHE_DIR:
    y_train.pkl
    y_test.pkl
    X_train_tfidf.npz
    X_test_tfidf.npz
    X_train_distilbert.npy
    X_test_distilbert.npy

The script trains every requested combination:
    tfidf + lstm
    tfidf + bilstm
    tfidf + cnn_rnn
    distilbert + lstm
    distilbert + bilstm
    distilbert + cnn_rnn

Outputs:
    results_deep/<vectorizer>_<architecture>_loss_curve.png
    results_deep/<vectorizer>_<architecture>_accuracy_curve.png
    results_deep/<vectorizer>_<architecture>_confusion_matrix.png
    results_deep/<vectorizer>_<architecture>_classification_report.csv
    results_deep/final_deep_results.csv
    results_deep/final_deep_summary.csv
    models/checkpoints/<vectorizer>_<architecture>_best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.sparse import issparse, load_npz
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from evaluation_dl import evaluate_predictions, save_confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_deep_models_module():
    """
    Import deep model definitions.

    Supports both names:
        deep_models.py
        deep_models(1).py
    because the uploaded file currently uses deep_models(1).py.
    """
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "deep_models.py",
        base_dir / "deep_models(1).py",
    ]

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("deep_models_dynamic", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not find deep_models.py or deep_models(1).py next to main.py."
    )


def resolve_cache_dir(config: Dict[str, Any], cli_cache_dir: str | None) -> Path:
    """
    Locate the feature cache folder.

    Priority:
        1. --cache_dir argument
        2. config paths.cache_dir
        3. ./cache
        4. current folder
    """
    base_dir = Path(__file__).resolve().parent

    candidates: List[Path] = []

    if cli_cache_dir:
        candidates.append(Path(cli_cache_dir))

    paths_cfg = config.get("paths", {}) or {}
    if paths_cfg.get("cache_dir"):
        candidates.append(Path(paths_cfg["cache_dir"]))

    candidates.extend([
        base_dir / "cache",
        base_dir,
    ])

    required_any = [
        "X_train_tfidf.npz",
        "X_train_distilbert.npy",
    ]

    for candidate in candidates:
        candidate = candidate if candidate.is_absolute() else (base_dir / candidate)
        if not candidate.exists():
            continue
        has_labels = (candidate / "y_train.pkl").exists() and (candidate / "y_test.pkl").exists()
        has_features = any((candidate / name).exists() for name in required_any)
        if has_labels and has_features:
            return candidate.resolve()

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not locate cache directory with y_train.pkl, y_test.pkl, and feature files.\n"
        f"Checked:\n{checked}\n\n"
        "Pass it manually with: python main.py --cache_dir path/to/cache"
    )


def load_cache(cache_dir: Path) -> Dict[str, Any]:
    cache: Dict[str, Any] = {
        "y_train": joblib.load(cache_dir / "y_train.pkl"),
        "y_test": joblib.load(cache_dir / "y_test.pkl"),
    }

    tfidf_train = cache_dir / "X_train_tfidf.npz"
    tfidf_test = cache_dir / "X_test_tfidf.npz"
    if tfidf_train.exists() and tfidf_test.exists():
        cache["tfidf"] = (
            load_npz(str(tfidf_train)),
            load_npz(str(tfidf_test)),
        )

    distilbert_train = cache_dir / "X_train_distilbert.npy"
    distilbert_test = cache_dir / "X_test_distilbert.npy"
    if distilbert_train.exists() and distilbert_test.exists():
        cache["distilbert"] = (
            np.load(str(distilbert_train)),
            np.load(str(distilbert_test)),
        )

    return cache


def to_float32_dense(x: Any) -> np.ndarray:
    """Convert sparse TF-IDF or dense embedding arrays to float32 dense arrays."""
    if issparse(x):
        return x.astype(np.float32).toarray()
    return np.asarray(x, dtype=np.float32)


def encode_labels(y_train: Any, y_test: Any) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train_arr, y_test_arr]))

    y_train_enc = encoder.transform(y_train_arr).astype(np.int64)
    y_test_enc = encoder.transform(y_test_arr).astype(np.int64)
    label_names = [str(x) for x in encoder.classes_]

    return y_train_enc, y_test_enc, label_names


def make_loaders(
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    random_state: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split the training cache into train/validation, keeping test untouched.
    """
    stratify = y_train_full if len(np.unique(y_train_full)) > 1 else None

    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.20,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.20,
            random_state=random_state,
            stratify=None,
        )

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    clip_grad_norm: float | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                loss.backward()
                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)

            all_true.extend(yb.detach().cpu().numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)

    return avg_loss, acc


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_true.extend(yb.numpy().tolist())
            all_pred.extend(preds.detach().cpu().numpy().tolist())

    return np.asarray(all_true), np.asarray(all_pred)


def save_loss_curve(history: Dict[str, List[float]], run_name: str, results_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{run_name} - Loss Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"{run_name}_loss_curve.png", dpi=300)
    plt.close()


def save_accuracy_curve(history: Dict[str, List[float]], run_name: str, results_dir: Path) -> None:
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{run_name} - Accuracy Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"{run_name}_accuracy_curve.png", dpi=300)
    plt.close()


def train_combination(
    vectorizer_name: str,
    architecture: str,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    config: Dict[str, Any],
    deep_models_module,
    results_dir: Path,
    checkpoints_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    dataset_cfg = config.get("dataset", {}) or {}

    random_state = int(dataset_cfg.get("random_state", 42))
    batch_size = int(training_cfg.get("batch_size", 32))
    epochs = int(training_cfg.get("epochs", 50))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    patience = int(training_cfg.get("patience", 5))
    scheduler_factor = float(training_cfg.get("scheduler_factor", 0.5))
    scheduler_patience = int(training_cfg.get("scheduler_patience", 3))
    clip_grad_norm = float(training_cfg.get("clip_grad_norm", 1.0))

    input_dim = int(x_train_full.shape[1])
    num_classes = len(label_names)

    train_loader, val_loader, test_loader = make_loaders(
        x_train_full=x_train_full,
        y_train_full=y_train_full,
        x_test=x_test,
        y_test=y_test,
        batch_size=batch_size,
        random_state=random_state,
    )

    model = deep_models_module.get_model(
        architecture=architecture,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        cnn_out_channels=int(model_cfg.get("cnn_out_channels", 128)),
        cnn_kernel_size=int(model_cfg.get("cnn_kernel_size", 3)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    run_name = f"{vectorizer_name}_{architecture}".lower().replace("-", "_")
    checkpoint_path = checkpoints_dir / f"{run_name}_best.pt"

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            clip_grad_norm=clip_grad_norm,
        )
        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"[{run_name}] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vectorizer": vectorizer_name,
                    "architecture": architecture,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "label_names": label_names,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{run_name}] Early stopping at epoch {epoch}.")
                break

    # Load best checkpoint before final test evaluation.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_train_true, y_train_pred = predict(model, train_loader, device)

    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    train_macro_f1 = f1_score(y_train_true, y_train_pred, average="macro", zero_division=0)

    train_report_df = evaluate_predictions(
        y_true=y_train_true,
        y_pred=y_train_pred,
        labels=label_names,
        model_name=run_name,
    )
    train_report_df.insert(1, "Vectorizer", vectorizer_name)
    train_report_df.insert(2, "Architecture", architecture)
    train_report_df.to_csv(results_dir / f"{run_name}_train_classification_report.csv", index=False)

    y_true, y_pred = predict(model, test_loader, device)

    test_accuracy = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report_df = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=label_names,
        model_name=run_name,
    )
    report_df.insert(1, "Vectorizer", vectorizer_name)
    report_df.insert(2, "Architecture", architecture)
    report_df.to_csv(results_dir / f"{run_name}_test_classification_report.csv", index=False)

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=label_names,
        model_name=run_name,
        save_dir=str(results_dir),
    )

    save_loss_curve(history, run_name, results_dir)
    save_accuracy_curve(history, run_name, results_dir)

    with open(results_dir / f"{run_name}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "Vectorizer": vectorizer_name,
        "Architecture": architecture,
        "Run": run_name,
        "Input_Dim": input_dim,
        "Best_Val_Loss": best_val_loss,
        "Train_Accuracy": train_accuracy,
        "Train_Macro_F1": train_macro_f1,
        "Test_Accuracy": test_accuracy,
        "Test_Macro_F1": test_macro_f1,
        "Epochs_Ran": len(history["train_loss"]),
        "Checkpoint": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--cache_dir", default=None, help="Folder containing cached embeddings/features")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    config = load_yaml_config(config_path)
    seed = int((config.get("dataset", {}) or {}).get("random_state", 42))
    set_seed(seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths_cfg = config.get("paths", {}) or {}
    base_dir = Path(__file__).resolve().parent
    results_dir = Path(paths_cfg.get("results", "results_deep"))
    checkpoints_dir = Path(paths_cfg.get("checkpoints", "models/checkpoints"))

    if not results_dir.is_absolute():
        results_dir = base_dir / results_dir
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = base_dir / checkpoints_dir

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = resolve_cache_dir(config, args.cache_dir)
    cache = load_cache(cache_dir)
    deep_models_module = import_deep_models_module()

    y_train, y_test, label_names = encode_labels(cache["y_train"], cache["y_test"])

    vectorizers = config.get("vectorizers", [])
    architectures = (config.get("model", {}) or {}).get("architectures", [])

    if isinstance(architectures, str):
        architectures = [architectures]

    print(f"Using device: {device}")
    print(f"Using cache: {cache_dir}")
    print(f"Labels: {label_names}")
    print(f"Vectorizers requested: {vectorizers}")
    print(f"Architectures requested: {architectures}")

    summary_rows: List[Dict[str, Any]] = []
    all_report_rows: List[pd.DataFrame] = []

    for vectorizer_name in vectorizers:
        if vectorizer_name not in cache:
            print(f"Skipping {vectorizer_name}: cache files were not found.")
            continue

        x_train_raw, x_test_raw = cache[vectorizer_name]
        x_train = to_float32_dense(x_train_raw)
        x_test = to_float32_dense(x_test_raw)

        for architecture in architectures:
            print("=" * 80)
            print(f"Training combination: {vectorizer_name} + {architecture}")
            print("=" * 80)

            row = train_combination(
                vectorizer_name=vectorizer_name,
                architecture=architecture,
                x_train_full=x_train,
                y_train_full=y_train,
                x_test=x_test,
                y_test=y_test,
                label_names=label_names,
                config=config,
                deep_models_module=deep_models_module,
                results_dir=results_dir,
                checkpoints_dir=checkpoints_dir,
                device=device,
            )
            summary_rows.append(row)

            report_path = results_dir / f"{row['Run']}_test_classification_report.csv"
            all_report_rows.append(pd.read_csv(report_path))

    if not summary_rows:
        raise RuntimeError(
            "No model was trained. Check your vectorizers in config.yaml and cache files."
        )

    final_summary = pd.DataFrame(summary_rows)
    final_summary.to_csv(results_dir / "final_deep_summary.csv", index=False)

    final_results = pd.concat(all_report_rows, ignore_index=True)
    final_results.to_csv(results_dir / "final_deep_results.csv", index=False)

    print("\nTraining complete.")
    print(f"Saved outputs to: {results_dir}")
    print(f"Saved checkpoints to: {checkpoints_dir}")
    print("\nSummary:")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
