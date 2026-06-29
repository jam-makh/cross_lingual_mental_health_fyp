"""
Train and evaluate ML models on cached embeddings.

Current configuration:
    TF-IDF      → Logistic Regression, SVM, Multinomial Naive Bayes
    DistilBERT  → Logistic Regression, SVM, Gaussian Naive Bayes

Saved artefacts
---------------
results/models/<model>_<vectorizer>.pt
results/models/<model>_<vectorizer>.pkl
results/final_model_results.csv
results/<name>_confusion_matrix.png
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pickle
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler

from ml.logistic_regression import LogisticRegressionModel
from ml.svm import SVMClassifier


from ml.TorchMultinomialNaiveBayes import TorchMultinomialNaiveBayes
from ml.TorchGaussianNaiveBayes import TorchGaussianNaiveBayes

from evaluation import (
    evaluate_model,
    save_confusion_matrix,
)


# ── Configuration dataclass ──────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    enable_tfidf: bool = False
    enable_distilbert: bool = True

    results_dir: Path = field(default_factory=lambda: Path("results"))

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    @property
    def models_dir(self) -> Path:
        return self.results_dir / "models"


# ── Cache Loading ─────────────────────────────────────────────────────────────

def load_cache(cache_dir: str) -> dict:
    """
    Load cached embeddings and labels.
    """

    base_dir = Path(__file__).resolve().parent

    cache_path = (
        (base_dir / cache_dir).resolve()
        if not Path(cache_dir).is_absolute()
        else Path(cache_dir)
    )

    cache = {
        "y_train": joblib.load(cache_path / "y_train.pkl"),
        "y_test": joblib.load(cache_path / "y_test.pkl"),
    }

    # TF-IDF cache
    tfidf_train = cache_path / "X_train_tfidf.npz"
    tfidf_test = cache_path / "X_test_tfidf.npz"

    if tfidf_train.exists() and tfidf_test.exists():
        cache["tfidf"] = (
            load_npz(str(tfidf_train)),
            load_npz(str(tfidf_test)),
        )

    # DistilBERT cache
    distilbert_train = cache_path / "X_train_distilbert.npy"
    distilbert_test = cache_path / "X_test_distilbert.npy"

    if distilbert_train.exists() and distilbert_test.exists():
        cache["distilbert"] = (
            np.load(str(distilbert_train)),
            np.load(str(distilbert_test)),
        )

    return cache


# ── Model Factory ─────────────────────────────────────────────────────────────

def make_models(
    vectorizer_name: str,
    config: PipelineConfig,
) -> Dict[str, object]:
    """
    Return models associated with a vectorizer.
    """

    models: Dict[str, object] = {
        #"LR": LogisticRegressionModel(),
        "SVM": SVMClassifier(),
    }
    
    # #TF-IDF pairings
    # if vectorizer_name == "TF-IDF":
    #     models["Multinomial NB"] = TorchMultinomialNaiveBayes(
    #         alpha=0.01,
    #         dtype=torch.float32,
    #         device=str(config.device),
    #     )

    # #DistilBERT pairings
    # if vectorizer_name == "DistilBERT":
    #     models["Gaussian NB"] = TorchGaussianNaiveBayes(
    #         var_smoothing=1e-9,
    #         dtype=torch.float32,
    #     )

    return models


# ── Label Encoding ────────────────────────────────────────────────────────────

def encode_labels(
    y,
    labels: List[str],
) -> np.ndarray:
    """
    Convert string labels into integer indices.
    """

    mapping = {label: idx for idx, label in enumerate(labels)}

    return np.array(
        [mapping[label] for label in y],
        dtype=np.int64,
    )


# ── Training Dispatcher ───────────────────────────────────────────────────────

def train_model(
    model,
    X_train,
    y_train,
    n_classes,
):
    """
    Dispatch training depending on model type.
    """
    if isinstance(model, TorchGaussianNaiveBayes):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        model.fit(X_tensor, y_tensor, n_classes=n_classes)

    elif isinstance(model, TorchMultinomialNaiveBayes):  # ← add this
        coo = X_train.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values  = torch.tensor(coo.data, dtype=torch.float32)
        X_tensor = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        model.fit(X_tensor, y_tensor, n_classes=n_classes)

    else:
        model.train(X_train, y_train)



# ── Torch Wrapper ─────────────────────────────────────────────────────────────

class TorchModelWrapper:
    """
    Converts Torch Gaussian NB predictions back into
    string labels for sklearn evaluation compatibility.
    """

    def __init__(self, torch_model, label_map):
        self.model = torch_model
        self.label_map = label_map

    def predict(self, X):
        # Multinomial NB expects a sparse tensor
        if isinstance(self.model, TorchMultinomialNaiveBayes):
            coo = X.tocoo()
            indices  = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
            values   = torch.tensor(coo.data, dtype=torch.float32)
            X_input  = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=torch.float32)
        else:
            X_input = torch.tensor(X, dtype=torch.float32)

        preds = self.model.predict(X_input).cpu().numpy()
        return np.array([self.label_map[p] for p in preds])


# ── Model Saving ──────────────────────────────────────────────────────────────

def save_model(
    model,
    model_name: str,
    vectorizer_name: str,
    config: PipelineConfig,
) -> Path:
    """
    Save trained model.
    """

    config.models_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{model_name}_{vectorizer_name}"
        .replace(" ", "_")
        .lower()
    )

    base = config.models_dir / stem

    # Torch models
    if hasattr(model, "save") and callable(model.save):
        path = base.with_suffix(".pt")
        model.save(path)

    # sklearn models
    else:
        path = base.with_suffix(".pkl")
        with open(path, "wb") as file:
            pickle.dump(model, file)

    print(f"Saved → {path}")
    return path


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_ml_pipeline(
    config: PipelineConfig,
    cache_dir: str = "cache",
    labels: Optional[List[str]] = None,
):
    """
    Run all configured experiments.
    """

    if labels is None:
        labels = ["Anxiety", "Depression", "Normal", "Suicidal"]

    config.results_dir.mkdir(exist_ok=True)

    cache = load_cache(cache_dir)

    y_train_raw = cache["y_train"]
    y_test_raw = cache["y_test"]

    y_train_int = encode_labels(y_train_raw, labels)
    y_test_int = encode_labels(y_test_raw, labels)

    vectorizer_sets = {}

    if config.enable_tfidf and "tfidf" in cache:
        vectorizer_sets["TF-IDF"] = cache["tfidf"]

    if config.enable_distilbert and "distilbert" in cache:
        X_train_db, X_test_db = cache["distilbert"]
        scaler = StandardScaler()
        vectorizer_sets["DistilBERT"] = (
            scaler.fit_transform(X_train_db),
            scaler.transform(X_test_db),
        )

    all_results = []

    for vec_name, (X_train, X_test) in vectorizer_sets.items():

        print("\n" + "=" * 60)
        print(f"Vectorizer: {vec_name}")
        print("=" * 60)

        models = make_models(vec_name, config)

        for model_name, model in models.items():

            print(f"\n--- {model_name} | {vec_name} ---")

            try:
                is_torch_nb = isinstance(model, (TorchGaussianNaiveBayes, TorchMultinomialNaiveBayes))

                # Torch NB requires integer labels
                y_tr = y_train_int if is_torch_nb else y_train_raw
                y_te = y_test_int if is_torch_nb else y_test_raw

                # Train
                train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_tr,
                    n_classes=len(labels),
                )

                # Save model
                save_model(
                    model=model,
                    model_name=model_name,
                    vectorizer_name=vec_name,
                    config=config,
                )

                # Evaluation wrapper for Torch models
                if is_torch_nb:
                    label_map = {i: label for i, label in enumerate(labels)}
                    eval_model = TorchModelWrapper(model, label_map)
                else:
                    eval_model = model

                stem = (
                    f"{model_name}_{vec_name}"
                    .replace(" ", "_")
                    .lower()
                )

                # Train evaluation
                train_results, y_train_pred = evaluate_model(
                    model=eval_model,
                    X=X_train,
                    y_true=y_train_raw,
                    labels=labels,
                    model_name=model_name,
                    vectorizer_name=vec_name,
                    split="Train",
                    save_csv=str(config.results_dir / f"{stem}_train.csv"),
                )

                # Test evaluation
                test_results, y_test_pred = evaluate_model(
                    model=eval_model,
                    X=X_test,
                    y_true=y_test_raw,
                    labels=labels,
                    model_name=model_name,
                    vectorizer_name=vec_name,
                    split="Test",
                    save_csv=str(config.results_dir / f"{stem}_test.csv"),
                )

                # Confusion matrix
                save_confusion_matrix(
                    y_true=y_test_raw,
                    y_pred=y_test_pred,
                    labels=labels,
                    model_name=model_name,
                    vectorizer_name=vec_name,
                    split="Test",
                    save_dir=str(config.results_dir),
                )

                all_results.append(train_results)
                all_results.append(test_results)

            except Exception as error:
                print(f"\n[ERROR] {model_name} | {vec_name} failed: {error}")
                import traceback
                traceback.print_exc()

    # Final CSV
    if all_results:

        summary_df = pd.concat(all_results, ignore_index=True)
        summary_path = config.results_dir / "final_model_results_tfidf.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\nFinal results saved.")
        print(summary_path)

        return summary_df

    print("No results generated.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models.")
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Embedding cache directory.",
    )
    parser.add_argument(
        "--tfidf",
        action="store_true",
        help="Enable TF-IDF models (LR, SVM, Multinomial NB).",
    )
    parser.add_argument(
        "--distilbert",
        action="store_true",
        help="Enable DistilBERT models (LR, SVM, Gaussian NB).",
    )
    args = parser.parse_args()

    # If neither flag is passed, default to both enabled
    tfidf_on     = args.tfidf     or (not args.tfidf and not args.distilbert)
    distilbert_on = args.distilbert or (not args.tfidf and not args.distilbert)

    cfg = PipelineConfig(
        enable_tfidf=tfidf_on,
        enable_distilbert=distilbert_on,
    )
    print(f"Using device     : {cfg.device}")
    print(f"TF-IDF enabled   : {cfg.enable_tfidf}")
    print(f"DistilBERT enabled: {cfg.enable_distilbert}")

    run_ml_pipeline(
        config=cfg,
        cache_dir=args.cache_dir,
    )
