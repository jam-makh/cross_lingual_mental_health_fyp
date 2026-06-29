"""AraBERT + PCA + Logistic Regression on Jean's final Arabic modeling dataset.

This script uses the team LogisticRegressionModel class and applies it
to the same final acceptable AraBERT pathway previously selected for Jean's data:

- Keep only:
    anxiety_fear, depression, ocd_obsessive
- Sublabel-aware downsampling to the Depression class size
- 70/15/15 stratified train/validation/test split
- Final accepted AraBERT vectorization path:
    * aubmindlab/bert-base-arabertv2
    * mean pooling
    * max_length=256
- Final accepted PCA pathway:
    * centered PCA
    * no whitening
    * 96 retained PCA dimensions
- Classifier:
    * LogisticRegressionModel from the team repository
    * Train on Train + Validation, evaluate once on Test

Expected folder:
LINEAR REgression/
├── final_modeling_dataset.csv
└── arabert_pca_logistic_regression_on_jean_dataset.py
"""

from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class AraBERTPCALogisticConfig:
    dataset_path: str = "final_modeling_dataset.csv"
    output_dir: str = "arabert_pca_logistic_regression_outputs"

    record_id_column: str = "record_id"
    text_column: str = "clean_text"
    label_column: str = "umbrella"
    sublabel_column: str = "label_clean"

    retained_labels: Tuple[str, ...] = (
        "anxiety_fear",
        "depression",
        "ocd_obsessive",
    )
    downsample_to_label: str = "depression"

    random_seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Same final accepted AraBERT vectorizer settings.
    pretrained_model_name: str = "aubmindlab/bert-base-arabertv2"
    max_length: int = 256
    batch_size: int = 16
    device: str = "auto"

    # Same final accepted PCA settings from Jean's best AraBERT experiment.
    pca_preprocessing_mode: str = "centered"
    pca_components: int = 96
    pca_whiten: bool = False
    standardization_epsilon: float = 1e-8
    whitening_epsilon: float = 1e-8
    l2_epsilon: float = 1e-12
    dtype: torch.dtype = torch.float32

    # Passed into LogisticRegressionModel.train().
    logistic_cv_splits: int = 5
    logistic_scoring: str = "f1_macro"


# =============================================================================
# 2. UTILITIES
# =============================================================================


def resolve_device(preference: str) -> torch.device:
    pref = str(preference).strip().lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if pref not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(pref)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3. DATA LOADING
# =============================================================================


class ModelingDatasetLoader:
    """Load and validate the final modeling dataset."""

    def __init__(self, config: AraBERTPCALogisticConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        path = Path(self.config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path.resolve()}\n"
                "Place final_modeling_dataset.csv next to this script."
            )

        df = pd.read_csv(path)

        required = {
            self.config.text_column,
            self.config.label_column,
            self.config.sublabel_column,
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        if self.config.record_id_column not in df.columns:
            df[self.config.record_id_column] = range(1, len(df) + 1)

        for col in [
            self.config.text_column,
            self.config.label_column,
            self.config.sublabel_column,
        ]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df[
            df[self.config.text_column].ne("")
            & df[self.config.label_column].ne("")
            & df[self.config.sublabel_column].ne("")
        ].copy()

        df = df[df[self.config.label_column].isin(self.config.retained_labels)].copy()
        if df.empty:
            raise ValueError("No rows remain after retaining the selected classes.")

        return df.reset_index(drop=True)


# =============================================================================
# 4. SUBLABEL-AWARE DOWNSAMPLING
# =============================================================================


class SublabelAwareDownsampler:
    """Downsample larger umbrellas while preserving sublabel proportions."""

    def __init__(self, config: AraBERTPCALogisticConfig) -> None:
        self.config = config

    def balance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        label_col = self.config.label_column
        sublabel_col = self.config.sublabel_column
        anchor_label = self.config.downsample_to_label

        umbrella_counts = df[label_col].value_counts()
        if anchor_label not in umbrella_counts:
            raise ValueError(f"Downsampling anchor label not found: {anchor_label}")

        target_count = int(umbrella_counts[anchor_label])
        rng = random.Random(self.config.random_seed)

        sampled_parts: List[pd.DataFrame] = []
        audit_rows: List[Dict[str, object]] = []

        for umbrella in self.config.retained_labels:
            group = df[df[label_col] == umbrella].copy()
            original_count = len(group)

            if original_count < target_count:
                raise ValueError(
                    f"Umbrella '{umbrella}' has {original_count} rows, below target {target_count}. "
                    "This script only downsamples."
                )

            sub_counts = group[sublabel_col].value_counts().sort_index()

            if original_count == target_count:
                sampled_parts.append(group.copy())
                for sublabel, sub_count in sub_counts.items():
                    audit_rows.append({
                        "umbrella": umbrella,
                        "sublabel": str(sublabel),
                        "original_sublabel_count": int(sub_count),
                        "sampled_sublabel_count": int(sub_count),
                        "original_within_umbrella_pct": round(sub_count / original_count * 100, 4),
                        "sampled_within_umbrella_pct": round(sub_count / target_count * 100, 4),
                        "sampling_mode": "kept_full_anchor_class",
                    })
                continue

            allocations: Dict[str, int] = {}
            fractional_parts: List[Tuple[str, float]] = []

            for sublabel, sub_count in sub_counts.items():
                sublabel_str = str(sublabel)
                ideal = (float(sub_count) / float(original_count)) * float(target_count)
                base = int(math.floor(ideal))
                allocations[sublabel_str] = min(base, int(sub_count))
                fractional_parts.append((sublabel_str, ideal - base))

            remaining = target_count - sum(allocations.values())
            fractional_parts.sort(key=lambda item: item[1], reverse=True)

            for sublabel_str, _ in fractional_parts:
                if remaining <= 0:
                    break
                capacity = int(sub_counts[sublabel_str]) - allocations[sublabel_str]
                if capacity > 0:
                    allocations[sublabel_str] += 1
                    remaining -= 1

            if remaining > 0:
                capacities = sorted(
                    [
                        (str(sublabel), int(sub_counts[sublabel]) - allocations[str(sublabel)])
                        for sublabel in sub_counts.index
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )
                for sublabel_str, capacity in capacities:
                    while remaining > 0 and capacity > 0:
                        allocations[sublabel_str] += 1
                        remaining -= 1
                        capacity -= 1
                    if remaining <= 0:
                        break

            if sum(allocations.values()) != target_count:
                raise RuntimeError(
                    f"Allocation mismatch for '{umbrella}'. "
                    f"Expected {target_count}, got {sum(allocations.values())}."
                )

            sampled_subparts: List[pd.DataFrame] = []

            for sublabel, original_sub_count in sub_counts.items():
                sublabel_str = str(sublabel)
                desired = allocations[sublabel_str]
                sub_df = group[group[sublabel_col] == sublabel].copy()

                if desired <= 0:
                    sampled_sub = sub_df.iloc[0:0].copy()
                elif desired == len(sub_df):
                    sampled_sub = sub_df.copy()
                else:
                    chosen_indices = rng.sample(sub_df.index.tolist(), desired)
                    sampled_sub = sub_df.loc[chosen_indices].copy()

                sampled_subparts.append(sampled_sub)
                audit_rows.append({
                    "umbrella": umbrella,
                    "sublabel": sublabel_str,
                    "original_sublabel_count": int(original_sub_count),
                    "sampled_sublabel_count": int(desired),
                    "original_within_umbrella_pct": round(original_sub_count / original_count * 100, 4),
                    "sampled_within_umbrella_pct": round(desired / target_count * 100, 4),
                    "sampling_mode": "proportional_sublabel_aware_downsampling",
                })

            sampled_parts.append(pd.concat(sampled_subparts, ignore_index=True))

        balanced_df = pd.concat(sampled_parts, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)
        audit_df = pd.DataFrame(audit_rows)
        return balanced_df, audit_df


# =============================================================================
# 5. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =============================================================================


class ThreeWayStratifiedSplitter:
    """Manual stratified 70/15/15 split by umbrella class."""

    def __init__(self, config: AraBERTPCALogisticConfig) -> None:
        self.config = config
        total = config.train_ratio + config.validation_ratio + config.test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        generator = torch.Generator().manual_seed(self.config.random_seed)
        label_col = self.config.label_column

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for label in self.config.retained_labels:
            indices = df[df[label_col] == label].index.tolist()
            if len(indices) < 3:
                raise ValueError(f"Class '{label}' has too few rows for a 3-way split.")

            perm = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in perm]

            n_total = len(shuffled)
            n_train = int(round(n_total * self.config.train_ratio))
            n_val = int(round(n_total * self.config.validation_ratio))
            n_test = n_total - n_train - n_val

            if n_val < 1:
                n_val = 1
                n_train -= 1
            if n_test < 1:
                n_test = 1
                n_train -= 1

            train_indices.extend(shuffled[:n_train])
            val_indices.extend(shuffled[n_train:n_train + n_val])
            test_indices.extend(shuffled[n_train + n_val:])

        rng = random.Random(self.config.random_seed)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        return (
            df.loc[train_indices].reset_index(drop=True),
            df.loc[val_indices].reset_index(drop=True),
            df.loc[test_indices].reset_index(drop=True),
        )


# =============================================================================
# 6. ARABERT EMBEDDING VECTORIZER — MEAN POOLING
# =============================================================================


class TextDataset(Dataset):
    """Minimal dataset for batched AraBERT inference."""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class AraBERTMeanEmbeddingVectorizer:
    """Frozen AraBERT mean-pooled embedding vectorizer."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype

        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print(f"Loading AraBERT encoder: {self.model_name}")
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _tokenize_batch(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def transform(self, texts: Sequence[str]) -> torch.Tensor:
        dataset = TextDataset(texts)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: batch)

        embeddings: List[torch.Tensor] = []
        total_batches = len(loader)

        with torch.no_grad():
            for batch_idx, batch_texts in enumerate(loader, start=1):
                encoded = self._tokenize_batch(batch_texts)
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                embeddings.append(pooled.detach().cpu().to(self.dtype))

                if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                    print(f"AraBERT vectorization batch {batch_idx:,}/{total_batches:,}")

        return torch.cat(embeddings, dim=0)


# =============================================================================
# 7. FINAL ACCEPTED PCA TRANSFORMER
# =============================================================================


class PCATransformerTorch:
    """PCA transformer matching Jean's accepted AraBERT/PCA setup."""

    def __init__(
        self,
        preprocessing_mode: str,
        n_components: int,
        whiten: bool,
        standardization_epsilon: float,
        whitening_epsilon: float,
        l2_epsilon: float,
        dtype: torch.dtype,
    ) -> None:
        if preprocessing_mode not in {"centered", "standardized", "l2_normalized"}:
            raise ValueError("Unsupported preprocessing_mode.")

        self.preprocessing_mode = preprocessing_mode
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.standardization_epsilon = float(standardization_epsilon)
        self.whitening_epsilon = float(whitening_epsilon)
        self.l2_epsilon = float(l2_epsilon)
        self.dtype = dtype

        self.center_: Optional[torch.Tensor] = None
        self.scale_: Optional[torch.Tensor] = None
        self.components_: Optional[torch.Tensor] = None
        self.singular_values_: Optional[torch.Tensor] = None
        self.explained_variance_: Optional[torch.Tensor] = None
        self.n_samples_fit_: Optional[int] = None

    def _preprocess_fit(self, X: torch.Tensor) -> torch.Tensor:
        X = X.detach().cpu().to(self.dtype)

        if self.preprocessing_mode == "centered":
            self.center_ = X.mean(dim=0)
            self.scale_ = torch.ones_like(self.center_)
            return X - self.center_

        if self.preprocessing_mode == "standardized":
            self.center_ = X.mean(dim=0)
            std = X.std(dim=0, unbiased=False)
            self.scale_ = torch.clamp(std, min=self.standardization_epsilon)
            return (X - self.center_) / self.scale_

        norms = torch.linalg.norm(X, dim=1, keepdim=True)
        X_norm = X / torch.clamp(norms, min=self.l2_epsilon)
        self.center_ = X_norm.mean(dim=0)
        self.scale_ = torch.ones_like(self.center_)
        return X_norm - self.center_

    def _preprocess_transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("PCA transformer has not been fitted.")

        X = X.detach().cpu().to(self.dtype)

        if self.preprocessing_mode == "centered":
            return X - self.center_

        if self.preprocessing_mode == "standardized":
            return (X - self.center_) / self.scale_

        norms = torch.linalg.norm(X, dim=1, keepdim=True)
        X_norm = X / torch.clamp(norms, min=self.l2_epsilon)
        return X_norm - self.center_

    def fit(self, X_train: torch.Tensor) -> "PCATransformerTorch":
        X_pre = self._preprocess_fit(X_train)
        n_samples, n_features = X_pre.shape
        max_components = min(n_samples, n_features)

        if self.n_components > max_components:
            raise ValueError(f"Requested {self.n_components} PCA components, but maximum is {max_components}.")

        _, singular_values, Vh = torch.linalg.svd(X_pre, full_matrices=False)
        self.components_ = Vh[: self.n_components, :].contiguous()
        self.singular_values_ = singular_values[: self.n_components].contiguous()
        self.n_samples_fit_ = int(n_samples)

        denom = max(n_samples - 1, 1)
        self.explained_variance_ = (self.singular_values_ ** 2) / float(denom)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.components_ is None or self.explained_variance_ is None:
            raise RuntimeError("PCA transformer has not been fitted.")

        X_pre = self._preprocess_transform(X)
        projected = X_pre @ self.components_.T

        if self.whiten:
            scale = torch.sqrt(torch.clamp(self.explained_variance_, min=self.whitening_epsilon))
            projected = projected / scale.unsqueeze(0)

        return projected.to(self.dtype)

    def fit_transform(self, X_train: torch.Tensor) -> torch.Tensor:
        self.fit(X_train)
        return self.transform(X_train)


# =============================================================================
# 8. JULIETTE'S SVM CLASS — UNCHANGED
# =============================================================================


# =============================================================================
# LOGISTIC REGRESSION MODEL CLASS — AS PROVIDED IN THE TEAM REPOSITORY
# =============================================================================


class LogisticRegressionModel:
    """
    Logistic Regression class that handles hyperparameter tuning,
    model fitting, and prediction.

    Receives pre-split, pre-vectorized arrays only.
    Splitting and vectorization are handled upstream.
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self.random_state = random_state
        self.test_size = test_size

        self.model: Optional[LogisticRegression] = None
        self.best_params_: Optional[Dict[str, object]] = None
        self.grid_search: Optional[GridSearchCV] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = "f1_macro",
        cv_splits: int = 5,
    ) -> None:
        """
        Train the Logistic Regression model using GridSearchCV.
        """
        base_model = LogisticRegression(
            max_iter=5000,
            random_state=self.random_state,
        )

        param_grid = [
            # lbfgs: l2 or None only
            {
                "solver": ["lbfgs"],
                "penalty": ["l2", None],
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
            # saga: all penalties including elasticnet
            {
                "solver": ["saga"],
                "penalty": ["l1", "l2", "elasticnet"],
                "C": [0.01, 0.1, 1.0, 10.0],
                "l1_ratio": [0.1, 0.5, 0.9],
                "class_weight": [None, "balanced"],
            },
            # liblinear: retained exactly as provided in the teammate model class
            {
                "solver": ["liblinear"],
                "penalty": ["l1", "l2"],
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
        ]

        stratified_cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=stratified_cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score=0.0,
        )

        print("Running GridSearchCV...")
        with tqdm(total=1, desc="Logistic Regression GridSearch") as pbar:
            grid_search.fit(X_train, y_train)
            pbar.update(1)

        # Store fitted GridSearchCV object, following the source model class.
        self.grid_search_ = grid_search

        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict labels using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.predict(X_test)


# =============================================================================
# EVALUATION HELPERS AROUND THE PROVIDED MODEL CLASS
# =============================================================================


def evaluate_predictions(
    model: LogisticRegressionModel,
    X_test,
    y_test,
    label_encoder: LabelEncoder,
    model_name: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Evaluate the trained LogisticRegressionModel using the same result table
    style used in the previous model experiments.
    """
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = report["macro avg"]["f1-score"]

    rows = []
    for i, class_name in enumerate(target_names):
        rows.append({
            "Model": model_name if i == 0 else "",
            "Class": class_name,
            "Accuracy": round(accuracy, 4) if i == 0 else "",
            "Precision": round(report[class_name]["precision"], 4),
            "Recall": round(report[class_name]["recall"], 4),
            "F1-score": round(report[class_name]["f1-score"], 4),
            "Support": int(report[class_name]["support"]),
            "Macro avg": round(macro_f1, 4) if i == 0 else "",
        })

    results_df = pd.DataFrame(rows)
    print(f"\n--- {model_name} Evaluation ---")
    print(results_df.to_string(index=False))

    return results_df, y_pred


def top_grid_results(model: LogisticRegressionModel, n: int = 10) -> pd.DataFrame:
    """
    Return top-n GridSearchCV rows sorted by rank.
    """
    grid_search = getattr(model, "grid_search_", None)
    if grid_search is None:
        raise RuntimeError("Grid search results are unavailable. Train the model first.")

    cv_df = pd.DataFrame(grid_search.cv_results_)
    preferred_cols = [
        "param_solver",
        "param_penalty",
        "param_C",
        "param_class_weight",
        "param_l1_ratio",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    cols = [col for col in preferred_cols if col in cv_df.columns]
    return (
        cv_df[cols]
        .sort_values("rank_test_score")
        .head(n)
        .reset_index(drop=True)
    )


def plot_confusion_matrix_for_model(
    y_test,
    y_pred,
    label_encoder: LabelEncoder,
    title: str,
) -> None:
    """
    Plot a heatmap confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 9. FULL EXPERIMENT
# =============================================================================


def run_experiment() -> None:
    config = AraBERTPCALogisticConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.random_seed)
    device = resolve_device(config.device)
    print(f"Device used for AraBERT vectorization: {device}")

    df = ModelingDatasetLoader(config).load()
    balanced_df, audit_df = SublabelAwareDownsampler(config).balance(df)
    train_df, val_df, test_df = ThreeWayStratifiedSplitter(config).split(balanced_df)
    train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Balanced rows used: {len(balanced_df):,}")
    print(f"Train/Validation/Test rows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"Final Logistic Regression training rows (Train + Validation): {len(train_plus_val_df):,}")

    vectorizer = AraBERTMeanEmbeddingVectorizer(
        model_name=config.pretrained_model_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
        device=device,
        dtype=config.dtype,
    )

    print("\nVectorizing Train + Validation split with AraBERT...")
    X_train_plus_val_raw = vectorizer.transform(train_plus_val_df[config.text_column].tolist())
    print("\nVectorizing Test split with AraBERT...")
    X_test_raw = vectorizer.transform(test_df[config.text_column].tolist())

    del vectorizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pca = PCATransformerTorch(
        preprocessing_mode=config.pca_preprocessing_mode,
        n_components=config.pca_components,
        whiten=config.pca_whiten,
        standardization_epsilon=config.standardization_epsilon,
        whitening_epsilon=config.whitening_epsilon,
        l2_epsilon=config.l2_epsilon,
        dtype=config.dtype,
    )

    X_train = pca.fit_transform(X_train_plus_val_raw).numpy()
    X_test = pca.transform(X_test_raw).numpy()

    print(f"PCA training matrix shape: {X_train.shape}")
    print(f"PCA test matrix shape: {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_plus_val_df[config.label_column].tolist())
    y_test = label_encoder.transform(test_df[config.label_column].tolist())

    logistic_model = LogisticRegressionModel(
        random_state=config.random_seed,
    )
    logistic_model.train(
        X_train,
        y_train,
        scoring=config.logistic_scoring,
        cv_splits=config.logistic_cv_splits,
    )

    print(f"\nBest params  : {logistic_model.best_params_}")

    results_df, y_pred = evaluate_predictions(
        logistic_model,
        X_test,
        y_test,
        label_encoder=label_encoder,
        model_name="AraBERT Mean + PCA96 + Logistic Regression",
    )

    results_df.to_csv(output_dir / "final_test_results.csv", index=False, encoding="utf-8-sig")
    top_grid_results(logistic_model, 10).to_csv(output_dir / "top_grid_results.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(output_dir / "sublabel_aware_sampling_audit.csv", index=False, encoding="utf-8-sig")

    predictions_df = test_df[
        [
            config.record_id_column,
            config.text_column,
            config.label_column,
            config.sublabel_column,
        ]
    ].copy()
    predictions_df.rename(columns={config.label_column: "actual_umbrella"}, inplace=True)
    predictions_df["predicted_umbrella"] = label_encoder.inverse_transform(y_pred)
    predictions_df["is_correct"] = predictions_df["actual_umbrella"] == predictions_df["predicted_umbrella"]
    predictions_df.to_csv(output_dir / "final_test_predictions.csv", index=False, encoding="utf-8-sig")

    (output_dir / "best_svm_params.json").write_text(
        json.dumps(logistic_model.best_params_, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "AraBERT + PCA + Logistic Regression Summary",
        "===========================================",
        "",
        "Vectorizer:",
        "- Final accepted AraBERT vectorization path from Jean's best AraBERT experiment",
        f"- Encoder: {config.pretrained_model_name}",
        "- Pooling: mean pooling",
        f"- Max sequence length: {config.max_length}",
        "",
        "PCA:",
        f"- Preprocessing: {config.pca_preprocessing_mode}",
        f"- Whitening: {config.pca_whiten}",
        f"- Retained components: {config.pca_components}",
        "",
        "Data:",
        f"- Balanced rows: {len(balanced_df):,}",
        f"- Train: {len(train_df):,}",
        f"- Validation: {len(val_df):,}",
        f"- Test: {len(test_df):,}",
        f"- Final Logistic Regression training rows (Train + Validation): {len(train_plus_val_df):,}",
        "",
        "Classifier:",
        "- LogisticRegressionModel from the team repository",
        f"- Best params: {logistic_model.best_params_}",
        "",
        "Main outputs:",
        "- final_test_results.csv",
        "- final_test_predictions.csv",
        "- top_grid_results.csv",
        "- best_svm_params.json",
        "- sublabel_aware_sampling_audit.csv",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nDone. Outputs saved under: {output_dir.resolve()}")
    plot_confusion_matrix_for_model(
        y_test,
        y_pred,
        label_encoder=label_encoder,
        title="Confusion Matrix — AraBERT Mean + PCA96 + Logistic Regression",
    )


if __name__ == "__main__":
    run_experiment()
