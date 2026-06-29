"""Balanced 3-class TF-IDF + Multinomial Naive Bayes experiment in PyTorch.

This script is the improved retraining version of the earlier TF-IDF baseline.
It follows the decision made after reviewing the first results:

1) Keep only the three main umbrellas:
      - anxiety_fear
      - ocd_obsessive
      - depression

2) Exclude:
      - bipolar_mania
      - other_unclear

3) Downsample the two larger retained classes to the size of Depression.
   This produces a balanced dataset without oversampling Depression.

   Expected class sizes from the current dataset:
      anxiety_fear  -> downsampled to depression size
      ocd_obsessive -> downsampled to depression size
      depression    -> kept as is

4) Use a train / validation / test split:
      - Train is used to fit TF-IDF + Naive Bayes.
      - Validation is used to select the best hyperparameter configuration.
      - Test is used only once for final unbiased evaluation.

5) Tune the TF-IDF + MNB setup across:
      - ngram_range
      - min_df
      - max_features
      - alpha smoothing

6) Select the best configuration primarily by Validation Macro F1-score.
   Accuracy is used only as a secondary tie-breaker.

No scikit-learn is used. The vectorizer, classifier, split logic, and metrics are
implemented with standard Python + pandas + PyTorch.

Expected folder layout:
Machine Learning/
├── final_modeling_dataset.csv
└── tfidf_multinomial_nb_balanced_tuned.py

Outputs:
tfidf_nb_balanced_tuned_outputs/
├── balanced_dataset_class_summary.csv
├── balanced_dataset_sampled_records.csv
├── split_summary.csv
├── hyperparameter_search_results.csv
├── best_validation_results.csv
├── best_validation_macro_summary.csv
├── final_test_results.csv
├── final_test_macro_summary.csv
├── final_test_confusion_matrix.csv
├── final_test_predictions.csv
├── train_plus_validation_class_summary.csv
├── final_vectorizer_state.pt
├── final_multinomial_nb_model.pt
├── best_config.json
└── run_summary.txt
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class BalancedTunedConfig:
    """Configuration for balanced 3-class TF-IDF + Multinomial NB retraining."""

    # Input / output
    dataset_path: str = "final_modeling_dataset.csv"
    output_dir: str = "tfidf_nb_balanced_tuned_outputs"

    # Columns in the exported modeling dataset
    text_column: str = "analysis_text"
    label_column: str = "umbrella"
    record_id_column: str = "record_id"

    # Final 3 target classes
    retained_labels: Tuple[str, ...] = (
        "anxiety_fear",
        "ocd_obsessive",
        "depression",
    )

    # Downsampling target anchor
    downsample_to_label: str = "depression"

    # Reproducibility
    random_seed: int = 42

    # Split ratios after balancing
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Hyperparameter search grid
    ngram_ranges: Tuple[Tuple[int, int], ...] = (
        (1, 1),
        (1, 2),
    )
    min_df_values: Tuple[int, ...] = (1, 2, 3)
    max_features_values: Tuple[Optional[int], ...] = (30000, 50000, 80000)
    alpha_values: Tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 2.0)

    # TF-IDF behavior
    max_df_ratio: float = 0.95
    l2_normalize: bool = True

    # PyTorch setup
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


# =============================================================================
# 2. DATA LOADING + BALANCING
# =============================================================================


class ModelingDatasetLoader:
    """Load the final modeling CSV and keep only valid rows."""

    def __init__(self, config: BalancedTunedConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        path = Path(self.config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path.resolve()}\n"
                "Place final_modeling_dataset.csv next to this script, "
                "or update BalancedTunedConfig.dataset_path."
            )

        df = pd.read_csv(path)

        required = {self.config.text_column, self.config.label_column}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        if self.config.record_id_column not in df.columns:
            df[self.config.record_id_column] = range(1, len(df) + 1)

        df[self.config.text_column] = df[self.config.text_column].fillna("").astype(str).str.strip()
        df[self.config.label_column] = df[self.config.label_column].fillna("").astype(str).str.strip()

        df = df[
            df[self.config.text_column].ne("")
            & df[self.config.label_column].ne("")
        ].copy()

        # Keep only the three selected classes.
        df = df[df[self.config.label_column].isin(self.config.retained_labels)].copy()

        if df.empty:
            raise ValueError("No rows remain after retaining the three target classes.")

        return df.reset_index(drop=True)


class ThreeClassDownsampler:
    """Downsample larger classes to the target count of the chosen anchor class."""

    def __init__(self, config: BalancedTunedConfig) -> None:
        self.config = config

    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        label_col = self.config.label_column
        target_label = self.config.downsample_to_label

        counts = df[label_col].value_counts()
        if target_label not in counts:
            raise ValueError(f"Downsampling anchor label not found: {target_label}")

        target_count = int(counts[target_label])
        if target_count <= 0:
            raise ValueError("Downsampling target count must be positive.")

        rng = random.Random(self.config.random_seed)
        balanced_parts: List[pd.DataFrame] = []

        for label in self.config.retained_labels:
            class_df = df[df[label_col] == label].copy()
            class_count = len(class_df)

            if class_count < target_count:
                raise ValueError(
                    f"Class '{label}' has {class_count} rows, which is smaller than "
                    f"the downsampling target {target_count}. This script only downsamples; "
                    "it never oversamples."
                )

            if class_count == target_count:
                sampled_df = class_df.copy()
            else:
                sampled_indices = rng.sample(class_df.index.tolist(), target_count)
                sampled_df = class_df.loc[sampled_indices].copy()

            balanced_parts.append(sampled_df)

        balanced_df = pd.concat(balanced_parts, ignore_index=True)

        # Shuffle final balanced dataset deterministically.
        balanced_df = balanced_df.sample(
            frac=1.0,
            random_state=self.config.random_seed,
        ).reset_index(drop=True)

        return balanced_df

    def export_summary(self, balanced_df: pd.DataFrame, output_dir: Path) -> None:
        label_col = self.config.label_column
        summary = balanced_df[label_col].value_counts().reset_index()
        summary.columns = ["umbrella", "row_count"]
        summary["percentage"] = (summary["row_count"] / len(balanced_df) * 100).round(2)
        summary.to_csv(
            output_dir / "balanced_dataset_class_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        sample_cols = [
            col for col in [
                self.config.record_id_column,
                self.config.text_column,
                self.config.label_column,
            ]
            if col in balanced_df.columns
        ]
        balanced_df[sample_cols].to_csv(
            output_dir / "balanced_dataset_sampled_records.csv",
            index=False,
            encoding="utf-8-sig",
        )


# =============================================================================
# 3. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =============================================================================


class ThreeWayStratifiedSplitter:
    """Manual stratified train/validation/test splitter."""

    def __init__(self, config: BalancedTunedConfig) -> None:
        self.config = config
        total = config.train_ratio + config.validation_ratio + config.test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("train_ratio + validation_ratio + test_ratio must equal 1.0")

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        label_col = self.config.label_column
        generator = torch.Generator().manual_seed(self.config.random_seed)

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for label in self.config.retained_labels:
            class_indices = df[df[label_col] == label].index.tolist()
            if len(class_indices) < 3:
                raise ValueError(f"Class '{label}' has too few rows for a 3-way split.")

            permutation = torch.randperm(len(class_indices), generator=generator).tolist()
            shuffled = [class_indices[i] for i in permutation]

            total_n = len(shuffled)
            n_train = int(round(total_n * self.config.train_ratio))
            n_val = int(round(total_n * self.config.validation_ratio))
            n_test = total_n - n_train - n_val

            # Safety: keep at least one row in validation and test.
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

        train_df = df.loc[train_indices].reset_index(drop=True)
        val_df = df.loc[val_indices].reset_index(drop=True)
        test_df = df.loc[test_indices].reset_index(drop=True)

        return train_df, val_df, test_df

    def export_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        label_col = self.config.label_column
        train_counts = train_df[label_col].value_counts().rename("train_count")
        val_counts = val_df[label_col].value_counts().rename("validation_count")
        test_counts = test_df[label_col].value_counts().rename("test_count")

        summary = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int).reset_index()
        summary.rename(columns={summary.columns[0]: "umbrella"}, inplace=True)
        summary["total_count"] = (
            summary["train_count"] + summary["validation_count"] + summary["test_count"]
        )
        summary["train_ratio"] = (summary["train_count"] / summary["total_count"]).round(4)
        summary["validation_ratio"] = (summary["validation_count"] / summary["total_count"]).round(4)
        summary["test_ratio"] = (summary["test_count"] / summary["total_count"]).round(4)
        summary = summary.sort_values("umbrella").reset_index(drop=True)
        summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# 4. LABEL ENCODER
# =============================================================================


class LabelEncoderTorch:
    """String label <-> integer class ID encoder."""

    def __init__(self) -> None:
        self.classes_: List[str] = []
        self.class_to_index_: Dict[str, int] = {}

    def fit(self, labels: Sequence[str]) -> "LabelEncoderTorch":
        self.classes_ = sorted({str(label) for label in labels})
        if not self.classes_:
            raise ValueError("Cannot fit label encoder on empty labels.")
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, labels: Sequence[str]) -> torch.Tensor:
        if not self.class_to_index_:
            raise RuntimeError("Label encoder has not been fitted.")
        encoded: List[int] = []
        for label in labels:
            label_str = str(label)
            if label_str not in self.class_to_index_:
                raise ValueError(f"Unknown label encountered: {label_str}")
            encoded.append(self.class_to_index_[label_str])
        return torch.tensor(encoded, dtype=torch.long)

    def inverse_transform(self, indices: Sequence[int]) -> List[str]:
        if not self.classes_:
            raise RuntimeError("Label encoder has not been fitted.")
        return [self.classes_[int(idx)] for idx in indices]


# =============================================================================
# 5. MANUAL TF-IDF VECTORIZER
# =============================================================================


class TorchTfidfVectorizer:
    """Manual TF-IDF vectorizer returning sparse PyTorch COO tensors."""

    def __init__(
        self,
        ngram_range: Tuple[int, int],
        min_df: int,
        max_df_ratio: float,
        max_features: Optional[int],
        l2_normalize: bool,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError("Invalid ngram_range.")
        if min_df < 1:
            raise ValueError("min_df must be >= 1")
        if not 0 < max_df_ratio <= 1:
            raise ValueError("max_df_ratio must be in (0, 1]")

        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_features = max_features
        self.l2_normalize = l2_normalize
        self.dtype = dtype
        self.device = device

        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[torch.Tensor] = None
        self.n_documents_: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in str(text).split() if tok]

    def _generate_ngrams(self, tokens: Sequence[str]) -> List[str]:
        terms: List[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                terms.append(" ".join(tokens[i:i + n]))
        return terms

    def fit(self, texts: Sequence[str]) -> "TorchTfidfVectorizer":
        self.n_documents_ = len(texts)
        if self.n_documents_ == 0:
            raise ValueError("Cannot fit TF-IDF on zero texts.")

        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            terms = self._generate_ngrams(tokens)
            term_frequency.update(terms)
            document_frequency.update(set(terms))

        max_df_count = max(1, int(math.floor(self.max_df_ratio * self.n_documents_)))
        eligible_terms = [
            term
            for term, df in document_frequency.items()
            if df >= self.min_df and df <= max_df_count
        ]
        if not eligible_terms:
            raise ValueError("No features survived min_df/max_df filtering.")

        eligible_terms.sort(key=lambda term: (-term_frequency[term], term))
        if self.max_features is not None:
            eligible_terms = eligible_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(eligible_terms)}
        idf_values = [
            math.log((1.0 + self.n_documents_) / (1.0 + document_frequency[term])) + 1.0
            for term in eligible_terms
        ]
        self.idf_ = torch.tensor(idf_values, dtype=self.dtype, device=self.device)
        return self

    def fit_transform(self, texts: Sequence[str]) -> torch.Tensor:
        self.fit(texts)
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> torch.Tensor:
        if not self.vocabulary_ or self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted.")

        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[float] = []

        for row_id, text in enumerate(texts):
            tokens = self._tokenize(text)
            terms = self._generate_ngrams(tokens)
            counts = Counter(term for term in terms if term in self.vocabulary_)
            if not counts:
                continue

            row_features: List[Tuple[int, float]] = []
            for term, count in counts.items():
                col = self.vocabulary_[term]
                value = float(count) * float(self.idf_[col].item())
                row_features.append((col, value))

            if self.l2_normalize:
                norm = math.sqrt(sum(value * value for _, value in row_features))
                if norm > 0:
                    row_features = [(col, value / norm) for col, value in row_features]

            for col, value in row_features:
                row_indices.append(row_id)
                col_indices.append(col)
                values.append(value)

        n_rows = len(texts)
        n_cols = len(self.vocabulary_)
        if values:
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=self.device)
            tensor_values = torch.tensor(values, dtype=self.dtype, device=self.device)
        else:
            indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            tensor_values = torch.empty((0,), dtype=self.dtype, device=self.device)

        return torch.sparse_coo_tensor(
            indices,
            tensor_values,
            size=(n_rows, n_cols),
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def save(self, path: Path) -> None:
        if self.idf_ is None:
            raise RuntimeError("Cannot save an unfitted vectorizer.")
        torch.save(
            {
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df_ratio": self.max_df_ratio,
                "max_features": self.max_features,
                "l2_normalize": self.l2_normalize,
                "vocabulary": self.vocabulary_,
                "idf": self.idf_.detach().cpu(),
                "n_documents": self.n_documents_,
            },
            path,
        )


# =============================================================================
# 6. MULTINOMIAL NAIVE BAYES IN PYTORCH
# =============================================================================


class TorchMultinomialNaiveBayes:
    """Multinomial NB fitted on sparse TF-IDF features."""

    def __init__(self, alpha: float, dtype: torch.dtype, device: str) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)
        self.dtype = dtype
        self.device = device

        self.class_log_prior_: Optional[torch.Tensor] = None
        self.feature_log_prob_: Optional[torch.Tensor] = None
        self.class_count_: Optional[torch.Tensor] = None
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None

    def fit(self, X_sparse: torch.Tensor, y: torch.Tensor, n_classes: int) -> "TorchMultinomialNaiveBayes":
        if not X_sparse.is_sparse:
            raise ValueError("X_sparse must be a sparse COO tensor.")
        if X_sparse.size(0) != y.numel():
            raise ValueError("X row count and y length do not match.")

        X_sparse = X_sparse.coalesce().to(self.device)
        y = y.to(self.device)
        self.n_classes_ = int(n_classes)
        self.n_features_ = int(X_sparse.size(1))

        class_count = torch.bincount(y, minlength=self.n_classes_).to(dtype=self.dtype)
        if torch.any(class_count == 0):
            raise ValueError("At least one class has zero training samples.")

        feature_count = torch.zeros(
            (self.n_classes_, self.n_features_),
            dtype=self.dtype,
            device=self.device,
        )

        sparse_indices = X_sparse.indices()
        doc_indices = sparse_indices[0]
        feature_indices = sparse_indices[1]
        feature_values = X_sparse.values().to(dtype=self.dtype)
        feature_class_indices = y[doc_indices]

        flat_indices = feature_class_indices * self.n_features_ + feature_indices
        feature_count.view(-1).scatter_add_(0, flat_indices, feature_values)

        smoothed_feature_count = feature_count + self.alpha
        smoothed_totals = smoothed_feature_count.sum(dim=1, keepdim=True)

        self.feature_log_prob_ = torch.log(smoothed_feature_count) - torch.log(smoothed_totals)
        self.class_log_prior_ = torch.log(class_count) - torch.log(class_count.sum())
        self.class_count_ = class_count
        return self

    def predict(self, X_sparse: torch.Tensor) -> torch.Tensor:
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_sparse = X_sparse.coalesce().to(self.device)
        scores = torch.sparse.mm(X_sparse, self.feature_log_prob_.T)
        scores = scores + self.class_log_prior_.unsqueeze(0)
        return torch.argmax(scores, dim=1)

    def save(self, path: Path) -> None:
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Cannot save an unfitted model.")
        torch.save(
            {
                "alpha": self.alpha,
                "class_log_prior": self.class_log_prior_.detach().cpu(),
                "feature_log_prob": self.feature_log_prob_.detach().cpu(),
                "class_count": self.class_count_.detach().cpu() if self.class_count_ is not None else None,
                "n_classes": self.n_classes_,
                "n_features": self.n_features_,
            },
            path,
        )


# =============================================================================
# 7. METRICS
# =============================================================================


class ClassificationEvaluator:
    """Manual metrics: accuracy, precision, recall, F1, macro averages, confusion matrix."""

    def __init__(self, class_labels: Sequence[str]) -> None:
        self.class_labels = list(class_labels)
        self.n_classes = len(self.class_labels)

    def confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.detach().cpu().long()
        y_pred = y_pred.detach().cpu().long()
        if y_true.numel() != y_pred.numel():
            raise ValueError("y_true and y_pred sizes do not match.")

        cm = torch.zeros((self.n_classes, self.n_classes), dtype=torch.long)
        flat = y_true * self.n_classes + y_pred
        cm.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        return cm

    def evaluate(
        self,
        model_name: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cm = self.confusion_matrix(y_true, y_pred)
        total = int(cm.sum().item())
        correct = int(torch.diag(cm).sum().item())
        accuracy = correct / total if total else 0.0

        precision_vals: List[float] = []
        recall_vals: List[float] = []
        f1_vals: List[float] = []
        supports: List[int] = []

        for class_idx in range(self.n_classes):
            tp = float(cm[class_idx, class_idx].item())
            predicted_total = float(cm[:, class_idx].sum().item())
            actual_total = float(cm[class_idx, :].sum().item())

            precision = tp / predicted_total if predicted_total > 0 else 0.0
            recall = tp / actual_total if actual_total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

            precision_vals.append(precision)
            recall_vals.append(recall)
            f1_vals.append(f1)
            supports.append(int(actual_total))

        macro_precision = sum(precision_vals) / self.n_classes if self.n_classes else 0.0
        macro_recall = sum(recall_vals) / self.n_classes if self.n_classes else 0.0
        macro_f1 = sum(f1_vals) / self.n_classes if self.n_classes else 0.0

        rows: List[Dict[str, object]] = []
        for idx, label in enumerate(self.class_labels):
            rows.append(
                {
                    "MODEL": model_name,
                    "Class": label,
                    "Accuracy": round(accuracy, 4),
                    "Precision": round(precision_vals[idx], 4),
                    "Recall": round(recall_vals[idx], 4),
                    "F1-score": round(f1_vals[idx], 4),
                    "Support": supports[idx],
                    "Macro Avg": round(macro_f1, 4),
                }
            )

        per_class_df = pd.DataFrame(rows)
        macro_df = pd.DataFrame(
            [
                {
                    "MODEL": model_name,
                    "Accuracy": round(accuracy, 4),
                    "Macro Precision": round(macro_precision, 4),
                    "Macro Recall": round(macro_recall, 4),
                    "Macro F1-score": round(macro_f1, 4),
                    "Total Support": total,
                }
            ]
        )
        cm_df = pd.DataFrame(
            cm.numpy(),
            index=[f"actual::{label}" for label in self.class_labels],
            columns=[f"predicted::{label}" for label in self.class_labels],
        )
        return per_class_df, macro_df, cm_df


# =============================================================================
# 8. SEARCH RESULT CONTAINER
# =============================================================================


@dataclass
class SearchRunResult:
    """One validation run result for the tuning table."""

    run_id: int
    ngram_min: int
    ngram_max: int
    min_df: int
    max_features: Optional[int]
    alpha: float
    retained_features: int
    validation_accuracy: float
    validation_macro_precision: float
    validation_macro_recall: float
    validation_macro_f1: float


# =============================================================================
# 9. MAIN EXPERIMENT PIPELINE
# =============================================================================


class BalancedTunedTFIDFMultinomialNBExperiment:
    """Balance, split, tune, retrain, and evaluate the improved TF-IDF + MNB pipeline."""

    def __init__(self, config: BalancedTunedConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        torch.manual_seed(self.config.random_seed)
        random.seed(self.config.random_seed)

        # ---------------------------------------------------------------------
        # A. Load and balance data
        # ---------------------------------------------------------------------
        loader = ModelingDatasetLoader(self.config)
        filtered_df = loader.load()

        downsampler = ThreeClassDownsampler(self.config)
        balanced_df = downsampler.balance(filtered_df)
        downsampler.export_summary(balanced_df, self.output_dir)

        # ---------------------------------------------------------------------
        # B. Split balanced dataset into train/validation/test
        # ---------------------------------------------------------------------
        splitter = ThreeWayStratifiedSplitter(self.config)
        train_df, val_df, test_df = splitter.split(balanced_df)
        splitter.export_summary(train_df, val_df, test_df, self.output_dir)

        # ---------------------------------------------------------------------
        # C. Label encoding fixed from training classes
        # ---------------------------------------------------------------------
        label_encoder = LabelEncoderTorch().fit(train_df[self.config.label_column].tolist())
        y_train = label_encoder.transform(train_df[self.config.label_column].tolist())
        y_val = label_encoder.transform(val_df[self.config.label_column].tolist())
        y_test = label_encoder.transform(test_df[self.config.label_column].tolist())

        # ---------------------------------------------------------------------
        # D. Hyperparameter search on validation set
        # ---------------------------------------------------------------------
        search_results: List[SearchRunResult] = []
        best_payload: Optional[Dict[str, object]] = None
        best_validation_macro_f1 = -1.0
        best_validation_accuracy = -1.0
        run_id = 0

        total_runs = (
            len(self.config.ngram_ranges)
            * len(self.config.min_df_values)
            * len(self.config.max_features_values)
            * len(self.config.alpha_values)
        )

        print(f"Balanced 3-class dataset rows: {len(balanced_df):,}")
        print(f"Train/Validation/Test rows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
        print(f"Hyperparameter runs to evaluate: {total_runs:,}")

        for ngram_range in self.config.ngram_ranges:
            for min_df in self.config.min_df_values:
                for max_features in self.config.max_features_values:
                    # Fit a vectorizer once for this TF-IDF setup.
                    vectorizer = TorchTfidfVectorizer(
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df_ratio=self.config.max_df_ratio,
                        max_features=max_features,
                        l2_normalize=self.config.l2_normalize,
                        dtype=self.config.dtype,
                        device=self.config.device,
                    )
                    X_train = vectorizer.fit_transform(train_df[self.config.text_column].tolist())
                    X_val = vectorizer.transform(val_df[self.config.text_column].tolist())

                    for alpha in self.config.alpha_values:
                        run_id += 1
                        model = TorchMultinomialNaiveBayes(
                            alpha=alpha,
                            dtype=self.config.dtype,
                            device=self.config.device,
                        )
                        model.fit(X_train, y_train, n_classes=len(label_encoder.classes_))
                        y_val_pred = model.predict(X_val)

                        evaluator = ClassificationEvaluator(label_encoder.classes_)
                        _, validation_macro_df, _ = evaluator.evaluate(
                            model_name="Validation TF-IDF + Multinomial NB",
                            y_true=y_val,
                            y_pred=y_val_pred,
                        )
                        row = validation_macro_df.iloc[0]

                        result = SearchRunResult(
                            run_id=run_id,
                            ngram_min=ngram_range[0],
                            ngram_max=ngram_range[1],
                            min_df=min_df,
                            max_features=max_features,
                            alpha=alpha,
                            retained_features=len(vectorizer.vocabulary_),
                            validation_accuracy=float(row["Accuracy"]),
                            validation_macro_precision=float(row["Macro Precision"]),
                            validation_macro_recall=float(row["Macro Recall"]),
                            validation_macro_f1=float(row["Macro F1-score"]),
                        )
                        search_results.append(result)

                        candidate_f1 = result.validation_macro_f1
                        candidate_acc = result.validation_accuracy
                        is_better = (
                            candidate_f1 > best_validation_macro_f1
                            or (
                                math.isclose(candidate_f1, best_validation_macro_f1)
                                and candidate_acc > best_validation_accuracy
                            )
                        )
                        if is_better:
                            best_validation_macro_f1 = candidate_f1
                            best_validation_accuracy = candidate_acc
                            best_payload = {
                                "ngram_range": ngram_range,
                                "min_df": min_df,
                                "max_features": max_features,
                                "alpha": alpha,
                                "retained_features_validation_fit": len(vectorizer.vocabulary_),
                                "validation_macro_f1": candidate_f1,
                                "validation_accuracy": candidate_acc,
                            }

                        if run_id == 1 or run_id % 10 == 0 or run_id == total_runs:
                            print(
                                f"Run {run_id:>3}/{total_runs}: "
                                f"ngrams={ngram_range}, min_df={min_df}, max_features={max_features}, "
                                f"alpha={alpha} -> val Macro F1={candidate_f1:.4f}, val Acc={candidate_acc:.4f}"
                            )

        if best_payload is None:
            raise RuntimeError("Hyperparameter search failed to produce a best configuration.")

        search_df = pd.DataFrame([asdict(item) for item in search_results])
        search_df = search_df.sort_values(
            ["validation_macro_f1", "validation_accuracy"],
            ascending=[False, False],
        ).reset_index(drop=True)
        search_df.to_csv(
            self.output_dir / "hyperparameter_search_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # ---------------------------------------------------------------------
        # E. Evaluate the selected validation configuration in detail
        # ---------------------------------------------------------------------
        best_ngram = tuple(best_payload["ngram_range"])  # type: ignore[arg-type]
        best_min_df = int(best_payload["min_df"])
        best_max_features = best_payload["max_features"]
        best_alpha = float(best_payload["alpha"])

        validation_vectorizer = TorchTfidfVectorizer(
            ngram_range=best_ngram,
            min_df=best_min_df,
            max_df_ratio=self.config.max_df_ratio,
            max_features=best_max_features if best_max_features is None else int(best_max_features),
            l2_normalize=self.config.l2_normalize,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        X_train_best = validation_vectorizer.fit_transform(train_df[self.config.text_column].tolist())
        X_val_best = validation_vectorizer.transform(val_df[self.config.text_column].tolist())
        validation_model = TorchMultinomialNaiveBayes(
            alpha=best_alpha,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        validation_model.fit(X_train_best, y_train, n_classes=len(label_encoder.classes_))
        y_val_best_pred = validation_model.predict(X_val_best)

        evaluator = ClassificationEvaluator(label_encoder.classes_)
        validation_results_df, validation_macro_df, _ = evaluator.evaluate(
            model_name="Best Validation TF-IDF + Multinomial NB",
            y_true=y_val,
            y_pred=y_val_best_pred,
        )
        validation_results_df.to_csv(
            self.output_dir / "best_validation_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        validation_macro_df.to_csv(
            self.output_dir / "best_validation_macro_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # ---------------------------------------------------------------------
        # F. Retrain best configuration on Train + Validation, then evaluate Test once
        # ---------------------------------------------------------------------
        train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True)
        y_train_plus_val = label_encoder.transform(train_plus_val_df[self.config.label_column].tolist())

        final_vectorizer = TorchTfidfVectorizer(
            ngram_range=best_ngram,
            min_df=best_min_df,
            max_df_ratio=self.config.max_df_ratio,
            max_features=best_max_features if best_max_features is None else int(best_max_features),
            l2_normalize=self.config.l2_normalize,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        X_train_plus_val = final_vectorizer.fit_transform(train_plus_val_df[self.config.text_column].tolist())
        X_test = final_vectorizer.transform(test_df[self.config.text_column].tolist())

        final_model = TorchMultinomialNaiveBayes(
            alpha=best_alpha,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        final_model.fit(X_train_plus_val, y_train_plus_val, n_classes=len(label_encoder.classes_))
        y_test_pred = final_model.predict(X_test)

        final_test_results_df, final_test_macro_df, final_test_cm_df = evaluator.evaluate(
            model_name="Balanced Tuned TF-IDF + Multinomial NB",
            y_true=y_test,
            y_pred=y_test_pred,
        )

        final_test_results_df.to_csv(
            self.output_dir / "final_test_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        final_test_macro_df.to_csv(
            self.output_dir / "final_test_macro_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        final_test_cm_df.to_csv(
            self.output_dir / "final_test_confusion_matrix.csv",
            encoding="utf-8-sig",
        )

        # Final predictions for detailed error inspection.
        predicted_labels = label_encoder.inverse_transform(y_test_pred.detach().cpu().tolist())
        prediction_cols = [
            col for col in [
                self.config.record_id_column,
                self.config.text_column,
                self.config.label_column,
            ]
            if col in test_df.columns
        ]
        final_predictions = test_df[prediction_cols].copy()
        final_predictions.rename(columns={self.config.label_column: "actual_umbrella"}, inplace=True)
        final_predictions["predicted_umbrella"] = predicted_labels
        final_predictions["is_correct"] = (
            final_predictions["actual_umbrella"] == final_predictions["predicted_umbrella"]
        )
        final_predictions.to_csv(
            self.output_dir / "final_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Train+validation class summary.
        tpv_summary = train_plus_val_df[self.config.label_column].value_counts().reset_index()
        tpv_summary.columns = ["umbrella", "train_plus_validation_count"]
        tpv_summary.to_csv(
            self.output_dir / "train_plus_validation_class_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Save best objects.
        final_vectorizer.save(self.output_dir / "final_vectorizer_state.pt")
        final_model.save(self.output_dir / "final_multinomial_nb_model.pt")

        best_config_export = {
            **best_payload,
            "final_retained_features_train_plus_validation_fit": len(final_vectorizer.vocabulary_),
            "class_labels": label_encoder.classes_,
            "balanced_total_rows": len(balanced_df),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        }
        (self.output_dir / "best_config.json").write_text(
            json.dumps(best_config_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._write_run_summary(
            balanced_df=balanced_df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            best_config=best_config_export,
            validation_macro_df=validation_macro_df,
            final_test_macro_df=final_test_macro_df,
        )

        print("\nDone. Balanced and tuned TF-IDF + Multinomial Naive Bayes experiment completed.")
        print(f"Balanced rows used: {len(balanced_df):,}")
        print(f"Train/Validation/Test: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
        print("Best validation configuration:")
        print(json.dumps(best_config_export, ensure_ascii=False, indent=2))
        print("\nFinal test macro summary:")
        print(final_test_macro_df.to_string(index=False))
        print(f"\nOutputs saved under: {self.output_dir.resolve()}")

    def _write_run_summary(
        self,
        balanced_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        best_config: Dict[str, object],
        validation_macro_df: pd.DataFrame,
        final_test_macro_df: pd.DataFrame,
    ) -> None:
        val_row = validation_macro_df.iloc[0].to_dict()
        test_row = final_test_macro_df.iloc[0].to_dict()

        lines = [
            "Balanced Tuned TF-IDF + Multinomial Naive Bayes Summary",
            "=========================================================",
            "",
            "Data decision:",
            "- Classes retained: anxiety_fear, ocd_obsessive, depression",
            "- Classes excluded: bipolar_mania, other_unclear",
            "- Anxiety/Fear and OCD/Obsessive were downsampled to Depression size",
            "- Depression was not oversampled",
            f"- Balanced rows used: {len(balanced_df):,}",
            "",
            "Split:",
            f"- Train rows: {len(train_df):,}",
            f"- Validation rows: {len(val_df):,}",
            f"- Test rows: {len(test_df):,}",
            "",
            "Best validation configuration:",
            f"- ngram_range: {best_config['ngram_range']}",
            f"- min_df: {best_config['min_df']}",
            f"- max_features: {best_config['max_features']}",
            f"- alpha: {best_config['alpha']}",
            f"- retained_features_validation_fit: {best_config['retained_features_validation_fit']}",
            "",
            "Best validation summary:",
            f"- Accuracy: {val_row['Accuracy']}",
            f"- Macro Precision: {val_row['Macro Precision']}",
            f"- Macro Recall: {val_row['Macro Recall']}",
            f"- Macro F1-score: {val_row['Macro F1-score']}",
            "",
            "Final test summary after retraining on Train + Validation:",
            f"- Accuracy: {test_row['Accuracy']}",
            f"- Macro Precision: {test_row['Macro Precision']}",
            f"- Macro Recall: {test_row['Macro Recall']}",
            f"- Macro F1-score: {test_row['Macro F1-score']}",
            "",
            "Main files:",
            "- hyperparameter_search_results.csv",
            "- best_validation_results.csv",
            "- best_validation_macro_summary.csv",
            "- final_test_results.csv",
            "- final_test_macro_summary.csv",
            "- final_test_confusion_matrix.csv",
            "- final_test_predictions.csv",
            "- best_config.json",
            "- final_vectorizer_state.pt",
            "- final_multinomial_nb_model.pt",
        ]

        (self.output_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    config = BalancedTunedConfig(
        dataset_path="final_modeling_dataset.csv",
        output_dir="tfidf_nb_balanced_tuned_outputs",
        text_column="analysis_text",
        label_column="umbrella",
        record_id_column="record_id",
        retained_labels=("anxiety_fear", "ocd_obsessive", "depression"),
        downsample_to_label="depression",
        random_seed=42,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        ngram_ranges=((1, 1), (1, 2)),
        min_df_values=(1, 2, 3),
        max_features_values=(30000, 50000, 80000),
        alpha_values=(0.01, 0.1, 0.5, 1.0, 2.0),
        max_df_ratio=0.95,
        l2_normalize=True,
        dtype=torch.float32,
        device="cpu",
    )

    experiment = BalancedTunedTFIDFMultinomialNBExperiment(config)
    experiment.run()
