"""Final AraBERT improvement attempt: PCA / whitening + Gaussian Naive Bayes.

Purpose
-------
This script is the final improvement attempt for the AraBERT vectorizer path.
It starts from the already-generated AraBERT embedding caches produced by:

    arabert_gaussian_nb_balanced_tuned.py

and tries to make those dense 768-dimensional embeddings more suitable for
Gaussian Naive Bayes.

The idea is:

    AraBERT mean embeddings
    -> optional embedding normalization mode
    -> PCA dimensionality reduction
    -> optional PCA whitening
    -> Gaussian Naive Bayes

Why this is a sensible final attempt
------------------------------------
Gaussian Naive Bayes assumes feature dimensions are modeled independently within
classes. Raw AraBERT embeddings are dense and correlated. PCA can rotate the
space into orthogonal directions, and whitening can scale those directions to
unit variance, which may better fit the Gaussian NB assumptions.

What this script does
---------------------
1) Rebuilds the exact same balanced 3-class dataset and exact same 70/15/15 split
   used in the prior AraBERT + Gaussian NB run:
      - anxiety_fear
      - depression
      - ocd_obsessive
   with sublabel-aware downsampling.

2) Loads the cached MEAN-pooled AraBERT embeddings from:
      arabert_gnb_balanced_tuned_outputs/
        - embedding_cache_train_mean.pt
        - embedding_cache_validation_mean.pt
        - embedding_cache_test_mean.pt

3) Tests several feature-space variants:
      - centered PCA
      - centered PCA + whitening
      - standardized PCA
      - standardized PCA + whitening
      - L2-normalized embeddings + PCA
      - L2-normalized embeddings + PCA + whitening

4) Tests PCA dimensions:
      16, 32, 64, 96, 128, 192, 256, 384, 512, 768

5) Tests Gaussian NB variance smoothing values:
      1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6

6) Selects the best configuration by Validation Macro F1-score, retrains on
   Train + Validation embeddings, and evaluates once on the Test set.

No scikit-learn is used. PCA, whitening, Gaussian NB, metrics, balancing, and
splitting are implemented using Python + pandas + PyTorch.

Expected folder layout
----------------------
Machine Learning/
├── final_modeling_dataset.csv
├── arabert_gaussian_nb_balanced_tuned.py
├── arabert_gnb_balanced_tuned_outputs/
│   ├── embedding_cache_train_mean.pt
│   ├── embedding_cache_validation_mean.pt
│   └── embedding_cache_test_mean.pt
└── arabert_gaussian_nb_pca_whitening_final.py

Outputs
-------
arabert_gnb_pca_whitening_final_outputs/
├── balanced_dataset_class_summary.csv
├── sublabel_aware_sampling_audit.csv
├── split_summary.csv
├── pca_search_results.csv
├── best_validation_results.csv
├── best_validation_macro_summary.csv
├── final_test_results.csv
├── final_test_macro_summary.csv
├── final_test_confusion_matrix.csv
├── final_test_predictions.csv
├── final_test_prediction_distribution.csv
├── best_config.json
├── final_pca_transform_state.pt
├── final_gaussian_nb_model.pt
└── run_summary.txt
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class PCAWhiteningConfig:
    """Configuration for final AraBERT + PCA/whitening + Gaussian NB attempt."""

    # Core data paths
    dataset_path: str = "final_modeling_dataset.csv"
    embedding_cache_dir: str = "arabert_gnb_balanced_tuned_outputs"
    output_dir: str = "arabert_gnb_pca_whitening_final_outputs"

    # Embedding cache names from the previous AraBERT-GNB run
    train_embedding_filename: str = "embedding_cache_train_mean.pt"
    validation_embedding_filename: str = "embedding_cache_validation_mean.pt"
    test_embedding_filename: str = "embedding_cache_test_mean.pt"

    # Columns from the final modeling dataset
    record_id_column: str = "record_id"
    text_column: str = "clean_text"
    label_column: str = "umbrella"
    sublabel_column: str = "label_clean"

    # Three retained classes
    retained_labels: Tuple[str, ...] = (
        "anxiety_fear",
        "depression",
        "ocd_obsessive",
    )
    downsample_to_label: str = "depression"

    # Must match the prior AraBERT GNB script exactly
    random_seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Final search grid
    preprocessing_modes: Tuple[str, ...] = (
        "centered",
        "standardized",
        "l2_normalized",
    )
    whitening_options: Tuple[bool, ...] = (False, True)
    pca_dimensions: Tuple[int, ...] = (16, 32, 64, 96, 128, 192, 256, 384, 512, 768)
    var_smoothing_values: Tuple[float, ...] = (
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
    )

    # Numeric stability
    standardization_epsilon: float = 1e-8
    whitening_epsilon: float = 1e-8
    l2_epsilon: float = 1e-12

    # Tensor settings
    dtype: torch.dtype = torch.float32


# =============================================================================
# 2. DATA LOADING + BALANCING + SPLITTING
# =============================================================================


class ModelingDatasetLoader:
    """Load the modeling CSV and retain the required rows/classes."""

    def __init__(self, config: PCAWhiteningConfig) -> None:
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
            raise ValueError(f"Dataset missing required columns: {missing}")

        if self.config.record_id_column not in df.columns:
            df[self.config.record_id_column] = range(1, len(df) + 1)

        for col in [self.config.text_column, self.config.label_column, self.config.sublabel_column]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df[
            df[self.config.text_column].ne("")
            & df[self.config.label_column].ne("")
            & df[self.config.sublabel_column].ne("")
        ].copy()

        df = df[df[self.config.label_column].isin(self.config.retained_labels)].copy()
        if df.empty:
            raise ValueError("No rows remain after retaining the three selected classes.")
        return df.reset_index(drop=True)


class SublabelAwareDownsampler:
    """Rebuild the exact sublabel-aware balanced dataset used in the previous script."""

    def __init__(self, config: PCAWhiteningConfig) -> None:
        self.config = config

    def balance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        label_col = self.config.label_column
        sublabel_col = self.config.sublabel_column
        anchor_label = self.config.downsample_to_label

        counts = df[label_col].value_counts()
        if anchor_label not in counts:
            raise ValueError(f"Anchor class not found: {anchor_label}")
        target_count = int(counts[anchor_label])
        rng = random.Random(self.config.random_seed)

        balanced_parts: List[pd.DataFrame] = []
        audit_rows: List[Dict[str, object]] = []

        for umbrella in self.config.retained_labels:
            group = df[df[label_col] == umbrella].copy()
            original_count = len(group)
            if original_count < target_count:
                raise ValueError(
                    f"Class '{umbrella}' has {original_count} rows, lower than target {target_count}. "
                    "This script only downsamples."
                )

            sub_counts = group[sublabel_col].value_counts().sort_index()
            if original_count == target_count:
                balanced_parts.append(group.copy())
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
            fractions: List[Tuple[str, float]] = []
            for sublabel, sub_count in sub_counts.items():
                sublabel_str = str(sublabel)
                ideal = float(sub_count) / float(original_count) * float(target_count)
                base = int(math.floor(ideal))
                allocations[sublabel_str] = min(base, int(sub_count))
                fractions.append((sublabel_str, ideal - base))

            remaining = target_count - sum(allocations.values())
            fractions.sort(key=lambda item: item[1], reverse=True)
            for sublabel_str, _ in fractions:
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
                    f"Allocation mismatch for '{umbrella}'. Expected {target_count}, "
                    f"got {sum(allocations.values())}."
                )

            sampled_subparts: List[pd.DataFrame] = []
            for sublabel, sub_count in sub_counts.items():
                sublabel_str = str(sublabel)
                desired = allocations[sublabel_str]
                sub_df = group[group[sublabel_col] == sublabel].copy()
                if desired <= 0:
                    sampled_sub = sub_df.iloc[0:0].copy()
                elif desired == len(sub_df):
                    sampled_sub = sub_df.copy()
                else:
                    sampled_indices = rng.sample(sub_df.index.tolist(), desired)
                    sampled_sub = sub_df.loc[sampled_indices].copy()

                sampled_subparts.append(sampled_sub)
                audit_rows.append({
                    "umbrella": umbrella,
                    "sublabel": sublabel_str,
                    "original_sublabel_count": int(sub_count),
                    "sampled_sublabel_count": int(desired),
                    "original_within_umbrella_pct": round(sub_count / original_count * 100, 4),
                    "sampled_within_umbrella_pct": round(desired / target_count * 100, 4),
                    "sampling_mode": "proportional_sublabel_aware_downsampling",
                })

            balanced_parts.append(pd.concat(sampled_subparts, ignore_index=True))

        balanced_df = pd.concat(balanced_parts, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1.0, random_state=self.config.random_seed).reset_index(drop=True)
        audit_df = pd.DataFrame(audit_rows)
        return balanced_df, audit_df

    def export_outputs(self, balanced_df: pd.DataFrame, audit_df: pd.DataFrame, output_dir: Path) -> None:
        label_col = self.config.label_column
        summary = balanced_df[label_col].value_counts().reset_index()
        summary.columns = ["umbrella", "row_count"]
        summary["percentage"] = (summary["row_count"] / len(balanced_df) * 100).round(2)
        summary.to_csv(output_dir / "balanced_dataset_class_summary.csv", index=False, encoding="utf-8-sig")
        audit_df.to_csv(output_dir / "sublabel_aware_sampling_audit.csv", index=False, encoding="utf-8-sig")


class ThreeWayStratifiedSplitter:
    """Rebuild the exact 70/15/15 split used for the embedding cache."""

    def __init__(self, config: PCAWhiteningConfig) -> None:
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
            indices = df[df[label_col] == label].index.tolist()
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

    def export_summary(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
        label_col = self.config.label_column
        train_counts = train_df[label_col].value_counts().rename("train_count")
        val_counts = val_df[label_col].value_counts().rename("validation_count")
        test_counts = test_df[label_col].value_counts().rename("test_count")
        summary = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int).reset_index()
        summary.rename(columns={summary.columns[0]: "umbrella"}, inplace=True)
        summary["total_count"] = summary["train_count"] + summary["validation_count"] + summary["test_count"]
        summary["train_ratio"] = (summary["train_count"] / summary["total_count"]).round(4)
        summary["validation_ratio"] = (summary["validation_count"] / summary["total_count"]).round(4)
        summary["test_ratio"] = (summary["test_count"] / summary["total_count"]).round(4)
        summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# 3. LABEL ENCODER
# =============================================================================


class LabelEncoderTorch:
    """Encode string labels to integer IDs."""

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
            raise RuntimeError("Label encoder not fitted.")
        encoded: List[int] = []
        for label in labels:
            label_str = str(label)
            if label_str not in self.class_to_index_:
                raise ValueError(f"Unknown label: {label_str}")
            encoded.append(self.class_to_index_[label_str])
        return torch.tensor(encoded, dtype=torch.long)

    def inverse_transform(self, indices: Sequence[int]) -> List[str]:
        return [self.classes_[int(idx)] for idx in indices]


# =============================================================================
# 4. EMBEDDING CACHE LOADER
# =============================================================================


class EmbeddingCacheLoader:
    """Load saved mean-pooled AraBERT embeddings from the previous experiment."""

    def __init__(self, config: PCAWhiteningConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.embedding_cache_dir)

    def load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_path = self.cache_dir / self.config.train_embedding_filename
        val_path = self.cache_dir / self.config.validation_embedding_filename
        test_path = self.cache_dir / self.config.test_embedding_filename
        for path in [train_path, val_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing embedding cache: {path.resolve()}\n"
                    "Run arabert_gaussian_nb_balanced_tuned.py first so the cache files exist."
                )

        X_train = torch.load(train_path, map_location="cpu").to(self.config.dtype)
        X_val = torch.load(val_path, map_location="cpu").to(self.config.dtype)
        X_test = torch.load(test_path, map_location="cpu").to(self.config.dtype)
        return X_train, X_val, X_test


# =============================================================================
# 5. PCA / WHITENING TRANSFORM
# =============================================================================


class PCATransformerTorch:
    """Fit PCA on training embeddings and transform validation/test embeddings."""

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

        # l2_normalized, then center
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
            raise ValueError(
                f"Requested {self.n_components} PCA components, but maximum is {max_components}."
            )

        # Full low-rank decomposition is stable at 4773 x 768 and avoids sklearn.
        # torch.linalg.svd returns U, S, Vh; principal axes are rows of Vh.
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

    def export_state(self) -> Dict[str, object]:
        if self.center_ is None or self.scale_ is None or self.components_ is None or self.explained_variance_ is None:
            raise RuntimeError("Cannot export unfitted PCA transformer.")
        return {
            "preprocessing_mode": self.preprocessing_mode,
            "n_components": self.n_components,
            "whiten": self.whiten,
            "standardization_epsilon": self.standardization_epsilon,
            "whitening_epsilon": self.whitening_epsilon,
            "l2_epsilon": self.l2_epsilon,
            "center": self.center_,
            "scale": self.scale_,
            "components": self.components_,
            "singular_values": self.singular_values_,
            "explained_variance": self.explained_variance_,
            "n_samples_fit": self.n_samples_fit_,
        }


# =============================================================================
# 6. GAUSSIAN NAIVE BAYES
# =============================================================================


class TorchGaussianNaiveBayes:
    """Gaussian NB for dense continuous PCA components."""

    def __init__(self, var_smoothing: float, dtype: torch.dtype) -> None:
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be >= 0")
        self.var_smoothing = float(var_smoothing)
        self.dtype = dtype
        self.theta_: Optional[torch.Tensor] = None
        self.var_: Optional[torch.Tensor] = None
        self.class_log_prior_: Optional[torch.Tensor] = None
        self.class_count_: Optional[torch.Tensor] = None
        self.epsilon_: Optional[float] = None
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None

    def fit(self, X: torch.Tensor, y: torch.Tensor, n_classes: int) -> "TorchGaussianNaiveBayes":
        if X.ndim != 2:
            raise ValueError("X must be a dense 2D tensor.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.size(0) != y.numel():
            raise ValueError("X rows and y length mismatch.")

        X = X.detach().cpu().to(self.dtype)
        y = y.detach().cpu().long()
        self.n_classes_ = int(n_classes)
        self.n_features_ = int(X.size(1))
        class_count = torch.bincount(y, minlength=self.n_classes_).to(self.dtype)
        if torch.any(class_count == 0):
            raise ValueError("At least one class has zero training rows.")

        means: List[torch.Tensor] = []
        variances: List[torch.Tensor] = []
        for class_idx in range(self.n_classes_):
            class_X = X[y == class_idx]
            means.append(class_X.mean(dim=0))
            variances.append(class_X.var(dim=0, unbiased=False))

        theta = torch.stack(means, dim=0)
        raw_var = torch.stack(variances, dim=0)
        global_feature_var = X.var(dim=0, unbiased=False)
        max_global_var = float(global_feature_var.max().item()) if X.numel() else 1.0
        epsilon = self.var_smoothing * max_global_var
        epsilon = max(epsilon, 1e-12)

        self.theta_ = theta
        self.var_ = raw_var + epsilon
        self.class_log_prior_ = torch.log(class_count) - torch.log(class_count.sum())
        self.class_count_ = class_count
        self.epsilon_ = epsilon
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.theta_ is None or self.var_ is None or self.class_log_prior_ is None:
            raise RuntimeError("Gaussian NB not fitted.")
        X = X.detach().cpu().to(self.dtype)
        diff = X.unsqueeze(1) - self.theta_.unsqueeze(0)
        log_likelihood = -0.5 * (
            torch.log(2.0 * torch.tensor(math.pi, dtype=self.dtype) * self.var_).unsqueeze(0)
            + (diff * diff) / self.var_.unsqueeze(0)
        )
        scores = log_likelihood.sum(dim=2) + self.class_log_prior_.unsqueeze(0)
        return torch.argmax(scores, dim=1)

    def save(self, path: Path) -> None:
        if self.theta_ is None or self.var_ is None or self.class_log_prior_ is None:
            raise RuntimeError("Cannot save unfitted Gaussian NB.")
        torch.save({
            "var_smoothing": self.var_smoothing,
            "epsilon": self.epsilon_,
            "theta": self.theta_,
            "var": self.var_,
            "class_log_prior": self.class_log_prior_,
            "class_count": self.class_count_,
            "n_classes": self.n_classes_,
            "n_features": self.n_features_,
        }, path)


# =============================================================================
# 7. EVALUATION
# =============================================================================


class ClassificationEvaluator:
    """Manual metrics with the same output table style as previous experiments."""

    def __init__(self, class_labels: Sequence[str]) -> None:
        self.class_labels = list(class_labels)
        self.n_classes = len(self.class_labels)

    def confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true = y_true.detach().cpu().long()
        y_pred = y_pred.detach().cpu().long()
        cm = torch.zeros((self.n_classes, self.n_classes), dtype=torch.long)
        flat = y_true * self.n_classes + y_pred
        cm.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        return cm

    def evaluate(self, model_name: str, y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cm = self.confusion_matrix(y_true, y_pred)
        total = int(cm.sum().item())
        correct = int(torch.diag(cm).sum().item())
        accuracy = correct / total if total else 0.0
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        supports: List[int] = []

        for idx in range(self.n_classes):
            tp = float(cm[idx, idx].item())
            pred_total = float(cm[:, idx].sum().item())
            actual_total = float(cm[idx, :].sum().item())
            precision = tp / pred_total if pred_total > 0 else 0.0
            recall = tp / actual_total if actual_total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(int(actual_total))

        macro_precision = sum(precisions) / self.n_classes if self.n_classes else 0.0
        macro_recall = sum(recalls) / self.n_classes if self.n_classes else 0.0
        macro_f1 = sum(f1s) / self.n_classes if self.n_classes else 0.0

        rows: List[Dict[str, object]] = []
        for i, label in enumerate(self.class_labels):
            rows.append({
                "MODEL": model_name,
                "Class": label,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precisions[i], 4),
                "Recall": round(recalls[i], 4),
                "F1-score": round(f1s[i], 4),
                "Support": supports[i],
                "Macro Avg": round(macro_f1, 4),
            })

        per_class_df = pd.DataFrame(rows)
        macro_df = pd.DataFrame([{
            "MODEL": model_name,
            "Accuracy": round(accuracy, 4),
            "Macro Precision": round(macro_precision, 4),
            "Macro Recall": round(macro_recall, 4),
            "Macro F1-score": round(macro_f1, 4),
            "Total Support": total,
        }])
        cm_df = pd.DataFrame(
            cm.numpy(),
            index=[f"actual::{label}" for label in self.class_labels],
            columns=[f"predicted::{label}" for label in self.class_labels],
        )
        return per_class_df, macro_df, cm_df


# =============================================================================
# 8. SEARCH RESULT DATACLASS
# =============================================================================


@dataclass
class SearchResult:
    run_id: int
    preprocessing_mode: str
    whiten: bool
    pca_dimensions: int
    var_smoothing: float
    validation_accuracy: float
    validation_macro_precision: float
    validation_macro_recall: float
    validation_macro_f1: float


# =============================================================================
# 9. FULL FINAL EXPERIMENT
# =============================================================================


class FinalPCAGaussianNBExperiment:
    """Search PCA/whitening configurations on cached AraBERT embeddings."""

    def __init__(self, config: PCAWhiteningConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        # A. Rebuild data rows/splits so labels align with cached embedding order.
        loader = ModelingDatasetLoader(self.config)
        df = loader.load()
        downsampler = SublabelAwareDownsampler(self.config)
        balanced_df, sampling_audit_df = downsampler.balance(df)
        downsampler.export_outputs(balanced_df, sampling_audit_df, self.output_dir)
        splitter = ThreeWayStratifiedSplitter(self.config)
        train_df, val_df, test_df = splitter.split(balanced_df)
        splitter.export_summary(train_df, val_df, test_df, self.output_dir)

        encoder = LabelEncoderTorch().fit(train_df[self.config.label_column].tolist())
        y_train = encoder.transform(train_df[self.config.label_column].tolist())
        y_val = encoder.transform(val_df[self.config.label_column].tolist())
        y_test = encoder.transform(test_df[self.config.label_column].tolist())

        # B. Load cached mean-pool AraBERT embeddings.
        cache_loader = EmbeddingCacheLoader(self.config)
        X_train_raw, X_val_raw, X_test_raw = cache_loader.load()

        if X_train_raw.size(0) != len(train_df) or X_val_raw.size(0) != len(val_df) or X_test_raw.size(0) != len(test_df):
            raise ValueError(
                "Embedding cache row counts do not match the rebuilt train/validation/test split. "
                "Do not proceed; this indicates cache/config mismatch."
            )

        print(f"Loaded cached AraBERT mean embeddings:")
        print(f"- Train: {tuple(X_train_raw.shape)}")
        print(f"- Validation: {tuple(X_val_raw.shape)}")
        print(f"- Test: {tuple(X_test_raw.shape)}")
        print("No AraBERT re-vectorization is needed for this run.")

        evaluator = ClassificationEvaluator(encoder.classes_)
        results: List[SearchResult] = []
        best_payload: Optional[Dict[str, object]] = None
        best_f1 = -1.0
        best_accuracy = -1.0
        run_id = 0

        total_runs = (
            len(self.config.preprocessing_modes)
            * len(self.config.whitening_options)
            * len(self.config.pca_dimensions)
            * len(self.config.var_smoothing_values)
        )
        print(f"Total PCA/GNB validation runs: {total_runs:,}")

        # Fit PCA once per preprocessing + whitening + component count.
        for mode in self.config.preprocessing_modes:
            for whiten in self.config.whitening_options:
                for n_components in self.config.pca_dimensions:
                    transformer = PCATransformerTorch(
                        preprocessing_mode=mode,
                        n_components=n_components,
                        whiten=whiten,
                        standardization_epsilon=self.config.standardization_epsilon,
                        whitening_epsilon=self.config.whitening_epsilon,
                        l2_epsilon=self.config.l2_epsilon,
                        dtype=self.config.dtype,
                    )
                    X_train_pca = transformer.fit_transform(X_train_raw)
                    X_val_pca = transformer.transform(X_val_raw)

                    for var_smoothing in self.config.var_smoothing_values:
                        run_id += 1
                        model = TorchGaussianNaiveBayes(var_smoothing=var_smoothing, dtype=self.config.dtype)
                        model.fit(X_train_pca, y_train, n_classes=len(encoder.classes_))
                        y_val_pred = model.predict(X_val_pca)
                        _, macro_df, _ = evaluator.evaluate(
                            model_name="AraBERT PCA/Whitened + Gaussian NB Validation",
                            y_true=y_val,
                            y_pred=y_val_pred,
                        )
                        row = macro_df.iloc[0]
                        macro_f1 = float(row["Macro F1-score"])
                        accuracy = float(row["Accuracy"])

                        result = SearchResult(
                            run_id=run_id,
                            preprocessing_mode=mode,
                            whiten=whiten,
                            pca_dimensions=n_components,
                            var_smoothing=var_smoothing,
                            validation_accuracy=accuracy,
                            validation_macro_precision=float(row["Macro Precision"]),
                            validation_macro_recall=float(row["Macro Recall"]),
                            validation_macro_f1=macro_f1,
                        )
                        results.append(result)

                        is_better = (
                            macro_f1 > best_f1
                            or (math.isclose(macro_f1, best_f1) and accuracy > best_accuracy)
                        )
                        if is_better:
                            best_f1 = macro_f1
                            best_accuracy = accuracy
                            best_payload = {
                                "preprocessing_mode": mode,
                                "whiten": whiten,
                                "pca_dimensions": n_components,
                                "var_smoothing": var_smoothing,
                                "validation_macro_f1": macro_f1,
                                "validation_accuracy": accuracy,
                            }

                        if run_id == 1 or run_id % 25 == 0 or run_id == total_runs:
                            print(
                                f"Run {run_id:>3}/{total_runs}: mode={mode}, whiten={whiten}, "
                                f"pca={n_components}, var_smoothing={var_smoothing:.0e} -> "
                                f"Val Macro F1={macro_f1:.4f}, Val Acc={accuracy:.4f}"
                            )

        if best_payload is None:
            raise RuntimeError("Search completed without a best configuration.")

        search_df = pd.DataFrame([asdict(item) for item in results])
        search_df = search_df.sort_values(
            ["validation_macro_f1", "validation_accuracy"],
            ascending=[False, False],
        ).reset_index(drop=True)
        search_df.to_csv(self.output_dir / "pca_search_results.csv", index=False, encoding="utf-8-sig")

        # C. Rebuild the best validation pipeline for a detailed validation report.
        best_transformer = PCATransformerTorch(
            preprocessing_mode=str(best_payload["preprocessing_mode"]),
            n_components=int(best_payload["pca_dimensions"]),
            whiten=bool(best_payload["whiten"]),
            standardization_epsilon=self.config.standardization_epsilon,
            whitening_epsilon=self.config.whitening_epsilon,
            l2_epsilon=self.config.l2_epsilon,
            dtype=self.config.dtype,
        )
        X_train_best = best_transformer.fit_transform(X_train_raw)
        X_val_best = best_transformer.transform(X_val_raw)
        best_val_model = TorchGaussianNaiveBayes(
            var_smoothing=float(best_payload["var_smoothing"]),
            dtype=self.config.dtype,
        )
        best_val_model.fit(X_train_best, y_train, n_classes=len(encoder.classes_))
        y_val_best_pred = best_val_model.predict(X_val_best)
        val_results_df, val_macro_df, _ = evaluator.evaluate(
            model_name="Best Validation AraBERT PCA/Whitening + Gaussian NB",
            y_true=y_val,
            y_pred=y_val_best_pred,
        )
        val_results_df.to_csv(self.output_dir / "best_validation_results.csv", index=False, encoding="utf-8-sig")
        val_macro_df.to_csv(self.output_dir / "best_validation_macro_summary.csv", index=False, encoding="utf-8-sig")

        # D. Retrain PCA and Gaussian NB on Train + Validation, test once.
        X_train_plus_val_raw = torch.cat([X_train_raw, X_val_raw], dim=0)
        y_train_plus_val = torch.cat([y_train, y_val], dim=0)
        final_transformer = PCATransformerTorch(
            preprocessing_mode=str(best_payload["preprocessing_mode"]),
            n_components=int(best_payload["pca_dimensions"]),
            whiten=bool(best_payload["whiten"]),
            standardization_epsilon=self.config.standardization_epsilon,
            whitening_epsilon=self.config.whitening_epsilon,
            l2_epsilon=self.config.l2_epsilon,
            dtype=self.config.dtype,
        )
        X_train_plus_val_final = final_transformer.fit_transform(X_train_plus_val_raw)
        X_test_final = final_transformer.transform(X_test_raw)

        final_model = TorchGaussianNaiveBayes(
            var_smoothing=float(best_payload["var_smoothing"]),
            dtype=self.config.dtype,
        )
        final_model.fit(X_train_plus_val_final, y_train_plus_val, n_classes=len(encoder.classes_))
        y_test_pred = final_model.predict(X_test_final)
        final_results_df, final_macro_df, final_cm_df = evaluator.evaluate(
            model_name="Final AraBERT PCA/Whitening + Gaussian NB",
            y_true=y_test,
            y_pred=y_test_pred,
        )

        final_results_df.to_csv(self.output_dir / "final_test_results.csv", index=False, encoding="utf-8-sig")
        final_macro_df.to_csv(self.output_dir / "final_test_macro_summary.csv", index=False, encoding="utf-8-sig")
        final_cm_df.to_csv(self.output_dir / "final_test_confusion_matrix.csv", encoding="utf-8-sig")

        predicted_labels = encoder.inverse_transform(y_test_pred.detach().cpu().tolist())
        prediction_cols = [
            col for col in [
                self.config.record_id_column,
                self.config.text_column,
                self.config.label_column,
                self.config.sublabel_column,
            ] if col in test_df.columns
        ]
        predictions_df = test_df[prediction_cols].copy()
        predictions_df.rename(columns={self.config.label_column: "actual_umbrella"}, inplace=True)
        predictions_df["predicted_umbrella"] = predicted_labels
        predictions_df["is_correct"] = predictions_df["actual_umbrella"] == predictions_df["predicted_umbrella"]
        predictions_df.to_csv(self.output_dir / "final_test_predictions.csv", index=False, encoding="utf-8-sig")

        pred_dist = predictions_df["predicted_umbrella"].value_counts().reset_index()
        pred_dist.columns = ["predicted_umbrella", "prediction_count"]
        pred_dist["prediction_percentage"] = (pred_dist["prediction_count"] / len(predictions_df) * 100).round(2)
        pred_dist.to_csv(self.output_dir / "final_test_prediction_distribution.csv", index=False, encoding="utf-8-sig")

        best_config_export = {
            **best_payload,
            "class_labels": encoder.classes_,
            "balanced_total_rows": len(balanced_df),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "embedding_source": "cached AraBERT mean-pooled embeddings",
            "embedding_cache_dir": self.config.embedding_cache_dir,
            "embedding_dimension_before_pca": int(X_train_raw.size(1)),
        }
        (self.output_dir / "best_config.json").write_text(
            json.dumps(best_config_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        torch.save(final_transformer.export_state(), self.output_dir / "final_pca_transform_state.pt")
        final_model.save(self.output_dir / "final_gaussian_nb_model.pt")

        self._write_summary(
            balanced_df=balanced_df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            best_config=best_config_export,
            val_macro_df=val_macro_df,
            final_macro_df=final_macro_df,
        )

        print("\nDone. Final AraBERT PCA/Whitening + Gaussian NB experiment completed.")
        print(f"Balanced rows: {len(balanced_df):,}")
        print(f"Train/Validation/Test: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
        print("Best validation configuration:")
        print(json.dumps(best_config_export, ensure_ascii=False, indent=2))
        print("\nFinal test macro summary:")
        print(final_macro_df.to_string(index=False))
        print(f"\nOutputs saved under: {self.output_dir.resolve()}")

    def _write_summary(
        self,
        balanced_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        best_config: Dict[str, object],
        val_macro_df: pd.DataFrame,
        final_macro_df: pd.DataFrame,
    ) -> None:
        val_row = val_macro_df.iloc[0].to_dict()
        test_row = final_macro_df.iloc[0].to_dict()
        lines = [
            "Final AraBERT PCA/Whitening + Gaussian Naive Bayes Summary",
            "============================================================",
            "",
            "Purpose:",
            "- Improve the AraBERT embedding + Gaussian NB path by reshaping the feature space",
            "- Use PCA dimensionality reduction and optional whitening to reduce correlation/redundancy",
            "",
            "Dataset:",
            "- Retained classes: anxiety_fear, depression, ocd_obsessive",
            "- Excluded classes: bipolar_mania, other_unclear",
            "- Downsampling: sublabel-aware proportional downsampling",
            f"- Balanced rows: {len(balanced_df):,}",
            "",
            "Split:",
            f"- Train: {len(train_df):,}",
            f"- Validation: {len(val_df):,}",
            f"- Test: {len(test_df):,}",
            "",
            "Best selected configuration:",
            f"- Preprocessing mode: {best_config['preprocessing_mode']}",
            f"- Whitening: {best_config['whiten']}",
            f"- PCA dimensions: {best_config['pca_dimensions']}",
            f"- Gaussian NB variance smoothing: {best_config['var_smoothing']}",
            f"- Validation Macro F1: {best_config['validation_macro_f1']}",
            f"- Validation Accuracy: {best_config['validation_accuracy']}",
            "",
            "Detailed validation report:",
            f"- Accuracy: {val_row['Accuracy']}",
            f"- Macro Precision: {val_row['Macro Precision']}",
            f"- Macro Recall: {val_row['Macro Recall']}",
            f"- Macro F1-score: {val_row['Macro F1-score']}",
            "",
            "Final held-out test report:",
            f"- Accuracy: {test_row['Accuracy']}",
            f"- Macro Precision: {test_row['Macro Precision']}",
            f"- Macro Recall: {test_row['Macro Recall']}",
            f"- Macro F1-score: {test_row['Macro F1-score']}",
            "",
            "Main outputs:",
            "- pca_search_results.csv",
            "- best_validation_results.csv",
            "- final_test_results.csv",
            "- final_test_macro_summary.csv",
            "- final_test_confusion_matrix.csv",
            "- final_test_predictions.csv",
            "- best_config.json",
            "- final_pca_transform_state.pt",
            "- final_gaussian_nb_model.pt",
        ]
        (self.output_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 10. MAIN
# =============================================================================


if __name__ == "__main__":
    config = PCAWhiteningConfig(
        dataset_path="final_modeling_dataset.csv",
        embedding_cache_dir="arabert_gnb_balanced_tuned_outputs",
        output_dir="arabert_gnb_pca_whitening_final_outputs",
        train_embedding_filename="embedding_cache_train_mean.pt",
        validation_embedding_filename="embedding_cache_validation_mean.pt",
        test_embedding_filename="embedding_cache_test_mean.pt",
        record_id_column="record_id",
        text_column="clean_text",
        label_column="umbrella",
        sublabel_column="label_clean",
        retained_labels=("anxiety_fear", "depression", "ocd_obsessive"),
        downsample_to_label="depression",
        random_seed=42,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        preprocessing_modes=("centered", "standardized", "l2_normalized"),
        whitening_options=(False, True),
        pca_dimensions=(16, 32, 64, 96, 128, 192, 256, 384, 512, 768),
        var_smoothing_values=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6),
        standardization_epsilon=1e-8,
        whitening_epsilon=1e-8,
        l2_epsilon=1e-12,
        dtype=torch.float32,
    )

    experiment = FinalPCAGaussianNBExperiment(config)
    experiment.run()
