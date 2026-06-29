"""TF-IDF + Juliette's SVM on Jean's final Arabic modeling dataset.

This script keeps Juliette's SVMClassifier class exactly as provided and uses it
on the same final acceptable TF-IDF setup previously selected for Jean's dataset:

- Keep only:
    anxiety_fear, depression, ocd_obsessive
- Sublabel-aware downsampling to the Depression class size
- 70/15/15 stratified train/validation/test split
- Final accepted hybrid TF-IDF representation:
    * analysis_text word TF-IDF, n-grams (1, 3), min_df=3, max_features=50,000
    * clean_text character TF-IDF, n-grams (4, 6), min_df=3, max_features=50,000
- Final classifier swap:
    * Juliette's unchanged SVMClassifier class
    * Train on Train + Validation, evaluate once on Test

Expected folder:
juliette model/
├── final_modeling_dataset.csv
└── tfidf_svm_juliette_on_jean_dataset.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class TFIDFSVMConfig:
    dataset_path: str = "final_modeling_dataset.csv"
    output_dir: str = "light_tfidf_svm_juliette_outputs"

    record_id_column: str = "record_id"
    analysis_text_column: str = "analysis_text"
    clean_text_column: str = "clean_text"
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

    # Same final accepted TF-IDF settings from Jean's best TF-IDF experiment.
    word_ngram_range: Tuple[int, int] = (1, 2)
    word_min_df: int = 3
    word_max_features: int = 30000

    char_ngram_range: Tuple[int, int] = (4, 6)
    char_min_df: int = 3
    char_max_features: int = 50000

    max_df_ratio: float = 0.98
    l2_normalize: bool = True

    # Passed into Juliette's unchanged SVMClassifier constructor.
    svm_cv: int = 5
    svm_scoring: str = "f1_macro"
    svm_n_jobs: int = -1
    svm_verbose: int = 1


# =============================================================================
# 2. DATA LOADING
# =============================================================================


class ModelingDatasetLoader:
    """Load and validate the final modeling dataset."""

    def __init__(self, config: TFIDFSVMConfig) -> None:
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
            self.config.analysis_text_column,
            self.config.clean_text_column,
            self.config.label_column,
            self.config.sublabel_column,
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        if self.config.record_id_column not in df.columns:
            df[self.config.record_id_column] = range(1, len(df) + 1)

        for col in [
            self.config.analysis_text_column,
            self.config.clean_text_column,
            self.config.label_column,
            self.config.sublabel_column,
        ]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df[
            df[self.config.analysis_text_column].ne("")
            & df[self.config.clean_text_column].ne("")
            & df[self.config.label_column].ne("")
            & df[self.config.sublabel_column].ne("")
        ].copy()

        df = df[df[self.config.label_column].isin(self.config.retained_labels)].copy()
        if df.empty:
            raise ValueError("No rows remain after retaining the selected classes.")

        return df.reset_index(drop=True)


# =============================================================================
# 3. SUBLABEL-AWARE DOWNSAMPLING
# =============================================================================


class SublabelAwareDownsampler:
    """Downsample larger umbrellas while preserving sublabel proportions."""

    def __init__(self, config: TFIDFSVMConfig) -> None:
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
# 4. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# =============================================================================


class ThreeWayStratifiedSplitter:
    """Manual stratified 70/15/15 split by umbrella class."""

    def __init__(self, config: TFIDFSVMConfig) -> None:
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
# 5. FINAL ACCEPTED MANUAL TF-IDF VECTORIZERS
# =============================================================================


def word_terms(text: str, ngram_range: Tuple[int, int]) -> List[str]:
    tokens = [token for token in str(text).split() if token]
    terms: List[str] = []

    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            terms.append(" ".join(tokens[i:i + n]))

    return terms


def char_terms(text: str, ngram_range: Tuple[int, int]) -> List[str]:
    normalized = " ".join(str(text).split())
    terms: List[str] = []

    if not normalized:
        return terms

    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(normalized) < n:
            continue
        for i in range(len(normalized) - n + 1):
            gram = normalized[i:i + n]
            if gram.strip():
                terms.append(gram)

    return terms


class GenericTfidfVectorizer:
    """Manual TF-IDF vectorizer matching Jean's final acceptable TF-IDF pipeline."""

    def __init__(
        self,
        analyzer: str,
        ngram_range: Tuple[int, int],
        min_df: int,
        max_df_ratio: float,
        max_features: Optional[int],
        l2_normalize: bool,
    ) -> None:
        if analyzer not in {"word", "char"}:
            raise ValueError("analyzer must be 'word' or 'char'.")

        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_features = max_features
        self.l2_normalize = l2_normalize

        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.n_documents_: int = 0

    def _terms(self, text: str) -> List[str]:
        if self.analyzer == "word":
            return word_terms(text, self.ngram_range)
        return char_terms(text, self.ngram_range)

    def fit(self, texts: Sequence[str]) -> "GenericTfidfVectorizer":
        self.n_documents_ = len(texts)
        if self.n_documents_ == 0:
            raise ValueError("Cannot fit TF-IDF on zero texts.")

        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()

        for text in texts:
            terms = self._terms(text)
            term_freq.update(terms)
            doc_freq.update(set(terms))

        max_df_count = max(1, int(math.floor(self.max_df_ratio * self.n_documents_)))
        eligible_terms = [
            term
            for term, df in doc_freq.items()
            if df >= self.min_df and df <= max_df_count
        ]

        if not eligible_terms:
            raise ValueError("No terms survived TF-IDF filtering.")

        eligible_terms.sort(key=lambda term: (-term_freq[term], term))

        if self.max_features is not None:
            eligible_terms = eligible_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(eligible_terms)}

        idf_values = [
            math.log((1.0 + self.n_documents_) / (1.0 + doc_freq[term])) + 1.0
            for term in eligible_terms
        ]
        self.idf_ = np.asarray(idf_values, dtype=np.float32)
        return self

    def fit_transform(self, texts: Sequence[str]):
        self.fit(texts)
        return self.transform(texts)

    def transform(self, texts: Sequence[str]):
        if not self.vocabulary_ or self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted.")

        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[float] = []

        for row_idx, text in enumerate(texts):
            terms = self._terms(text)
            counts = Counter(term for term in terms if term in self.vocabulary_)
            if not counts:
                continue

            row_features: List[Tuple[int, float]] = []
            for term, count in counts.items():
                col = self.vocabulary_[term]
                value = float(count) * float(self.idf_[col])
                row_features.append((col, value))

            if self.l2_normalize:
                norm = math.sqrt(sum(value * value for _, value in row_features))
                if norm > 0:
                    row_features = [(col, value / norm) for col, value in row_features]

            for col, value in row_features:
                row_indices.append(row_idx)
                col_indices.append(col)
                values.append(value)

        matrix = coo_matrix(
            (
                np.asarray(values, dtype=np.float32),
                (
                    np.asarray(row_indices, dtype=np.int64),
                    np.asarray(col_indices, dtype=np.int64),
                ),
            ),
            shape=(len(texts), len(self.vocabulary_)),
            dtype=np.float32,
        )
        return matrix.tocsr()


# =============================================================================
# 6. JULIETTE'S SVM CLASS — UNCHANGED
# =============================================================================


class SVMClassifier:
    """
    Support Vector Machine classifier with built-in GridSearch hyperparameter tuning.

    Attributes:
        best_model   : best SVC found by GridSearchCV
        best_params  : dict of best hyperparameters
        cv_results   : full GridSearchCV results dataframe
    """

    # Hyperparameter grid — covers kernel, C (regularization), gamma (for rbf/poly)
    PARAM_GRID = {
        'kernel' : ['rbf', 'poly', 'sigmoid'],
        'C'      : [0.1, 1, 10, 100],
        'gamma'  : ['scale', 'auto'],
    }

    def __init__(self, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1):
        """
        Args:
            cv      : number of cross-validation folds for GridSearch
            scoring : metric to optimise during grid search
            n_jobs  : parallel jobs (-1 = all cores)
            verbose : verbosity level of GridSearchCV
        """
        self.cv      = cv
        self.scoring = scoring
        self.n_jobs  = n_jobs
        self.verbose = verbose

        self.best_model  = None
        self.best_params = None
        self.cv_results  = None

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """
        Run GridSearchCV over PARAM_GRID and store the best estimator.

        Args:
            X_train : feature matrix (dense or sparse)
            y_train : target labels
        """
        grid_search = GridSearchCV(
            estimator  = SVC(),
            param_grid = self.PARAM_GRID,
            cv         = self.cv,
            scoring    = self.scoring,
            n_jobs     = self.n_jobs,
            verbose    = self.verbose,
            refit      = True   # refit best model on full train set
        )
        grid_search.fit(X_train, y_train)

        self.best_model  = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results  = pd.DataFrame(grid_search.cv_results_)

        print(f"\nBest params  : {self.best_params}")
        print(f"Best CV score ({self.scoring}): {grid_search.best_score_:.4f}")
        return self

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels.

        Args:
            X : feature matrix
        Returns:
            np.ndarray of predicted labels
        """
        if self.best_model is None:
            raise RuntimeError("Call .fit() before .predict()")
        return self.best_model.predict(X)

    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test, label_encoder=None, model_name='SVM'):
        """
        Compute evaluation metrics and return a per-class results table.

        Args:
            X_test        : test feature matrix
            y_test        : true labels
            label_encoder : sklearn LabelEncoder (for class names)
            model_name    : string label shown in the Model column

        Returns:
            results_df : DataFrame with columns:
                         Model | Class | Accuracy | Precision | Recall | F1-score | Support | Macro avg
            y_pred     : predicted labels array
        """
        y_pred       = self.predict(X_test)
        target_names = label_encoder.classes_ if label_encoder else None

        # Get full classification report as dict
        report   = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = report['macro avg']['f1-score']

        # Build one row per class
        rows = []
        for i, class_name in enumerate(target_names):
            rows.append({
                'Model'     : model_name if i == 0 else '',   # model name only on first row
                'Class'     : class_name,
                'Accuracy'  : round(accuracy, 2) if i == 0 else '',
                'Precision' : round(report[class_name]['precision'], 2),
                'Recall'    : round(report[class_name]['recall'], 2),
                'F1-score'  : round(report[class_name]['f1-score'], 2),
                'Support'   : int(report[class_name]['support']),
                'Macro avg' : round(macro_f1, 2) if i == 0 else '',
            })

        results_df = pd.DataFrame(rows)

        print(f"\n--- {model_name} Evaluation ---")
        print(results_df.to_string(index=False))

        return results_df, y_pred

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, y_test, y_pred, label_encoder=None):
        """
        Plot a heatmap confusion matrix.

        Args:
            y_test        : true labels
            y_pred        : predicted labels
            label_encoder : sklearn LabelEncoder (for axis labels)
        """
        cm = confusion_matrix(y_test, y_pred)
        labels = label_encoder.classes_ if label_encoder else None

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues'
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix — SVM')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def top_grid_results(self, n=10):
        """Return top-n grid search results sorted by mean test score."""
        cols = ['param_kernel', 'param_C', 'param_gamma',
                'mean_test_score', 'std_test_score', 'rank_test_score']
        return (
            self.cv_results[cols]
            .sort_values('rank_test_score')
            .head(n)
            .reset_index(drop=True)
        )


# =============================================================================
# 7. FULL EXPERIMENT
# =============================================================================


def run_experiment() -> None:
    config = TFIDFSVMConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    df = ModelingDatasetLoader(config).load()
    balanced_df, audit_df = SublabelAwareDownsampler(config).balance(df)
    train_df, val_df, test_df = ThreeWayStratifiedSplitter(config).split(balanced_df)
    train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Balanced rows used: {len(balanced_df):,}")
    print(f"Train/Validation/Test rows: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"Final SVM training rows (Train + Validation): {len(train_plus_val_df):,}")

    # Lighter word-only TF-IDF path.
    # The prior 100,000-feature hybrid TF-IDF representation was too slow
    # with Juliette's unchanged 120-fit SVM grid search.
    word_vectorizer = GenericTfidfVectorizer(
        analyzer="word",
        ngram_range=config.word_ngram_range,
        min_df=config.word_min_df,
        max_df_ratio=config.max_df_ratio,
        max_features=config.word_max_features,
        l2_normalize=config.l2_normalize,
    )

    X_train = word_vectorizer.fit_transform(
        train_plus_val_df[config.analysis_text_column].tolist()
    )
    X_test = word_vectorizer.transform(
        test_df[config.analysis_text_column].tolist()
    )

    print(f"Lighter TF-IDF training features: {X_train.shape[1]:,}")
    print(f"Lighter TF-IDF test rows: {X_test.shape[0]:,}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_plus_val_df[config.label_column].tolist())
    y_test = label_encoder.transform(test_df[config.label_column].tolist())

    svm = SVMClassifier(
        cv=config.svm_cv,
        scoring=config.svm_scoring,
        n_jobs=config.svm_n_jobs,
        verbose=config.svm_verbose,
    )
    svm.fit(X_train, y_train)

    results_df, y_pred = svm.evaluate(
        X_test,
        y_test,
        label_encoder=label_encoder,
        model_name="Light TF-IDF + Juliette SVM",
    )

    results_df.to_csv(output_dir / "final_test_results.csv", index=False, encoding="utf-8-sig")
    svm.top_grid_results(10).to_csv(output_dir / "top_grid_results.csv", index=False, encoding="utf-8-sig")
    audit_df.to_csv(output_dir / "sublabel_aware_sampling_audit.csv", index=False, encoding="utf-8-sig")

    predictions_df = test_df[
        [
            config.record_id_column,
            config.analysis_text_column,
            config.clean_text_column,
            config.label_column,
            config.sublabel_column,
        ]
    ].copy()
    predictions_df.rename(columns={config.label_column: "actual_umbrella"}, inplace=True)
    predictions_df["predicted_umbrella"] = label_encoder.inverse_transform(y_pred)
    predictions_df["is_correct"] = predictions_df["actual_umbrella"] == predictions_df["predicted_umbrella"]
    predictions_df.to_csv(output_dir / "final_test_predictions.csv", index=False, encoding="utf-8-sig")

    (output_dir / "best_svm_params.json").write_text(
        json.dumps(svm.best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "TF-IDF + Juliette SVM Summary",
        "================================",
        "",
        "Vectorizer:",
        "- Final accepted hybrid TF-IDF path from Jean's best TF-IDF experiment",
        f"- Word side: analysis_text, n-grams={config.word_ngram_range}, min_df={config.word_min_df}, max_features={config.word_max_features}",
        f"- Character side: clean_text, n-grams={config.char_ngram_range}, min_df={config.char_min_df}, max_features={config.char_max_features}",
        f"- Retained features in final hybrid matrix: {X_train.shape[1]:,}",
        "",
        "Data:",
        f"- Balanced rows: {len(balanced_df):,}",
        f"- Train: {len(train_df):,}",
        f"- Validation: {len(val_df):,}",
        f"- Test: {len(test_df):,}",
        f"- Final SVM training rows (Train + Validation): {len(train_plus_val_df):,}",
        "",
        "Classifier:",
        "- Juliette's SVMClassifier class, unchanged",
        f"- Best params: {svm.best_params}",
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
    svm.plot_confusion_matrix(y_test, y_pred, label_encoder=label_encoder)


if __name__ == "__main__":
    run_experiment()
