"""PyTorch Multinomial Naive Bayes model class.

Extracted from the final TF-IDF + Multinomial Naive Bayes experiment.
This file contains only the reusable model class itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TorchMultinomialNaiveBayes:
    """Multinomial NB for sparse non-negative TF-IDF-like inputs."""

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

    def fit(
        self,
        X_sparse: torch.Tensor,
        y: torch.Tensor,
        n_classes: int,
    ) -> "TorchMultinomialNaiveBayes":
        if not X_sparse.is_sparse:
            raise ValueError("X_sparse must be sparse COO.")
        if X_sparse.size(0) != y.numel():
            raise ValueError("X rows and y length do not match.")

        X_sparse = X_sparse.coalesce().to(self.device)
        y = y.to(self.device)

        self.n_classes_ = int(n_classes)
        self.n_features_ = int(X_sparse.size(1))

        class_count = torch.bincount(y, minlength=self.n_classes_).to(dtype=self.dtype)
        if torch.any(class_count == 0):
            raise ValueError("A class has zero training samples.")

        feature_count = torch.zeros(
            (self.n_classes_, self.n_features_),
            dtype=self.dtype,
            device=self.device,
        )

        idx = X_sparse.indices()
        doc_idx = idx[0]
        feat_idx = idx[1]
        vals = X_sparse.values().to(dtype=self.dtype)
        cls_idx = y[doc_idx]

        flat = cls_idx * self.n_features_ + feat_idx
        feature_count.view(-1).scatter_add_(0, flat, vals)

        smoothed = feature_count + self.alpha
        totals = smoothed.sum(dim=1, keepdim=True)

        self.feature_log_prob_ = torch.log(smoothed) - torch.log(totals)
        self.class_log_prior_ = torch.log(class_count) - torch.log(class_count.sum())
        self.class_count_ = class_count

        return self

    def predict(self, X_sparse) -> torch.Tensor:
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Model has not been fitted.")

        if not isinstance(X_sparse, torch.Tensor):
            try:
                from scipy.sparse import spmatrix
            except ImportError:
                raise ValueError("scipy is required to predict from sparse matrices.")

            if isinstance(X_sparse, spmatrix):
                X_coo = X_sparse.tocoo()
                indices = torch.LongTensor(np.vstack((X_coo.row, X_coo.col)))
                values = torch.FloatTensor(X_coo.data)
                X_sparse = torch.sparse_coo_tensor(
                    indices,
                    values,
                    X_coo.shape,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                raise ValueError(
                    "X_sparse must be a torch sparse tensor or scipy sparse matrix."
                )

        X_sparse = X_sparse.coalesce().to(self.device)
        scores = torch.sparse.mm(X_sparse, self.feature_log_prob_.T)
        scores = scores + self.class_log_prior_.unsqueeze(0)

        return torch.argmax(scores, dim=1)

    @classmethod
    def load(cls, path: Path) -> "TorchMultinomialNaiveBayes":
        data = torch.load(path, map_location="cpu")
        model = cls(
            alpha=data["alpha"],
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.class_log_prior_ = data["class_log_prior"].to(model.device)
        model.feature_log_prob_ = data["feature_log_prob"].to(model.device)
        model.class_count_ = (
            data["class_count"].to(model.device)
            if data["class_count"] is not None
            else None
        )
        model.n_classes_ = data["n_classes"]
        model.n_features_ = data["n_features"]
        return model

    def save(self, path: Path) -> None:
        if self.class_log_prior_ is None or self.feature_log_prob_ is None:
            raise RuntimeError("Cannot save unfitted model.")

        torch.save(
            {
                "alpha": self.alpha,
                "class_log_prior": self.class_log_prior_.detach().cpu(),
                "feature_log_prob": self.feature_log_prob_.detach().cpu(),
                "class_count": (
                    self.class_count_.detach().cpu()
                    if self.class_count_ is not None
                    else None
                ),
                "n_classes": self.n_classes_,
                "n_features": self.n_features_,
            },
            path,
        )
