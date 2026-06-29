"""PyTorch Gaussian Naive Bayes model class.

Extracted from the final AraBERT/PCA + Gaussian Naive Bayes experiment.
This file contains only the reusable model class itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import torch


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

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_classes: int,
    ) -> "TorchGaussianNaiveBayes":
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

    def predict(self, X) -> torch.Tensor:
        if self.theta_ is None or self.var_ is None or self.class_log_prior_ is None:
            raise RuntimeError("Gaussian NB not fitted.")

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=self.dtype)

        X = X.detach().cpu().to(self.dtype)

        diff = X.unsqueeze(1) - self.theta_.unsqueeze(0)
        log_likelihood = -0.5 * (
            torch.log(2.0 * torch.tensor(math.pi, dtype=self.dtype) * self.var_).unsqueeze(0)
            + (diff * diff) / self.var_.unsqueeze(0)
        )

        scores = log_likelihood.sum(dim=2) + self.class_log_prior_.unsqueeze(0)
        return torch.argmax(scores, dim=1)

    @classmethod
    def load(cls, path: Path) -> "TorchGaussianNaiveBayes":
        data = torch.load(path, map_location="cpu")
        model = cls(
            var_smoothing=data["var_smoothing"],
            dtype=torch.float32,
        )
        model.epsilon_ = data["epsilon"]
        model.theta_ = data["theta"].to(model.dtype)
        model.var_ = data["var"].to(model.dtype)
        model.class_log_prior_ = data["class_log_prior"].to(model.dtype)
        model.class_count_ = data["class_count"]
        model.n_classes_ = data["n_classes"]
        model.n_features_ = data["n_features"]
        return model

    def save(self, path: Path) -> None:
        if self.theta_ is None or self.var_ is None or self.class_log_prior_ is None:
            raise RuntimeError("Cannot save unfitted Gaussian NB.")

        torch.save(
            {
                "var_smoothing": self.var_smoothing,
                "epsilon": self.epsilon_,
                "theta": self.theta_,
                "var": self.var_,
                "class_log_prior": self.class_log_prior_,
                "class_count": self.class_count_,
                "n_classes": self.n_classes_,
                "n_features": self.n_features_,
            },
            path,
        )
