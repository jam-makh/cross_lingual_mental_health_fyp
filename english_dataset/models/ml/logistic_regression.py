"""
logistic_regression.py

Logistic Regression classifier.  Inherits cross-validation, evaluation,
and all metric logic from BaseModel — only train() and predict() live here.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
)


class LogisticRegressionModel():
    """
    Logistic Regression with GridSearchCV hyperparameter tuning.

    Receives pre-split, pre-vectorized arrays only.
    Splitting and vectorization are handled upstream in
    cache_embeddings.py to avoid recomputing expensive embeddings.

    Inherits from :class:`BaseModel`:
        - :meth:`cross_validate_model`
        - :meth:`evaluate_model`
        - :meth:`build_results_table`
        - :meth:`_compute_metrics`
        - :meth:`_print_metrics`
    """

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self.random_state  = random_state
        self.test_size     = test_size

        self.model: Optional[LogisticRegression] = None   # set by train()
        self.best_params_: Optional[Dict[str, Any]] = None
        self.grid_search:  Optional[GridSearchCV]   = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: str = "f1_macro",
        cv_splits: int = 5,
    ) -> None:
        """
        Fit with GridSearchCV and store the best estimator in ``self.model``.

        :param X_train: Training feature matrix.
        :type X_train: np.ndarray
        :param y_train: Training labels.
        :type y_train: np.ndarray
        :param scoring: GridSearch optimisation metric.
        :type scoring: str
        :param cv_splits: Inner CV folds for GridSearch.
        :type cv_splits: int
        :returns: None
        :rtype: None
        """
        base_model = LogisticRegression(
            max_iter=5000,
            random_state=self.random_state,
        )

        #param_grid = [
            # lbfgs: l2 or no penalty only
            #{
                #"solver":       ["lbfgs"],
                #"penalty":      ["l2", None],
                #"C":            [0.1, 1.0, 10.0],
                #"class_weight": [None, "balanced"],
            #},
            # saga: elasticnet
            #{
                #"solver":       ["saga"],
                #"penalty":      ["elasticnet"],
                #"C":            [0.1, 1.0, 10.0],
                #"l1_ratio":     [0.1, 0.5, 0.9],
                #"class_weight": [None, "balanced"],
            #},
            # liblinear: l1/l2 only
            #{
                #"solver":       ["liblinear"],
                #"penalty":      ["l1", "l2"],
                #"C":            [0.1, 1.0, 10.0],
                #"class_weight": [None, "balanced"],
            #},
        #]
        # param_grid = {
        #     "solver": ["saga"],
        #     "penalty": ["elasticnet"],
        #     "l1_ratio": [0.1, 0.5, 0.9],
        #     "C": [0.1, 1.0],
        #     "class_weight": [None, "balanced"]
        # }

        # param_grid = {
        #     "solver": ["saga"],
        #     "penalty": ["elasticnet"],
        #     "l1_ratio": [0.1],
        #     "C": [1.0],
        #     "class_weight": ["balanced"]
        # }
        param_grid = {
            "C": [0.1, 1.0],
            "class_weight": [None, "balanced"],
            "penalty": ["l2", None],
            "solver": ["lbfgs"]
        }
        # param_grid = {
        #     "C": [1.0],
        #     "class_weight": ["balanced"],
        #     "l1_ratio": [0.1],
        #     "penalty": ["elasticnet"],
        #     "solver": ["saga"]
        # }

        stratified_cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        param_list = list(ParameterGrid(param_grid))
        print(f"Testing {len(param_list)} parameter combinations:")
        for params in param_list:
            print(f"  {params}")

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=stratified_cv,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
            refit=True,    # refit best model on full train set
            error_score=0.0,
        )

        print("Running GridSearchCV for Logistic Regression...")
        grid_search.fit(X_train, y_train)

        self.grid_search  = grid_search
        self.model        = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        print(f"\nBest params  : {self.best_params_}")
        print(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the trained model.

        :param X: Feature matrix.
        :type X: np.ndarray
        :returns: Predicted label array.
        :rtype: np.ndarray
        :raises ValueError: If :meth:`train` has not been called.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        return self.model.predict(X)
