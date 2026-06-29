"""
SVM classifier for mental health text classification.
"""

from typing import Optional

from PIL.ImageOps import scale
import numpy as np
import pandas as pd
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ParameterGrid
import warnings
warnings.filterwarnings("ignore")


class SVMClassifier:
    """
    Support Vector Machine classifier with built-in GridSearch hyperparameter tuning.

    Attributes:
        best_model   : best SVC found by GridSearchCV
        best_params  : dict of best hyperparameters
        cv_results   : full GridSearchCV results dataframe
    """
    
    #Best params distilbert
    PARAM_GRID = {
        "kernel": ["rbf"],
        "C":      [1.0],
        "gamma":  ["scale"],
    }
    #best param tfidf
    # PARAM_GRID = {
    #     "kernel": ["sigmoid"],
    #     "C": [1.0],
    #     "gamma": ["scale"],
    # }
    #best params tfidf - overfit
    # PARAM_GRID = {
    #     "kernel": ["rbf"],
    #     "C": [1.0],
    #     "gamma": ["scale"],
    # }
    
    
    def __init__(self, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2):
        self.cv      = cv
        self.scoring = scoring
        self.n_jobs  = n_jobs
        self.verbose = verbose
        self.model: Optional[SVC] = None
        self.best_params: Optional[dict] = None
        self.cv_results: Optional[pd.DataFrame] = None

    # fit changed to train to match my code
    def train(self, X_train, y_train):
        """
        Run GridSearchCV over PARAM_GRID and store the best estimator.

        Args:
            X_train : feature matrix (dense or sparse)
            y_train : target labels
        """
        param_list = list(ParameterGrid(self.PARAM_GRID))
        print(f"Testing {len(param_list)} SVM parameter combinations:")
        for params in param_list:
            print(f"  {params}")

        grid_search = GridSearchCV(
            estimator=SVC(),
            param_grid=self.PARAM_GRID,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=3,
            return_train_score=True,
            refit=True,    # refit best model on full train set
        )
        grid_search.fit(X_train, y_train)
        self.model       = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results  = pd.DataFrame(grid_search.cv_results_)
        print(f"Best params: {self.best_params}")
        print(f"Best CV {self.scoring}: {grid_search.best_score_:.4f}")

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X : feature matrix
        Returns:
            np.ndarray of predicted labels
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .predict()")
        return self.model.predict(X)

    def top_grid_results(self, n=5):
        """
        Return top GridSearchCV results.
        """

        cols = [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]

        return (
            self.cv_results[cols]
            .sort_values("rank_test_score")
            .head(n)
            .reset_index(drop=True)
        )