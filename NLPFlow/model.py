# Design Patterns
from abc import ABC, abstractmethod

# Hyperparameter Tuning
import optuna

# Logging
import logging
optuna.logging.set_verbosity(logging.WARNING)
logging.basicConfig(level=logging.INFO)
from sklearn.metrics import accuracy_score
# Type Hinting
from typing import Union, List, Dict
from numpy import ndarray

# Models
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
    
# Misc
import joblib
import pickle
import numpy as np

class ModelStrategy(ABC):
    @abstractmethod
    def tune_model(self, X_train: ndarray, y_train: ndarray, param_ranges: Union[str, Dict[str, float]], 
                   scoring: Union[str, List[str]], n_trials: int, cross_validation: bool, cv: int) -> object:
        """
        Abstract method to tune the model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            object: Tuned model
        """
        pass
    
    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        """
        Abstract method to make predictions.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: ndarray) -> ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> ndarray:
        pass
    
class MultinomialNBStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="largest", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for MultinomialNB.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": (1e-3, 10.0),
            "large": (1e-2, 8.0),
            "medium": (1e-1, 5.0),
            "small": (0, 1),
            "smallest": (0, 0.5)
        }


        # Determine if the param_ranges is a preset string or a dictionary
        if isinstance(param_ranges, str):
            # Use preset alpha ranges if param_ranges is a string
            alpha_range = param_presets.get(param_ranges)
            alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        elif isinstance(param_ranges, dict):
            # Use custom alpha from param_ranges dictionary
            alpha = trial.suggest_float("alpha", param_ranges.get("alpha", 1e-1), param_ranges.get("alpha_max", 5.0), log=True)
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")

        # Create the model with the suggested alpha
        model = MultinomialNB(alpha=alpha)

        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")

        # Perform cross-validation or direct model evaluation
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> MultinomialNB:
        """
        Tune the Multinomial Naive Bayes model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            MultinomialNB: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")

        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges,
                                                    cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)

        # Train the best model with the optimal parameters
        self.best_model = MultinomialNB(**self.best_params)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")

        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet or if X has an incorrect shape.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances (feature log probabilities for MultinomialNB).

        Returns:
            ndarray: Feature log probabilities.

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if self.best_model is None:
            logging.warning("Get_feature_importance called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        return self.best_model.feature_log_prob_
    


class ComplementNBStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="largest", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for ComplementNB.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": (1e-3, 10.0),
            "large": (1e-2, 8.0),
            "medium": (1e-1, 5.0),
            "small": (0, 1),
            "smallest": (0, 0.5)
        }
        


        # Determine if param_ranges is a preset string or a dictionary
        if isinstance(param_ranges, str):
            alpha_range = param_presets.get(param_ranges)
            alpha = trial.suggest_float("alpha", *alpha_range, log=True)
        elif isinstance(param_ranges, dict):
            alpha = trial.suggest_float("alpha", param_ranges.get("alpha", 1e-1), param_ranges.get("alpha_max", 5.0), log=True)
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")

        # Create the ComplementNB model with the suggested alpha
        model = ComplementNB(alpha=alpha)

        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")

        # Perform cross-validation or direct model evaluation
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> ComplementNB:
        """
        Tune the Complement Naive Bayes model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            ComplementNB: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")

        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges,
                                                    cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)

        # Train the best model with the optimal parameters
        self.best_model = ComplementNB(**self.best_params)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")

        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet or if X has an incorrect shape.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances (feature log probabilities for ComplementNB).

        Returns:
            ndarray: Feature log probabilities.

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if self.best_model is None:
            logging.warning("Get_feature_importance called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        return self.best_model.feature_log_prob_
    
class LogisticRegressionStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="largest", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for Logistic Regression.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": (1e-3, 10.0),
            "large": (1e-2, 8.0),
            "medium": (1e-1, 5.0),
            "small": (0.01, 1.0),
            "smallest": (0.001, 0.1)
        }
        


        # Determine if param_ranges is a preset string or a dictionary
        if isinstance(param_ranges, str):
            C_range = param_presets.get(param_ranges)
            C = trial.suggest_float("C", *C_range, log=True)
        elif isinstance(param_ranges, dict):
            C = trial.suggest_float("C", param_ranges.get('C', 1e-1), param_ranges.get('C_max', 10.0), log=True)
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")
        
        # Penalty and solver parameters
        penalty = trial.suggest_categorical('penalty', ['l2', 'none'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        
        if penalty == 'none' and solver == 'saga':
            # 'none' penalty is not allowed with 'saga', so we ignore this combination.
            raise optuna.TrialPruned()

        # Create the LogisticRegression model with the suggested parameters
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
        
        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")

        # Perform cross-validation or direct model evaluation
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> LogisticRegression:
        """
        Tune the Logistic Regression model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            LogisticRegression: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")
            
        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges,
                                                    cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)
        
        # Train the best model with the optimal parameters
        self.best_model = LogisticRegression(**self.best_params, max_iter=1000)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")
        
        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet or if X has an incorrect shape.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        return self.best_model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances (coefficients for LogisticRegression).

        Returns:
            ndarray: Feature coefficients.

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if self.best_model is None:
            logging.warning("Get_feature_importance called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")
        
        return self.best_model.coef_
    
class RandomForestStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="medium", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for RandomForestClassifier.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": {
                "n_estimators": (100, 500),
                "max_depth": (20, 50),
                "min_samples_split": (2, 10),
                "max_features": ["sqrt", "log2", None]
            },
            "large": {
                "n_estimators": (100, 300),
                "max_depth": (15, 40),
                "min_samples_split": (2, 8),
                "max_features": ["sqrt", "log2"]
            },
            "medium": {
                "n_estimators": (50, 200),
                "max_depth": (10, 30),
                "min_samples_split": (2, 6),
                "max_features": ["sqrt", "log2"]
            },
            "small": {
                "n_estimators": (10, 100),
                "max_depth": (5, 20),
                "min_samples_split": (2, 4),
                "max_features": ["sqrt"]
            },
            "smallest": {
                "n_estimators": (5, 50),
                "max_depth": (3, 10),
                "min_samples_split": (2, 4),
                "max_features": ["sqrt"]
            }
        }

        # If param_ranges is a string, use the preset ranges from param_presets
        if isinstance(param_ranges, str):
            param_preset = param_presets.get(param_ranges, param_presets['medium'])
        # If param_ranges is a dict, use custom ranges
        elif isinstance(param_ranges, dict):
            param_preset = param_ranges
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")

        # Suggest hyperparameters using the param_preset
        n_estimators = trial.suggest_int("n_estimators", *param_preset["n_estimators"])
        max_depth = trial.suggest_int("max_depth", *param_preset["max_depth"])
        min_samples_split = trial.suggest_int("min_samples_split", *param_preset["min_samples_split"])
        max_features = trial.suggest_categorical("max_features", param_preset["max_features"])

        # Define the RandomForestClassifier with the selected hyperparameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42
        )

        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")
        
        # Perform cross-validation or direct training
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> RandomForestClassifier:
        """
        Tune the RandomForestClassifier model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            RandomForestClassifier: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")

        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges, cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)

        self.best_model = RandomForestClassifier(**self.best_params, random_state=42)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")

        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet or if X has an incorrect shape.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances from the RandomForestClassifier.

        Returns:
            ndarray: Feature importances.

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if self.best_model is None:
            logging.warning("Get_feature_importance called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        return self.best_model.feature_importances_

class KNNStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="medium", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for KNeighborsClassifier.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": {
                "n_neighbors": (10, 50),
                "leaf_size": (20, 50),
                "p": [1, 2],
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            "large": {
                "n_neighbors": (10, 40),
                "leaf_size": (20, 40),
                "p": [1, 2],
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            "medium": {
                "n_neighbors": (5, 30),
                "leaf_size": (20, 30),
                "p": [1, 2],
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'kd_tree', 'ball_tree']
            },
            "small": {
                "n_neighbors": (3, 20),
                "leaf_size": (20, 30),
                "p": [1, 2],
                "weights": ['uniform', 'distance'],
                "algorithm": ['auto', 'kd_tree']
            },
            "smallest": {
                "n_neighbors": (1, 10),
                "leaf_size": (10, 30),
                "p": [1],
                "weights": ['uniform'],
                "algorithm": ['auto']
            }
        }

        # If param_ranges is a string, use the preset ranges from param_presets
        if isinstance(param_ranges, str):
            param_preset = param_presets.get(param_ranges, param_presets['medium'])
        # If param_ranges is a dict, use custom ranges
        elif isinstance(param_ranges, dict):
            param_preset = param_ranges
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")

        # Suggest hyperparameters using the param_preset
        n_neighbors = trial.suggest_int("n_neighbors", *param_preset["n_neighbors"])
        leaf_size = trial.suggest_int("leaf_size", *param_preset["leaf_size"])
        p = trial.suggest_categorical("p", param_preset["p"])
        weights = trial.suggest_categorical("weights", param_preset["weights"])
        algorithm = trial.suggest_categorical("algorithm", param_preset["algorithm"])

        # Define the KNeighborsClassifier with the selected hyperparameters
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            p=p,
            weights=weights,
            algorithm=algorithm
        )

        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")
        
        # Perform cross-validation or direct training
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> KNeighborsClassifier:
        """
        Tune the KNeighborsClassifier model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            KNeighborsClassifier: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")

        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges, cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)

        self.best_model = KNeighborsClassifier(**self.best_params)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")

        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet or if X has an incorrect shape.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict_proba(X)

    def get_feature_importance(self) -> None:
        """
        KNeighborsClassifier does not support feature importances.
        This method will raise an exception if called.
        
        Raises:
            NotImplementedError: KNeighborsClassifier does not support feature importances.
        """
        raise NotImplementedError("KNeighborsClassifier does not support feature importances.")


class SVMStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="medium", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for SVC.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": {
                "C": (0.001, 1000),
                "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                "degree": (2, 6),  # Only for 'poly' kernel
                "gamma": ['scale', 'auto'],
            },
            "large": {
                "C": (0.01, 500),
                "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                "degree": (2, 5),
                "gamma": ['scale', 'auto'],
            },
            "medium": {
                "C": (0.1, 100),
                "kernel": ['linear', 'poly', 'rbf'],
                "degree": (2, 4),
                "gamma": ['scale', 'auto'],
            },
            "small": {
                "C": (0.1, 50),
                "kernel": ['linear', 'rbf'],
                "degree": (2, 3),
                "gamma": ['scale', 'auto'],
            },
            "smallest": {
                "C": (0.5, 10),
                "kernel": ['linear', 'rbf'],
                "degree": (2, 3),
                "gamma": ['scale'],
            }
        }

        # If param_ranges is a string, use the preset ranges from param_presets
        if isinstance(param_ranges, str):
            param_preset = param_presets.get(param_ranges, param_presets['medium'])
        # If param_ranges is a dict, use custom ranges
        elif isinstance(param_ranges, dict):
            param_preset = param_ranges
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")

        # Suggest hyperparameters using the param_preset
        C = trial.suggest_float("C", *param_preset["C"], log=True)
        kernel = trial.suggest_categorical("kernel", param_preset["kernel"])

        # If the kernel is 'poly', suggest degree, otherwise set it to None
        if kernel == "poly":
            degree = trial.suggest_int("degree", *param_preset["degree"])
        else:
            degree = None

        # Suggest gamma only for 'rbf', 'poly', and 'sigmoid' kernels
        if kernel in ['rbf', 'poly', 'sigmoid']:
            gamma = trial.suggest_categorical("gamma", param_preset["gamma"])
        else:
            gamma = None

        # Define the SVC with the selected hyperparameters
        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree if kernel == "poly" else 3,
            gamma=gamma,
            random_state=42
        )

        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")
        
        # Perform cross-validation or direct training
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> SVC:
        """
        Tune the SVC model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            SVC: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")

        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges, cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)

        self.best_model = SVC(**self.best_params, random_state=42)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")

        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        SVC does not support predict_proba for non-probabilistic kernels.
        Raise an exception if called.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet.
            NotImplementedError: SVC does not support probabilities for all kernels.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if self.best_model.kernel not in ['rbf', 'poly', 'sigmoid']:
            raise NotImplementedError(f"SVC with {self.best_model.kernel} kernel does not support predict_proba")

        return self.best_model.predict_proba(X)

    def get_feature_importance(self):
        """
        SVC does not support feature importances natively.
        This method will raise an exception if called.

        Raises:
            NotImplementedError: SVC does not support feature importances.
        """
        raise NotImplementedError("SVC does not support feature importances.")
    
class LGBMStrategy:

    def __init__(self):
        self.best_params = None
        self.best_model = None
        self.supported_metrics = ['accuracy', 'precision', 'recall', 'f1']

    def objective(self, trial, X_train, y_train, scoring, param_ranges="medium", cross_validation=False, cv=5):
        """
        Objective function for hyperparameter optimization for LGBMClassifier.

        Args:
            trial: Optuna trial object
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            scoring (Union[str, List[str]]): Scoring metric(s)
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            float: Overall score for optimization
        """
        param_presets = {
            "largest": {
                "n_estimators": (100, 5000),
                "learning_rate": (0.0001, 0.5),
                "num_leaves": (15, 150),
                "max_depth": (-1, 30),  # -1 means no limit
                "min_child_samples": (10, 100),
                "subsample": (0.5, 1.0),
                "colsample_bytree": (0.5, 1.0)
            },
            "large": {
                "n_estimators": (100, 2000),
                "learning_rate": (0.001, 0.3),
                "num_leaves": (20, 100),
                "max_depth": (-1, 25),
                "min_child_samples": (10, 80),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0)
            },
            "medium": {
                "n_estimators": (100, 1000),
                "learning_rate": (0.01, 0.2),
                "num_leaves": (20, 80),
                "max_depth": (-1, 20),
                "min_child_samples": (15, 70),
                "subsample": (0.7, 1.0),
                "colsample_bytree": (0.7, 1.0)
            },
            "small": {
                "n_estimators": (100, 500),
                "learning_rate": (0.01, 0.1),
                "num_leaves": (30, 60),
                "max_depth": (5, 15),
                "min_child_samples": (20, 50),
                "subsample": (0.8, 1.0),
                "colsample_bytree": (0.8, 1.0)
            },
            "smallest": {
                "n_estimators": (100, 300),
                "learning_rate": (0.05, 0.1),
                "num_leaves": (31, 50),
                "max_depth": (6, 10),
                "min_child_samples": (20, 40),
                "subsample": (0.9, 1.0),
                "colsample_bytree": (0.9, 1.0)
            }
        }

        # If param_ranges is a string, use the preset ranges from param_presets
        if isinstance(param_ranges, str):
            param_preset = param_presets.get(param_ranges, param_presets['medium'])
        # If param_ranges is a dict, use custom ranges
        elif isinstance(param_ranges, dict):
            param_preset = param_ranges
        else:
            raise ValueError("Invalid argument: param_ranges should be a string or dictionary.")

        # Suggest hyperparameters using the param_preset
        n_estimators = trial.suggest_int("n_estimators", *param_preset["n_estimators"])
        learning_rate = trial.suggest_float("learning_rate", *param_preset["learning_rate"], log=True)
        num_leaves = trial.suggest_int("num_leaves", *param_preset["num_leaves"])
        max_depth = trial.suggest_int("max_depth", *param_preset["max_depth"])
        min_child_samples = trial.suggest_int("min_child_samples", *param_preset["min_child_samples"])
        subsample = trial.suggest_float("subsample", *param_preset["subsample"])
        colsample_bytree = trial.suggest_float("colsample_bytree", *param_preset["colsample_bytree"])

        # Define the LGBMClassifier with the selected hyperparameters
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1
        )

        # Check supported metrics
        metrics = [scoring] if isinstance(scoring, str) else scoring
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}. Supported metrics are {self.supported_metrics}")
        
        # Perform cross-validation or direct training
        if cross_validation:
            if len(metrics) > 1:
                scores = {}
                for metric in metrics:
                    scores[metric] = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric).mean()
                overall_score = sum(scores.values()) / len(scores)
            else:
                overall_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metrics[0]).mean()
        else:
            overall_score = model.fit(X_train, y_train).score(X_train, y_train)

        return overall_score

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, param_ranges: Union[str, Dict[str, float]] = "medium",
                   scoring: Union[str, List[str]] = 'accuracy', n_trials: int = 100, cross_validation: bool = False, cv: int = 5) -> LGBMClassifier:
        """
        Tune the LGBMClassifier model.

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            param_ranges (Union[str, Dict[str, float]]): Parameter ranges for tuning
            scoring (Union[str, List[str]]): Scoring metric(s)
            n_trials (int): Number of optimization trials
            cross_validation (bool): Whether to use cross-validation
            cv (int): Number of cross-validation folds

        Returns:
            LGBMClassifier: Tuned model
        """
        if isinstance(scoring, list):
            logging.warning("Multiple metrics specified. Will optimize based on average score.")

        logging.info("Starting model tuning with %d trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, scoring=scoring, param_ranges=param_ranges, cross_validation=cross_validation, cv=cv), n_trials=n_trials)
        self.best_params = study.best_params
        logging.info("Best parameters found: %s", self.best_params)

        self.best_model = LGBMClassifier(**self.best_params, random_state=42, n_jobs=-1)
        self.best_model.fit(X_train, y_train)
        logging.info("Best model trained successfully")

        return self.best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the tuned model.

        Args:
            X (ndarray): Features to predict on

        Returns:
            ndarray: Predictions

        Raises:
            ValueError: If the model hasn't been trained yet
            ValueError: If X has an incorrect shape
        """
        if self.best_model is None:
            logging.warning("Predict called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.

        Args:
            X (ndarray): The input samples.

        Returns:
            ndarray: The class probabilities of the input samples.

        Raises:
            ValueError: If the model hasn't been trained yet or if X has an incorrect shape.
        """
        if self.best_model is None:
            logging.warning("Predict_proba called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        return self.best_model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the best model.

        Returns:
            ndarray: Feature importances

        Raises:
            ValueError: If the model hasn't been trained yet
        """
        if self.best_model is None:
            logging.warning("get_feature_importance called before model was trained. Call tune_model first.")
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")

        return self.best_model.feature_importances_
            
class ModelTrainer:
    def __init__(self, X_train, X_test, y_train,  y_test, task="classification"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None
        
        if task == "classification":
            self.model_registry = {
                "MultinomialNB": MultinomialNBStrategy(),
                "ComplementNB": ComplementNBStrategy(),
                "LogisticRegression": LogisticRegressionStrategy(),
                "RandomForest": RandomForestStrategy(),
                "SVC": SVMStrategy(),
                "KNeighbors": KNNStrategy(),
                "LightGBM": LGBMStrategy()
            }
        elif task == "regression":
            pass
        elif task == "clustering":
            pass
        else:
            pass

    def tune_and_select_best(self, model_names,  param_ranges="medium", scoring='accuracy', n_trials=100, cross_validation=False):
        model_performances = {}
        if not isinstance(model_names, list):
            model_names = [model_names]
        for model_name in model_names:
            if model_name in self.model_registry:
                model_strategy = self.model_registry[model_name]
                print(f"Tuning {model_name}...")
                model = model_strategy.tune_model(self.X_train, self.y_train, param_ranges=param_ranges, scoring=scoring, n_trials=n_trials, cross_validation=cross_validation)
                performance = self.evaluate_model(model, self.X_test, self.y_test, {'accuracy': accuracy_score})
                print(f"{model_name} {scoring}: {performance[scoring]:.4f}")
                model_performances[model_name] = (performance[scoring], model)
            else:
                print(f"Model {model_name} is not recognized. Available models: {list(self.model_registry.keys())}")

        if model_performances:
            best_model_name = max(model_performances, key=lambda name: model_performances[name][0])
            self.best_model = model_performances[best_model_name][1]
            print(f"The best model is {best_model_name} with {scoring} {model_performances[best_model_name][0]:.4f}")
            return self.best_model
        else:
            print("No valid models were selected.")
            return None
        
    def evaluate_model(self,model, X_val, y_val, metrics):
        if self.best_model is None:
            raise ValueError("Model hasn't been trained yet. Call tune_model first.")
        
        results = {}
        if isinstance(metrics, dict):
            for metric_name, metric_func in metrics.items():
                results[metric_name] = metric_func(y_val, self.best_model.predict(X_val))
        else:
            raise ValueError("metrics should be a dictionary: {metric_name: metric_function}")
        return results
        
    def save_model(self, model, filepath, use="joblib"):
        if use == "joblib":
            joblib.dump(model, filepath)
        elif use == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError("Invalid Argument. Specify 'joblib' or 'pickle' for saving")
            
    def load_model(self, filepath, use="joblib"):
        if use == "joblib":
            return joblib.load(filepath)
        elif use == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Invalid Argument. Specify 'joblib' or 'pickle' for loading")

    def save_best_model(self, filepath, use="joblib"):
        if self.best_model is None:
            raise ValueError("No best model has been selected. Run tune_and_select_best first.")
        self.save_model(self.best_model, filepath, use)

    def load_best_model(self, filepath, use="joblib"):
        self.best_model = self.load_model(filepath, use)