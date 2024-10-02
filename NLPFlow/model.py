from abc import ABC, abstractmethod
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import logging
optuna.logging.set_verbosity(logging.WARNING)

class ModelStrategy(ABC):
    @abstractmethod
    def tune_model(self, X_train, y_train, n_trials):
        pass

    @abstractmethod
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        pass
    
class MultinomialNBStrategy(ModelStrategy):
    def __init__(self):
        self.best_params = None
        self.best_model = None

    def objective(self, trial, X_train, y_train):
        alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        model = MultinomialNB(alpha=alpha)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    def tune_model(self, X_train, y_train, n_trials=100, train_best=True):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        self.best_params = study.best_params
        self.best_model = MultinomialNB(alpha=self.best_params["alpha"])
        if train_best:
            self.best_model.fit(X_train, y_train)
        return self.best_model

class ComplementNBStrategy(ModelStrategy):
    def __init__(self):
        self.best_params = None
        self.best_model = None

    def objective(self, trial, X_train, y_train):
        alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        model = ComplementNB(alpha=alpha)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    def tune_model(self, X_train, y_train, n_trials=100, train_best=True):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        self.best_params = study.best_params
        self.best_model = ComplementNB(alpha=self.best_params["alpha"])
        if train_best:
            self.best_model.fit(X_train, y_train)
        return self.best_model
    
class LogisticRegressionStrategy(ModelStrategy):
    def __init__(self):
        self.best_params = None
        self.best_model = None

    def objective(self, trial, X_train, y_train):
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        
        if penalty == 'l1':
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
        else:
            solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
            
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, tol=tol)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    def tune_model(self, X_train, y_train, n_trials=100, train_best=True):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        self.best_params = study.best_params
        
        self.best_model = LogisticRegression(
            C=self.best_params["C"],
            penalty=self.best_params["penalty"],
            solver=self.best_params["solver"],
            max_iter=self.best_params["max_iter"],
            tol=self.best_params["tol"]
        )
        if train_best:
            self.best_model.fit(X_train, y_train)
        return self.best_model
    
class RandomForestStrategy(ModelStrategy):
    def __init__(self):
        self.best_params = None
        self.best_model = None

    def objective(self, trial, X_train, y_train):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 2, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    def tune_model(self, X_train, y_train, n_trials=100, train_best=True):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        self.best_params = study.best_params
        
        self.best_model = RandomForestClassifier(
            n_estimators=self.best_params["n_estimators"],
            max_depth=self.best_params["max_depth"],
            min_samples_split=self.best_params["min_samples_split"],
            min_samples_leaf=self.best_params["min_samples_leaf"],
            max_features=self.best_params["max_features"],
            random_state=42
        )
        
        if train_best:
            self.best_model.fit(X_train, y_train)
        return self.best_model

class SVMStrategy(ModelStrategy):
    def __init__(self):
        self.best_params = None
        self.best_model = None

    def objective(self, trial, X_train, y_train):
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
        else:
            degree = 3
        
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=42)

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    def tune_model(self, X_train, y_train, n_trials=100, train_best=True):
        study = optuna.create_study(direction="maximize")
        
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_model = SVC(
            C=self.best_params["C"],
            kernel=self.best_params["kernel"],
            degree=self.best_params.get("degree", 3),
            gamma=self.best_params["gamma"],
            random_state=42
        )
        
        if train_best:
            self.best_model.fit(X_train, y_train)
        
        return self.best_model

class KNNStrategy(ModelStrategy):
    def __init__(self):
        self.best_params = None
        self.best_model = None

    def objective(self, trial, X_train, y_train):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)

        # Building the model
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p
        )
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    def tune_model(self, X_train, y_train, n_trials=100, train_best=True):
        study = optuna.create_study(direction="maximize")
        
        study.optimize(lambda trial: self.objective(trial, X_train, y_train), n_trials=n_trials)
        
        self.best_params = study.best_params
        
        self.best_model = KNeighborsClassifier(
            n_neighbors=self.best_params["n_neighbors"],
            weights=self.best_params["weights"],
            p=self.best_params["p"]
        )
        
        if train_best:
            self.best_model.fit(X_train, y_train)
        
        return self.best_model
    
class Classifier:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        
        self.model_registry = {
            "MultinomialNB": MultinomialNBStrategy,
            "ComplementNB": ComplementNBStrategy,
            "LogisticRegression": LogisticRegressionStrategy(),
            "RandomForest": RandomForestStrategy(),
            "SVC": SVMStrategy(),
            "KNeighbors": KNNStrategy()
        }

    def tune_and_select_best(self, model_names, n_trials=100):
        model_accuracies = {}

        for model_name in model_names:
            if model_name in self.model_registry:
                strategy = self.model_registry[model_name]()
                print(f"Tuning {model_name}...")
                strategy.tune_model(self.X_train, self.y_train, n_trials=n_trials)
                accuracy, model = strategy.train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test)
                print(f"{model_name} accuracy: {accuracy:.4f}")
                model_accuracies[model_name] = (accuracy, model)
            else:
                print(f"Model {model_name} is not recognized. Available models: {list(self.model_registry.keys())}")

        if model_accuracies:
            best_model_name = max(model_accuracies, key=lambda name: model_accuracies[name][0])
            self.best_model = model_accuracies[best_model_name][1]
            print(f"The best model is {best_model_name} with accuracy {model_accuracies[best_model_name][0]:.4f}")
            return self.best_model
        else:
            print("No valid models were selected.")
            return None