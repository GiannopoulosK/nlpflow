import optuna
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import logging
optuna.logging.set_verbosity(logging.WARNING)
class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_model = None
        self.best_alpha = None
        self.best_score = 0

    def objective_multinomial(self, trial):
        alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        model = MultinomialNB(alpha=alpha)
        
        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy').mean()
        
        return score

    def objective_complement(self, trial):
        alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        model = ComplementNB(alpha=alpha)

        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy').mean()
        
        return score

    def trial_callback(self, study, trial):
        # Print status every 10 trials
        if trial.number % 10 == 0:
            print(f"Trial {trial.number} completed. Best value so far: {study.best_value}, Best params: {study.best_params}")
            
    def tune_models(self):
        # Tune MultinomialNB
        study_multinomial = optuna.create_study(direction="maximize")
        study_multinomial.optimize(self.objective_multinomial, n_trials=100, callbacks=[self.trial_callback])

        # Store best MultinomialNB model
        self.best_alpha = study_multinomial.best_params['alpha']
        self.best_model = MultinomialNB(alpha=self.best_alpha)

        # Train the best model
        self.best_model.fit(self.X_train, self.y_train)
        multinomial_accuracy = self.best_model.score(self.X_test, self.y_test)
        print(f"Best MultinomialNB alpha: {self.best_alpha}, Accuracy: {multinomial_accuracy:.4f}")

        # Now tune ComplementNB
        study_complement = optuna.create_study(direction="maximize")
        study_complement.optimize(self.objective_complement, n_trials=100, callbacks=[self.trial_callback])

        # Store best ComplementNB model
        complement_alpha = study_complement.best_params['alpha']
        complement_model = ComplementNB(alpha=complement_alpha)

        # Train the best ComplementNB model
        complement_model.fit(self.X_train, self.y_train)
        complement_accuracy = complement_model.score(self.X_test, self.y_test)
        print(f"Best ComplementNB alpha: {complement_alpha}, Accuracy: {complement_accuracy:.4f}")

        # Compare models and select the best one
        if multinomial_accuracy > complement_accuracy:
            print(f"MultinomialNB is the best model with alpha = {self.best_alpha} and accuracy score = {multinomial_accuracy}")
            return self.best_model
        else:
            print(f"ComplementNB is the best model with alpha = {complement_alpha} and accuracy score = {complement_accuracy}")
            return complement_model