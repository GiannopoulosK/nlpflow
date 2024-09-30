import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from nlpflow.model import ModelTrainer

# Sample mock data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 10)
y_test = np.random.randint(0, 2, 20)

@pytest.fixture
def model_trainer():
    return ModelTrainer(X_train, y_train, X_test, y_test)

@patch('optuna.create_study')
def test_tune_models(mock_create_study, model_trainer):
    mock_study = MagicMock()
    mock_create_study.return_value = mock_study
    mock_study.best_params = {'alpha': 1.0}
    mock_study.best_value = 0.9

    with patch('sklearn.naive_bayes.MultinomialNB'), patch('sklearn.naive_bayes.ComplementNB'):
        model_trainer.tune_models()

    # Assert that the best model was trained
    assert model_trainer.best_model is not None
    assert model_trainer.best_alpha == 1.0
    assert isinstance(model_trainer.best_model, MultinomialNB)

@patch('optuna.create_study')
def test_objective_multinomial(mock_create_study, model_trainer):
    # Create a mock trial
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 1.0

    # Mock the score method of the model
    with patch('sklearn.model_selection.cross_val_score', return_value=np.array([0.8, 0.9, 0.85, 0.9, 0.95])):
        score = model_trainer.objective_multinomial(mock_trial)

@patch('optuna.create_study')
def test_objective_complement(mock_create_study, model_trainer):
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 1.0

    with patch('sklearn.model_selection.cross_val_score', return_value=np.array([0.7, 0.75, 0.8, 0.85, 0.9])):
        score = model_trainer.objective_complement(mock_trial)

@patch('optuna.create_study')
def test_trial_callback(mock_create_study, model_trainer):
    mock_study = MagicMock()
    mock_trial = MagicMock(number=10, best_value=0.8, best_params={'alpha': 2.0})
    model_trainer.trial_callback(mock_study, mock_trial)

    mock_study.best_value = 0.8
    mock_study.best_params = {'alpha': 2.0}