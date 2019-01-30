import logging
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


class Neural(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('predict', MLPClassifier(random_state=10, max_iter=2000, early_stopping=True))])
        params = {
            'predict__activation': ['logistic', 'relu', 'tanh'],
            'predict__alpha': np.arange(0.0001, .05, 3, 0.1),
            'predict__hidden_layer_sizes': [(32, 64, 32), (64, 128, 64), (50,50,50), (50,100,50), (100,)],
            'predict__solver': ['sgd', 'adam'],
            'predict__learning_rate': ['constant', 'adaptive']
        }
        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        super().__init__(attributes, classifications, dataset, 'neural', pipeline, params,
                         learning_curve_train_sizes, True, verbose=1, iteration_curve=True)
