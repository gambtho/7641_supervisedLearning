"""Module documentation goes here
   and here
   and ...
"""

import logging
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


class Nearest(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        # weight_functions = ["uniform", "distance"]
        # p_values = [1, 2]
        # n_range = list(range(1, 51))
        #
        # param_grid = {"n_neighbors": n_range,
        #               "weights": weight_functions,
        #               "p": p_values
        #               }  # setting grid of parameters

        pipeline = Pipeline([('scale', StandardScaler()), ('predict', KNeighborsClassifier())])
        params = {
            'predict__metric': ['manhattan', 'euclidean', 'chebyshev'],
            'predict__n_neighbors': np.arange(1, 30, 3),
            'predict__weights': ['uniform', 'distance']
        }
        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        super().__init__(attributes, classifications, dataset, 'nearest', pipeline, params,
                         learning_curve_train_sizes, True)
