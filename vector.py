"""Module documentation goes here
   and here
   and ...
"""

import logging
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class Vector(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):

        kernel_types = ["linear", "poly", "rbf", "sigmoid"]  # types of kernels to be tested
        C_range = [0.01, 0.1, 1, 10, 100, 1000]  # range of C to be tested
        degree_range = [1, 2, 3, 4, 5, 6]  # degrees to be tested

        params = {"predict__kernel": kernel_types,
                      "predict__C": C_range,
                      "predict__degree": degree_range,
                      }  # setting grid of parameters

        pipeline = Pipeline([('scale', StandardScaler()), ('predict', SVC())])
        # params = {
        #     'predict__kernel': ['linear', 'poly', 'rbf'],
        #     'predict__C': 10.0 ** np.arange(-3, 8),
        #     # penalize distance, low = use all, high = use close b/c distance to decision boundary to penalized
        #     'predict__gamma': 10. ** np.arange(-5, 4),
        #     'predict__cache_size': [200],
        #     'predict__max_iter': [3000],
        #     'predict__degree': [2, 3],
        #     'predict__coef0': [0, 1]
        # }
        learning_curve_train_sizes = np.arange(0.05, 1.0, 0.05)
        super().__init__(attributes, classifications, dataset, 'vector', pipeline, params,
                         learning_curve_train_sizes, True, verbose=0, iteration_curve=True)
