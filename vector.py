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
        c_range = [0.01, 0.1, 1, 10, 100, 1000]  # range of C to be tested
        degree_range = [1, 2, 3, 4, 5, 6]  # degrees to be tested

        params = {"predict__kernel": kernel_types,
                  "predict__C": c_range,
                  "predict__degree": degree_range,
                  }  # setting grid of parameters

        pipeline = Pipeline([('scale', StandardScaler()), ('predict', SVC())])
        learning_curve_train_sizes = np.arange(0.05, 1.0, 0.05)
        super().__init__(attributes, classifications, dataset, 'vector', pipeline, params,
                         learning_curve_train_sizes, True, verbose=0, iteration_curve=True)
