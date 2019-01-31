import logging
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class Vector(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        self._csv_str = './results/{}/vector/'.format(dataset)

        # kernel_types = ["linear", "poly", "rbf", "sigmoid"]  # types of kernels to be tested
        # c_range = [0.01, 0.1, 1, 10, 100, 1000]  # range of C to be tested
        # degree_range = [1, 2, 3, 4, 5, 6]  # degrees to be tested

        kernel_types = ["rbf"]  # types of kernels to be tested
        c_range = [10, 100, 200]  # range of C to be tested
        # degree_range = [1, 3, 10]  # degrees to be tested

        params = {"predict__kernel": kernel_types,
                  "predict__C": c_range,
                  "predict__gamma": ['auto']
                  # "predict__degree": degree_range,
                  }  # setting grid of parameters

        pipeline = Pipeline([('scale', StandardScaler()), ('predict', SVC())])
        learning_curve_train_sizes = np.arange(0.05, 1.0, 0.05)
        super().__init__(attributes, classifications, dataset, 'vector', pipeline, params,
                         learning_curve_train_sizes, True, verbose=0, iteration_curve=True)

    def run(self):
        super().run()
        x_train, x_test, y_train, y_test = super().get_data_split()
        self.naive_report(x_test, x_train, y_test, y_train, self._csv_str)

    @staticmethod
    def naive_report(x_test, x_train, y_test, y_train, csv_str):
        _, ax = plt.subplots()
        visualizer = ClassificationReport(
            SVC(gamma='auto')
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/naive-classification.png".format(csv_str))

