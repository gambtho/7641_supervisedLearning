import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt


class Nearest(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        # 'predict__metric': ['manhattan', 'euclidean', 'chebyshev'],
        # 'predict__n_neighbors': [1, 3, 5, 9, 15],
        # 'predict__weights': ['uniform', 'distance'],
        # 'predict__leaf_size': [1, 2, 3, 5],
        # 'predict__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        self._csv_str = './results/{}/nearest/'.format(dataset)
        pipeline = Pipeline([('scale', StandardScaler()), ('predict', KNeighborsClassifier())])
        params = {
            'predict__metric': ['manhattan'],
            'predict__n_neighbors': [1, 3, 5],
            'predict__weights': ['uniform', 'distance'],
            'predict__leaf_size': [1, 3],
            'predict__algorithm': ['auto'],
        }
        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        super().__init__(attributes, classifications, dataset, 'nearest', pipeline, params,
                         learning_curve_train_sizes, True)

    @staticmethod
    def naive_report(x_test, x_train, y_test, y_train, csv_str):
        _, ax = plt.subplots()
        visualizer = ClassificationReport(
            KNeighborsClassifier()
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/naive-classification.png".format(csv_str))

        _, ax = plt.subplots()
        visualizer = ClassificationReport(
            KNeighborsClassifier(n_neighbors=10)
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/worst-classification.png".format(csv_str))

    def run(self):
        super().run()
        x_train, x_test, y_train, y_test = super().get_data_split()
        self.naive_report(x_test, x_train, y_test, y_train, self._csv_str)

