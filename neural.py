import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt


class Neural(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        # 'predict__activation': ['logistic', 'relu', 'tanh'],
        # 'predict__alpha': [.0001, .01, 1],
        # 'predict__hidden_layer_sizes': [(32, 64, 32), (64, 128, 64), (50, 50, 50), (50, 100, 50), (100,)],
        # 'predict__solver': ['sgd', 'adam'],
        # 'predict__learning_rate': ['constant', 'adaptive']
        self._csv_str = './results/{}/neural/'.format(dataset)
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('predict', MLPClassifier(random_state=10, max_iter=2000, early_stopping=True))])
        params = {
            'predict__activation': ['relu'],
            'predict__alpha': [.0001, .001, .01],
            'predict__hidden_layer_sizes': [(32, 64, 32), (64, 128, 64)],
            'predict__solver': ['adam'],
            'predict__learning_rate': ['constant']
        }
        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        super().__init__(attributes, classifications, dataset, 'neural', pipeline, params,
                         learning_curve_train_sizes, True, verbose=1, iteration_curve=True)

    def run(self):
        super().run()
        x_train, x_test, y_train, y_test = super().get_data_split()
        self.naive_report(x_test, x_train, y_test, y_train, self._csv_str)

    @staticmethod
    def naive_report(x_test, x_train, y_test, y_train, csv_str):
        _, ax = plt.subplots()
        visualizer = ClassificationReport(
            MLPClassifier(max_iter=100)
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/naive-classification.png".format(csv_str))

