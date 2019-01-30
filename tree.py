from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiment import Experiment
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt
# from debug import Debug
# import logging
# logger = logging.getLogger(__name__)


class Tree(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        self._csv_str = './results/{}/tree/'.format(dataset)

        criteria = ["gini", "entropy"]  # criteria to be tested
        min_sample_split_range = [2, 10, 20]  # min sample split to be tested
        max_depth_range = [None, 2, 5, 10]  # max depth to be tested
        min_samples_leaf_range = [1, 2, 5, 10]  # min samples in the leaf to be tested
        min_leaf_nodes_range = [None, 5, 10, 20]  # min leaf nodes to be tested

        params = {"predict__criterion": criteria,
                  "predict__min_samples_split": min_sample_split_range,
                  "predict__max_depth": max_depth_range,
                  "predict__min_samples_leaf": min_samples_leaf_range,
                  "predict__max_leaf_nodes": min_leaf_nodes_range
                  }

        pipeline = Pipeline([('scale', StandardScaler()), ('predict', DecisionTreeClassifier())])
        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        super().__init__(attributes, classifications, dataset, 'tree', pipeline, params,
                         learning_curve_train_sizes, True, verbose=1)

    def run(self):
        super().run()
        x_train, x_test, y_train, y_test = super().get_data_split()
        self.naive_report(x_test, x_train, y_test, y_train)

        # self.info_print(x_test, x_train, y_test, y_train)

    def naive_report(self, x_test, x_train, y_test, y_train):
        _, ax = plt.subplots()
        visualizer = ClassificationReport(
            DecisionTreeClassifier(random_state=0, min_samples_leaf=10, min_samples_split=10), ax=ax
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/naive-classification.png".format(self._csv_str))








