import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiment import Experiment
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ClassificationReport
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class Tree(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        self._attributes = attributes
        self._classifications = classifications
        self._dataset = dataset
        self._csv_str = './results/{}/tree/'.format(self._dataset)

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
        cv = super().run()
        x_train, x_test, y_train, y_test = super().get_data_split()
        # self.info_print(x_test, x_train, y_test, y_train)
        self.naive_report(x_test, x_train, y_test, y_train)

    def naive_report(self, x_test, x_train, y_test, y_train):
        _, ax = plt.subplots()
        visualizer = ClassificationReport(
            DecisionTreeClassifier(random_state=0, min_samples_leaf=2), ax=ax
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/naive-classification.png".format(self._csv_str))

    def info_print(self, x_test, x_train, y_test, y_train):
        clf_gini = self.train_using_gini(x_train, y_train)
        clf_entropy = self.train_using_entropy(x_train, y_train)
        print("Results Using Gini Index:")
        y_pred_gini = self.prediction(x_test, clf_gini)
        self.cal_accuracy(y_test, y_pred_gini)
        print("Results Using Entropy:")
        y_pred_entropy = self.prediction(x_test, clf_entropy)
        self.cal_accuracy(y_test, y_pred_entropy)

    def train_using_gini(self, x_train, y_train):
        clf_gini = DecisionTreeClassifier(criterion="gini",
                                          random_state=100, max_depth=3, min_samples_leaf=5)
        clf_gini.fit(x_train, y_train)
        return clf_gini

    def train_using_entropy(self, x_train, y_train):
        clf_entropy = DecisionTreeClassifier(
            criterion="entropy", random_state=100,
            max_depth=3, min_samples_leaf=5)
        clf_entropy.fit(x_train, y_train)
        return clf_entropy

    def prediction(self, x_test, clf_object):
        y_pred = clf_object.predict(x_test)
        return y_pred

    def cal_accuracy(self, y_test, y_pred):
        print("Confusion Matrix: ",
              confusion_matrix(y_test, y_pred))

        print("Accuracy : ",
              accuracy_score(y_test, y_pred) * 100)

        print("Report : ",
              classification_report(y_test, y_pred))






