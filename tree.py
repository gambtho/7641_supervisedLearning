import logging
from utilities import Utilities
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from experiment import Experiment
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger(__name__)


class Tree(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        # self.classifications = classifications
        # self.attributes = attributes
        # self.feature_names = list(attributes)
        # self.target_names = list(set(classifications))
        # self.dataset = dataset
        # self.strategy = 'tree'

        criteria = ["gini", "entropy"]  # criteria to be tested
        min_sample_split_range = [2, 10, 20]  # min sample split to be tested
        max_depth_range = [None, 2, 5, 10]  # max depth to be tested
        min_samples_leaf_range = [1, 5, 10]  # min samples in the leaf to be tested
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

    # # Import
    # from sklearn.grid_search import GridSearchCV
    #
    # # Define the parameter values that should be searched
    # sample_split_range = list(range(1, 50))
    #
    # # Create a parameter grid: map the parameter names to the values that should be searched
    # # Simply a python dictionary
    # # Key: parameter name
    # # Value: list of values that should be searched for that parameter
    # # Single key-value pair for param_grid
    # param_grid = dict(min_samples_split=sample_split_range)
    #
    # # instantiate the grid
    # grid = GridSearchCV(dtc, param_grid, cv=10, scoring='accuracy')
    #
    # # fit the grid with data
    # grid.fit(X_train, y_train)

    # def run(self):
    #
    #     le = LabelEncoder()
    #     mapping_file = open('results/{}/{}-mappings.png'.format(self.dataset, self.strategy), "w+")
    #
    #     for column in self.attributes.columns:
    #         if self.attributes[column].dtype == type(object):
    #             self.attributes[column] = le.fit_transform(self.attributes[column].astype(str))
    #
    #             le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #             mapping_file.write('{}\n'.format(le_name_mapping))
    #
    #     x_train, x_test, y_train, y_test = train_test_split(self.attributes, self.classifications, test_size=0.4, random_state=17)
    #
    #     # default criterion=gini, can swap to criterion='entropy
    #     dtc = DecisionTreeClassifier(random_state=15)
    #
    #     clf = dtc.fit(x_train, y_train)
    #
    #     y_pred = clf.predict(x_test)
    #     # logger.info(le.inverse_transform(y_pred))
    #
    #     accuracy_score(y_test, y_pred)
    #     print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
    #
    #     dot = tree.export_graphviz(clf, out_file=None, feature_names=self.feature_names, class_names=self.target_names,
    #                                filled=True, rounded=True, special_characters=True)
    #
    #     graph = graphviz.Source(dot)
    #     graph.format = 'png'
    #     graph.render('tree', directory='results/{}'.format(self.dataset), view=False)
    #
    #     u = Utilities()
    #     u.plot_classification_report(classification_report(y_test, y_pred, target_names=self.target_names))
    #
    #     plt.savefig('results/{}/{}-report.png'.format(self.dataset, self.strategy), dpi=200, format='png', bbox_inches='tight')
    #     plt.close()







