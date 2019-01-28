import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from experiment import Experiment


logger = logging.getLogger(__name__)


class Tree(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
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






