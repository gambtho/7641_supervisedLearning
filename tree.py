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

        pipeline = Pipeline([('scale', StandardScaler()), ('predict', DecisionTreeClassifier())])
        params = {
            'predict__criterion': ['gini', 'entropy'],
            'predict__class_weight': ['balanced']
        }
        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        super().__init__(attributes, classifications, dataset, 'tree', pipeline, params,
                         learning_curve_train_sizes, True, verbose=1)

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







