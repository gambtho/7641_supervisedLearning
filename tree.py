import logging
from utilities import Utilities
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger(__name__)


class Tree:

    def __init__(self, data, target, feature_names, dataset, **kwargs):
        self.target = target
        self.data = data
        self.feature_names = feature_names
        self.target_names = list(set(target))
        self.dataset = dataset
        self.directory = 'tree'

    def run(self):

        le = LabelEncoder()
        mapping_file = open('results/{}/{}-mappings.png'.format(self.directory, self.dataset), "w+")

        for column in self.data.columns:
            if self.data[column].dtype == type(object):
                self.data[column] = le.fit_transform(self.data[column].astype(str))

                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                mapping_file.write('{}\n'.format(le_name_mapping))

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.4, random_state=17)

        # default criterion=gini, can swap to criterion='entropy
        dtc = DecisionTreeClassifier(random_state=15)

        clf = dtc.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        # logger.info(le.inverse_transform(y_pred))

        accuracy_score(y_test, y_pred)
        print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))

        dot = tree.export_graphviz(clf, out_file=None, feature_names=self.feature_names, class_names=self.target_names,
                                   filled=True, rounded=True, special_characters=True)

        graph = graphviz.Source(dot)
        graph.format = 'png'
        graph.render('{}-tree'.format(self.dataset), directory='results/{}'.format(self.directory), view=False)

        u = Utilities()
        u.plot_classification_report(classification_report(y_test, y_pred, target_names=self.target_names))

        plt.savefig('results/{}/{}-report.png'.format(self.directory, self.dataset), dpi=200, format='png', bbox_inches='tight')
        plt.close()







