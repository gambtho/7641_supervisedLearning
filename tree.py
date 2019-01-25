import logging
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
                self.data[column] = le.fit_transform(self.data[column])
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                mapping_file.write('{}\n'.format(le_name_mapping))

        x_train, x_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.4, random_state=17)
        # default criterion=gini, can swap to criterion='entropy

        dtc = DecisionTreeClassifier(random_state=0)

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

        self.plot_classification_report(classification_report(y_test, y_pred, target_names=self.target_names))
        plt.savefig('results/{}/{}-report.png'.format(self.directory, self.dataset), dpi=200, format='png', bbox_inches='tight')
        plt.close()

    def show_values(self, pc, fmt="%.2f", **kw):
        '''
        Heatmap with text in each cell with matplotlib's pyplot
        Source: https://stackoverflow.com/a/25074150/395857
        By HYRY
        '''
        from itertools import zip_longest as zip
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    def cm2inch(self, *tupl):
        '''
        Specify figure size in centimeter in matplotlib
        Source: https://stackoverflow.com/a/22787457/395857
        By gns-ank
        '''
        inch = 2.54
        if type(tupl[0]) == tuple:
            return tuple(i / inch for i in tupl[0])
        else:
            return tuple(i / inch for i in tupl)

    def heatmap(self, AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20,
                correct_orientation=False, cmap='RdBu'):
        '''
        Inspired by:
        - https://stackoverflow.com/a/16124677/395857
        - https://stackoverflow.com/a/25074150/395857
        '''

        # Plot it out
        fig, ax = plt.subplots()
        # c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
        c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

        # set tick labels
        # ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
        ax.set_xticklabels(xticklabels, minor=False)
        ax.set_yticklabels(yticklabels, minor=False)

        # set title and x/y labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Remove last blank column
        plt.xlim((0, AUC.shape[1]))

        # Turn off all the ticks
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # Add color bar
        plt.colorbar(c)

        # Add text in each cell
        self.show_values(c)

        # Proper orientation (origin at the top left instead of bottom left)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            # resize
        fig = plt.gcf()
        # fig.set_size_inches(cm2inch(40, 20))
        # fig.set_size_inches(cm2inch(40*4, 20*4))
        fig.set_size_inches(self.cm2inch(figure_width, figure_height))

    def plot_classification_report(self, classification_report, title='Classification report ', cmap='RdBu'):
        '''
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857
        '''
        lines = classification_report.split('\n')

        classes = []
        plotMat = []
        support = []
        class_names = []
        for line in lines[2: (len(lines) - 2)]:
            t = line.strip().split()
            if len(t) < 2: continue
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            # print(v)
            plotMat.append(v)

        # print('plotMat: {0}'.format(plotMat))
        # print('support: {0}'.format(support))

        xlabel = 'Metrics'
        ylabel = 'Classes'
        xticklabels = ['Precision', 'Recall', 'F1-score']
        yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = False
        self.heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height,
                correct_orientation, cmap=cmap)






