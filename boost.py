import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from yellowbrick.classifier import ClassificationReport


class Boost(Experiment):

    def __init__(self, attributes, classifications, dataset, **kwargs):
        dtc = DecisionTreeClassifier(random_state=10)
        self._csv_str = './results/{}/boost/'.format(dataset)


        params = {
            'predict__n_estimators': [1, 10, 50, 100],
            'predict__learning_rate': [0.1, 0.5, 1, 10],
            'predict__base_estimator__criterion': ["gini", "entropy"],
            "predict__base_estimator__splitter": ["best", "random"],
        }

        learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
        pipeline = Pipeline([('scale', StandardScaler()),
                             ('predict',
                              AdaBoostClassifier(random_state=10, base_estimator=dtc))])

        super().__init__(attributes, classifications, dataset, 'boost', pipeline, params,
                         learning_curve_train_sizes, True, verbose=1, iteration_curve=False)

    def run(self):
        cv = super().run()
        n_estimators = np.arange(1, 24, 2)
        train_iter = []
        estimator_iter = []
        final_df = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_estimator = cv.best_estimator_
            x_train, x_test, y_train, y_test = super().get_data_split()
            for i, n_estimator in enumerate(n_estimators):
                best_estimator.set_params(**{'predict__n_estimators': n_estimator})
                best_estimator.fit(x_train, y_train)
                train_iter.append(
                    np.mean(cross_val_score(best_estimator, x_train, y_train, cv=self._cv)))
                estimator_iter.append(
                    np.mean(cross_val_score(best_estimator, x_test, y_test, cv=self._cv)))
                final_df.append([n_estimator, train_iter[i], estimator_iter[i]])
            plt.figure()
            plt.plot(n_estimators, train_iter,
                     marker='o', color='blue', label='Train Score')
            plt.plot(n_estimators, estimator_iter,
                     marker='o', color='green', label='Test Score')
            plt.legend()
            plt.grid(linestyle='dotted')
            plt.xlabel('Estimators')
            plt.ylabel('Accuracy')
            csv_str = '{}/{}'.format(self._dataset, self._strategy)
            plt.savefig('./results/{}/estimator-curve.png'.format(csv_str))
            iter_csv = pd.DataFrame(data=final_df, columns=['Estimators', 'Train Accuracy', 'Test Accuracy'])
            iter_csv.to_csv('./results/{}/estimator.csv'.format(csv_str), index=False)

            self.naive_report(x_test, x_train, y_test, y_train, self._csv_str)

    @staticmethod
    def naive_report(x_test, x_train, y_test, y_train, csv_str):
        _, ax = plt.subplots()
        dtc = DecisionTreeClassifier(random_state=0)
        visualizer = ClassificationReport(
            AdaBoostClassifier(base_estimator=dtc)
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/naive-classification.png".format(csv_str))

        _, ax = plt.subplots()
        dtc = DecisionTreeClassifier(random_state=0, min_samples_leaf=20, min_samples_split=20, max_depth=5, max_leaf_nodes=5)
        visualizer = ClassificationReport(
            AdaBoostClassifier(base_estimator=dtc)
        )
        visualizer.fit(x_train, y_train)
        visualizer.score(x_test, y_test)
        visualizer.poof(outpath="{}/worst-classification.png".format(csv_str))

