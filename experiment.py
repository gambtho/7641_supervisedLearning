"""Abstraction for analysis and models
"""

import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from time import process_time
from sklearn.model_selection import ShuffleSplit, cross_val_score
from utilities import Utilities
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score
import graphviz
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
import warnings


def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


scorer = make_scorer(balanced_accuracy)


class Experiment:

    def __init__(self, attributes, classifications,
                 dataset, strategy, pipeline, params,
                 learning_curve_train_sizes,
                 timing_curve=False,
                 verbose=1,
                 iteration_curve=False):
        ''' Constructor
        '''
        # what data are we looking at
        self._attributes = attributes
        self._classifications = classifications
        self._dataset = dataset
        self._strategy = strategy
        self._pipeline = pipeline
        self._params = params
        self._learning_curve_train_sizes = learning_curve_train_sizes
        self._verbose = verbose
        self._random = 10
        self._cv = ShuffleSplit(random_state=self._random)
        self._timing_curve = timing_curve
        self._iteration_curve = iteration_curve
        # force a seed for the experiment
        np.random.seed(self._random)

    def run(self):
        logger.info('Running the experiment')
        x_train, x_test, y_train, y_test = self._split_train_test()
        logger.info('Got data split')
        experiment_pipe = self._pipeline
        model_params = self._params

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = GridSearchCV(experiment_pipe,
                              n_jobs=4, param_grid=model_params, cv=self._cv, scoring=scorer,
                              refit=True, verbose=self._verbose)
            logger.info('Searching params')
            cv.fit(x_train, y_train)
            cv_all = pd.DataFrame(cv.cv_results_)
            csv_str = '{}/{}'.format(self._dataset, self._strategy)
            cv_all.to_csv('./results/{}/cv.csv'.format(csv_str), index=False)
            print("Accuracy of the tuned model: %.4f" % cv.best_score_)
            print(cv.best_params_)
            self._basic_accuracy(cv, x_test, y_test, csv_str)
            self._learning_curve(cv, x_train, y_train, csv_str)
            self._classification_report(cv, x_test, y_test, csv_str)
            logger.info(model_params)
            logger.info(cv.best_params_)
            # self._plot_grid_search(cv.cv_results_, model_params[0].value, model_params[1].value, model_params[0].key, model_params[1].key)

            # if self._strategy == 'tree':
            #     self._create_tree(cv, csv_str, x_train, y_train)
            if self._timing_curve:
                self._create_timing_curve(cv, csv_str)
            if self._iteration_curve:
                self._create_iteration_curve(cv, csv_str, x_train, x_test, y_train, y_test)
            return cv

    def _create_tree(self, cv, x_train, y_train, csv_str):
        feature_names = list(self._attributes)
        target_names = list(set(self._classifications))
        # logger.info(x_train, y_train)
        # cv.best_estimator_.fit(x_train, y_train)

        dot = tree.export_graphviz(cv.best_estimator_, out_file=None, feature_names=feature_names,
                                   class_names=target_names, filled=True, rounded=True, special_characters=True)

        graph = graphviz.Source(dot)
        graph.format = 'png'
        graph.render('tree', directory='./results/{}'.format(csv_str), view=False)

    def _classification_report(self, cv, x_test, y_test, csv_str):
        u = Utilities()
        u.plot_classification_report(classification_report(cv.best_estimator_.predict(x_test), y_test))
        plt.savefig('./results/{}/classification-report.png'.format(csv_str), dpi=200, format='png',
                    bbox_inches='tight')
        plt.close()

    def _basic_accuracy(self, cv, x_test, y_test, csv_str):
        logger.info('Writing out basic result')
        results_df = pd.DataFrame(columns=['best_estimator', 'best_score', 'best_params', 'test_score'],
                                  data=[
                                      [cv.best_estimator_, cv.best_score_, cv.best_params_, cv.score(x_test, y_test)]])
        results_df.to_csv('./results/{}/basic.csv'.format(csv_str), index=False)

    def _learning_curve(self, cv, x_train, y_train, csv_str):
        logger.info('Creating learning curve')
        accuracy_learning_curve = learning_curve(cv.best_estimator_, x_train, y_train,
                                                 cv=self._cv, train_sizes=self._learning_curve_train_sizes,
                                                 verbose=self._verbose,
                                                 scoring=scorer, n_jobs=4)
        train_scores = pd.DataFrame(index=accuracy_learning_curve[0], data=accuracy_learning_curve[1])
        train_scores.to_csv('./results/{}/lc-train.csv'.format(csv_str), index=False)
        test_scores = pd.DataFrame(index=accuracy_learning_curve[0], data=accuracy_learning_curve[2])
        test_scores.to_csv('./results/{}/lc-test.csv'.format(csv_str), index=False)
        logger.info('Saving learning curves')
        plt.figure(1)
        plt.plot(self._learning_curve_train_sizes, np.mean(train_scores, axis=1),
                 marker='o', color='blue', label='Training Score')
        plt.plot(self._learning_curve_train_sizes, np.mean(test_scores, axis=1),
                 marker='o', color='green', label='Cross-Validation Score')
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.xlabel('Total Data Used for Training as a Percentage')
        plt.ylabel('Accuracy')
        plt.savefig('./results/{}/learning-curve.png'.format(csv_str))

    def _create_timing_curve(self, estimator, csv_str):
        logger.info('Creating timing curve')
        training_data_sizes = np.arange(0.1, 1.0, 0.1)
        train_time = []
        predict_time = []
        final_df = []
        best_estimator = estimator.best_estimator_
        for i, train_data in enumerate(training_data_sizes):
            x_train, x_test, y_train, _ = self._split_train_test(1 - train_data)
            start = process_time()
            best_estimator.fit(x_train, y_train)
            end_train = process_time()
            best_estimator.predict(x_test)
            end_predict = process_time()
            train_time.append(end_train - start)
            predict_time.append(end_predict - end_train)
            final_df.append([train_data, train_time[i], predict_time[i]])
        plt.figure(2)
        plt.plot(training_data_sizes, train_time,
                 marker='o', color='blue', label='Training')
        plt.plot(training_data_sizes, predict_time,
                 marker='o', color='green', label='Predicting')
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.xlabel('Total Data Used for Training as a Percentage')
        plt.ylabel('Time in Seconds')
        plt.savefig('./results/{}/timing-curve.png'.format(csv_str))
        time_csv = pd.DataFrame(data=final_df, columns=['Training Percentage', 'Train Time', 'Test Time'])
        time_csv.to_csv('./results/{}/time.csv'.format(csv_str), index=False)

    def _create_iteration_curve(self, estimator, csv_str, x_train, x_test, y_train, y_test):
        logger.info('Creating iteration curve')
        iterations = np.arange(1, 5000, 250)
        train_iter = []
        predict_iter = []
        final_df = []
        best_estimator = estimator.best_estimator_
        for i, iteration in enumerate(iterations):
            best_estimator.set_params(**{'predict__max_iter': iteration})
            best_estimator.fit(x_train, y_train)
            train_iter.append(np.mean(cross_val_score(best_estimator, x_train, y_train, scoring=scorer, cv=self._cv)))
            predict_iter.append(np.mean(cross_val_score(best_estimator, x_test, y_test, scoring=scorer, cv=self._cv)))
            final_df.append([iteration, train_iter[i], predict_iter[i]])
        plt.figure(3)
        plt.plot(iterations, train_iter,
                 marker='o', color='blue', label='Train Score')
        plt.plot(iterations, predict_iter,
                 marker='o', color='green', label='Test Score')
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('./results/{}/iteration-curve.png'.format(csv_str))
        iter_csv = pd.DataFrame(data=final_df, columns=['Iterations', 'Train Accuracy', 'Test Accuracy'])
        iter_csv.to_csv('./results/{}/iteration.csv'.format(csv_str), index=False)

    def _split_train_test(self, test_size=0.3):
        return train_test_split(self._attributes, self._classifications,
                                test_size=test_size, random_state=self._random, stratify=self._classifications)



