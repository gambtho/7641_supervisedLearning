"""Abstraction for analysis and models
"""

import logging
from sklearn.model_selection import train_test_split, validation_curve, ShuffleSplit, cross_val_score, GridSearchCV, learning_curve
from sklearn.utils import compute_sample_weight
from time import process_time
from sklearn.metrics import classification_report, accuracy_score
from yellowbrick.classifier import ClassificationReport
from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores
from yellowbrick.features.importances import FeatureImportances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# matplotlib.use('Agg')

logger = logging.getLogger(__name__)



class Experiment:

    def __init__(self, attributes, classifications,
                 dataset, strategy, pipeline, params,
                 learning_curve_train_sizes,
                 timing_curve=False,
                 verbose=1,
                 iteration_curve=False):

        self._attributes = attributes
        self._classifications = classifications
        self._dataset = dataset
        self._strategy = strategy
        self._pipeline = pipeline
        self._params = params
        self._scoring = "accuracy"
        self._learning_curve_train_sizes = learning_curve_train_sizes
        self._verbose = verbose
        self._random = 10
        self._cv = ShuffleSplit(random_state=self._random)
        self._timing_curve = timing_curve
        self._iteration_curve = iteration_curve
        self.x_train, self.x_test, self.y_train, self.y_test = self._split_train_test()

        # force a seed for the experiment
        np.random.seed(self._random)

    def get_data_split(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def run(self):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test

        experiment_pipe = self._pipeline
        model_params = self._params

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = GridSearchCV(experiment_pipe,
                              n_jobs=8, param_grid=model_params, cv=self._cv, scoring=self._scoring,
                              refit=True, verbose=self._verbose)
            cv.fit(x_train, y_train)
            cv_all = pd.DataFrame(cv.cv_results_)
            csv_str = '{}/{}'.format(self._dataset, self._strategy)
            cv_all.to_csv('./results/{}/cv.csv'.format(csv_str), index=False)
            print("Accuracy of the tuned model: %.4f" % cv.best_score_)
            print(cv.best_params_)
            # self._basic_accuracy(cv, x_test, y_test, csv_str)
            # self._learning_curve(cv, x_train, y_train, csv_str)
            # self._classification_report(cv, x_test, y_test, x_train, y_train, csv_str)

            if self._strategy == 'tree':
                self._validation_curve(cv, x_train, y_train, 'predict__min_samples_split', [2, 10, 20], csv_str)
                self._validation_curve(cv, x_train, y_train, 'predict__max_depth', [1, 2, 5, 10], csv_str)
                self._validation_curve(cv, x_train, y_train, 'predict__min_samples_leaf', [1, 2, 5, 10], csv_str)
                self._validation_curve(cv, x_train, y_train, 'predict__max_leaf_nodes', [5, 10, 20], csv_str)

            # if self._timing_curve:
            #     self._create_timing_curve(cv, csv_str)
            # if self._iteration_curve:
            #     self._create_iteration_curve(cv, csv_str, x_train, x_test, y_train, y_test)
            return cv

    def _classification_report(self, cv, x_test, y_test, x_train, y_train, csv_str):
        _, ax = plt.subplots()

        visualizer = ClassificationReport(cv.best_estimator_, ax=ax, classes=list(set(self._classifications)))
        visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
        visualizer.score(x_test, y_test)  # Evaluate the model on the test data
        visualizer.poof(outpath="./results/{}/classification.png".format(csv_str))

    def _basic_accuracy(self, cv, x_test, y_test, csv_str):
        results_df = pd.DataFrame(columns=['best_estimator', 'best_score', 'best_params', 'test_score'],
                                  data=[
                                      [cv.best_estimator_, cv.best_score_, cv.best_params_, cv.score(x_test, y_test)]])
        results_df.to_csv('./results/{}/basic.csv'.format(csv_str), index=False)

    def _validation_curve(self, cv, x_train, y_train, param_name, param_range, csv_str):
        _, ax = plt.subplots()
        viz = ValidationCurve(
            cv.best_estimator_, param_name=param_name, param_range=param_range,
            cv=self._cv, scoring=self._scoring, n_jobs=8, ax=ax
        )
        viz.fit(x_train, y_train)
        viz.poof(outpath='./results/{}/validation-{}-curve.png'.format(csv_str, param_name))

    def _learning_curve(self, cv, x_train, y_train, csv_str):
        viz = LearningCurve(
            cv.best_estimator_, cv=self._cv, train_sizes=self._learning_curve_train_sizes,
            scoring=self._scoring, n_jobs=8
        )

        # Fit and poof the visualizer
        viz.fit(x_train, y_train)
        viz.poof()
        #plt.savefig('./results/{}/learning-curve.png'.format(csv_str))


        # accuracy_learning_curve = learning_curve(cv.best_estimator_, x_train, y_train,
        #                                          cv=self._cv, train_sizes=self._learning_curve_train_sizes,
        #                                          verbose=self._verbose,
        #                                          scoring=scoring, n_jobs=4)
        # train_scores = pd.DataFrame(index=accuracy_learning_curve[0], data=accuracy_learning_curve[1])
        # train_scores.to_csv('./results/{}/lc-train.csv'.format(csv_str), index=False)
        # test_scores = pd.DataFrame(index=accuracy_learning_curve[0], data=accuracy_learning_curve[2])
        # test_scores.to_csv('./results/{}/lc-test.csv'.format(csv_str), index=False)
        # plt.figure(1)
        # plt.plot(self._learning_curve_train_sizes, np.mean(train_scores, axis=1),
        #          marker='o', color='blue', label='Training Score')
        # plt.plot(self._learning_curve_train_sizes, np.mean(test_scores, axis=1),
        #          marker='o', color='green', label='Cross-Validation Score')
        # plt.legend()
        # plt.grid(linestyle='dotted')
        # plt.tight_layout()
        # plt.xlabel('Total Data Used for Training as a Percentage')
        # plt.ylabel('Accuracy')

    def _cross_validation_curve(self, cv, x_train, y_train, csv_str):
        viz = LearningCurve(
            cv.best_estimator_, cv=self._cv, train_sizes=self._learning_curve_train_sizes,
            scoring=self._scoring, n_jobs=8
        )

        # Fit and poof the visualizer
        viz.fit(x_train, y_train)
        viz.poof()
        # plt.savefig('./results/{}/learning-curve.png'.format(csv_str))
        _, ax = plt.subplots()

        oz = CVScores(
            cv.best_estimator_, ax=ax, cv=self._cv, scoring=self._scoring
        )

        oz.fit(x_train, y_train)
        oz.poof()



    def _create_timing_curve(self, estimator, csv_str):
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
        plt.tight_layout()
        plt.grid(linestyle='dotted')
        plt.xlabel('Total Data Used for Training as a Percentage')
        plt.ylabel('Time in Seconds')
        plt.savefig('./results/{}/timing-curve.png'.format(csv_str))
        time_csv = pd.DataFrame(data=final_df, columns=['Training Percentage', 'Train Time', 'Test Time'])
        time_csv.to_csv('./results/{}/time.csv'.format(csv_str), index=False)

    def _create_iteration_curve(self, estimator, csv_str, x_train, x_test, y_train, y_test):
        iterations = np.arange(1, 5000, 250)
        train_iter = []
        predict_iter = []
        final_df = []
        best_estimator = estimator.best_estimator_
        for i, iteration in enumerate(iterations):
            best_estimator.set_params(**{'predict__max_iter': iteration})
            best_estimator.fit(x_train, y_train)
            train_iter.append(np.mean(cross_val_score(best_estimator, x_train, y_train, scoring=self._scoring, cv=self._cv)))
            predict_iter.append(np.mean(cross_val_score(best_estimator, x_test, y_test, scoring=self._scoring, cv=self._cv)))
            final_df.append([iteration, train_iter[i], predict_iter[i]])
        plt.figure(3)
        plt.plot(iterations, train_iter,
                 marker='o', color='blue', label='Train Score')
        plt.plot(iterations, predict_iter,
                 marker='o', color='green', label='Test Score')
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('./results/{}/iteration-curve.png'.format(csv_str))
        iter_csv = pd.DataFrame(data=final_df, columns=['Iterations', 'Train Accuracy', 'Test Accuracy'])
        iter_csv.to_csv('./results/{}/iteration.csv'.format(csv_str), index=False)

    def _print_table(self, file_name, df):

        # https://stackoverflow.com / a / 45936469
        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')

        ax.table(cellText=df.values, colLabels=df.columns, loc='center')

        fig.tight_layout()

        plt.show()
        plt.savefig('{}'.format(file_name))

    def _feature_import(self, cv, x_train, y_train):
        # Create a new matplotlib figure
        fig = plt.figure()
        ax = fig.add_subplot()

        viz = FeatureImportances(cv.best_estimator_, ax=ax)
        viz.fit(x_train, y_train)
        viz.poof()

    def _split_train_test(self, test_size=0.3):
        return train_test_split(self._attributes, self._classifications,
                                test_size=test_size, random_state=self._random, stratify=self._classifications)



