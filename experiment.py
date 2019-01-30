"""Abstraction for analysis and models
"""

import numpy as np
import pandas as pd
from printing import Printing
import warnings
import matplotlib.pyplot as plt
from time import process_time
from sklearn.model_selection import train_test_split, \
    validation_curve, ShuffleSplit, cross_val_score, \
    GridSearchCV, learning_curve
# matplotlib.use('Agg')


def split_train_test(attributes, classifications, random, test_size=0.3):
    return train_test_split(attributes, classifications,
                            test_size=test_size, random_state=random, stratify=classifications)


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
        self.x_train, self.x_test, self.y_train, self.y_test = split_train_test(attributes=self._attributes,
                                                                                classifications=self._classifications,
                                                                                random=self._random)

    def get_data_split(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    def run(self):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test

        experiment_pipe = self._pipeline
        model_params = self._params
        classes = list(set(self._classifications))
        scoring = self._scoring
        shuffle = self._cv
        train_sizes = self._learning_curve_train_sizes

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
            Printing.basic_accuracy(cv, x_test, y_test, csv_str)
            Printing.learning_curve(cv, x_train, y_train, csv_str, scoring, shuffle, train_sizes)
            Printing.cross_validation_curve(cv, x_train, y_train, csv_str, scoring, shuffle)
            Printing.classification_report(cv, x_test, y_test, x_train, y_train, csv_str, classes)

            if self._strategy == 'tree':
                Printing.validation_curve(cv, x_train, y_train, 'predict__min_samples_split', [2, 10, 20], csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__max_depth', [1, 2, 5, 10], csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__min_samples_leaf', [1, 2, 5, 10], csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__max_leaf_nodes', [5, 10, 20], csv_str, scoring, shuffle)

            if self._strategy == 'nearest':
                Printing.validation_curve(cv, x_train, y_train, 'predict__n_neighbors', [5, 6, 7, 8, 9, 10], csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__leaf_size', [1, 3, 5, 9, 15], csv_str, scoring, shuffle)

            if self._strategy == 'vector':
                Printing.validation_curve(cv, x_train, y_train, 'predict__c_range', [0.01, 0.1, 1, 10, 100, 1000], csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__degree', [1, 2, 3, 4, 5, 6], csv_str, scoring, shuffle)

            if self._strategy == 'neural':
                Printing.validation_curve(cv, x_train, y_train, 'predict__alpha', np.arange(0.0001, .05, 3, 0.1), csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__hidden_layer_sizes', [(32, 64, 32), (64, 128, 64), (50, 50, 50), (50, 100, 50), (100,)], csv_str, scoring, shuffle)

            if self._strategy == 'boosting':
                Printing.validation_curve(cv, x_train, y_train, 'predict__n_estimators', [1, 10, 50, 100], csv_str, scoring, shuffle)
                Printing.validation_curve(cv, x_train, y_train, 'predict__learning_rate', [0.1, 0.5, 1, 10], csv_str, scoring, shuffle)

            if self._timing_curve:
                self._create_timing_curve(cv, csv_str)
            if self._iteration_curve:
                Printing.create_iteration_curve(cv, csv_str, x_train, x_test, y_train, y_test, scoring, shuffle)
            return cv

    def _create_timing_curve(self, estimator, csv_str):
        plt.figure()

        training_data_sizes = np.arange(0.1, 1.0, 0.1)
        train_time = []
        predict_time = []
        final_df = []
        best_estimator = estimator.best_estimator_
        for i, train_data in enumerate(training_data_sizes):
            x_train, x_test, y_train, _ = split_train_test(test_size=(1 - train_data),
                                                           attributes=self._attributes,
                                                           classifications=self._classifications,
                                                           random=self._random)
            start = process_time()
            best_estimator.fit(x_train, y_train)
            end_train = process_time()
            best_estimator.predict(x_test)
            end_predict = process_time()
            train_time.append(end_train - start)
            predict_time.append(end_predict - end_train)
            final_df.append([train_data, train_time[i], predict_time[i]])
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



