from yellowbrick.classifier import ClassificationReport
from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores
from yellowbrick.features.importances import FeatureImportances
from pandas.plotting import table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from yellowbrick.target import FeatureCorrelation
from yellowbrick import ClassBalance


class Printing:

    @staticmethod
    def classification_report(cv, x_test, y_test, x_train, y_train, csv_str, classes):
        _, ax = plt.subplots()
        visualizer = ClassificationReport(cv.best_estimator_, ax=ax, classes=classes)
        visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
        visualizer.score(x_test, y_test)  # Evaluate the model on the test data
        visualizer.poof(outpath="./results/{}/classification.png".format(csv_str))

    @staticmethod
    def basic_accuracy(cv, x_test, y_test, csv_str):
        results_df = pd.DataFrame(columns=['best_estimator', 'best_score', 'best_params', 'test_score'],
                                  data=[
                                      [cv.best_estimator_, cv.best_score_, cv.best_params_, cv.score(x_test, y_test)]])
        results_df.to_csv('./results/{}/basic.csv'.format(csv_str), index=False)
        # self._print_table('./results/{}/basic.png'.format(csv_str), results_df)

    @staticmethod
    def validation_curve(cv, x_train, y_train, param_name, param_range, csv_str, scoring, shuffle):
        _, ax = plt.subplots()
        viz = ValidationCurve(
            cv.best_estimator_, param_name=param_name, param_range=param_range,
            cv=shuffle, scoring=scoring, n_jobs=8, ax=ax
        )
        viz.fit(x_train, y_train)
        viz.poof(outpath='./results/{}/validation-{}-curve.png'.format(csv_str, param_name))

    @staticmethod
    def cross_validation_curve(cv, x_train, y_train, csv_str, scoring, shuffle):
        _, ax = plt.subplots()
        oz = CVScores(
            cv.best_estimator_, ax=ax, cv=shuffle, scoring=scoring
        )
        oz.fit(x_train, y_train)
        oz.poof(outpath='./results/{}/cross-validation-curve.png'.format(csv_str))

    @staticmethod
    def print_table(file_name, df):
        fig, ax = plt.subplots()  # set size frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
        tabla = table(ax, df, loc='upper right', colWidths=[0.17] * len(df.columns))  # where df is your data frame
        tabla.auto_set_font_size(False)  # Activate set fontsize manually
        tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
        tabla.scale(1.2, 1.2)  # change size table

        plt.savefig('{}'.format(file_name), transparent=True)

    @staticmethod
    def feature_importance(cv, x_train, y_train):
        fig = plt.figure()
        ax = fig.add_subplot()

        viz = FeatureImportances(cv.best_estimator_, ax=ax)
        viz.fit(x_train, y_train)
        viz.poof()

    @staticmethod
    def learning_curve(cv, x_train, y_train, csv_str, scoring, shuffle, train_sizes):
        plt.figure()
        viz = LearningCurve(
            cv.best_estimator_, cv=shuffle, train_sizes=train_sizes,
            scoring=scoring, n_jobs=8
        )
        viz.fit(x_train, y_train)
        viz.poof(outpath='./results/{}/learning-curve.png'.format(csv_str))

    @staticmethod
    def plot_data_info(attributes, classifications, dataset):
        plt.figure()
        x, y = attributes, classifications
        feature_names = list(attributes)
        classes = list(set(classifications))
        x_pd = pd.DataFrame(x, columns=feature_names)
        correlation = FeatureCorrelation(method='mutual_info-classification',
                                         feature_names=feature_names, sort=True)
        correlation.fit(x_pd, y, random_state=0)
        correlation.poof(outpath='./results/{}/correlation.png'.format(dataset))
        _, _, y_train, y_test = train_test_split(x, y, test_size=0.2)
        plt.figure()
        balance = ClassBalance(labels=classes)
        balance.fit(y_train, y_test)
        balance.poof(outpath='./results/{}/balance.png'.format(dataset))

    @staticmethod
    def create_iteration_curve(estimator, csv_str, x_train, x_test, y_train, y_test, scoring, shuffle):
        plt.figure()
        iterations = np.arange(1, 5000, 250)
        train_iter = []
        predict_iter = []
        final_df = []
        best_estimator = estimator.best_estimator_
        for i, iteration in enumerate(iterations):
            best_estimator.set_params(**{'predict__max_iter': iteration})
            best_estimator.fit(x_train, y_train)
            train_iter.append(np.mean(cross_val_score(best_estimator, x_train, y_train, scoring=scoring, cv=shuffle)))
            predict_iter.append(np.mean(cross_val_score(best_estimator, x_test, y_test, scoring=scoring, cv=shuffle)))
            final_df.append([iteration, train_iter[i], predict_iter[i]])
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

