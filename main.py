"""Parse arguments and run analysis
"""

import argparse
from nearest import Nearest
from vector import Vector
from neural import Neural
from tree import Tree
from boost import Boost
import logging
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yellowbrick.target import FeatureCorrelation
from yellowbrick import ClassBalance

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSIFIERS = {
    'nearest': Nearest,
    'vector': Vector,
    'neural': Neural,
    'tree': Tree,
    'boost': Boost
}


def load_data(data='car'):
    dataset = pd.read_csv(f"./data/{data}.data")
    class_target = 'class'

    classifications = dataset[class_target]
    attributes = dataset.drop(class_target, axis=1)

    le = LabelEncoder()

    for column in attributes.columns:
        if attributes[column].dtype == type(object):
            attributes[column] = le.fit_transform(attributes[column].astype(str))
            attributes[column] = attributes[column].astype(float)
        elif pd.api.types.is_int64_dtype(attributes[column].dtype):
            attributes[column] = attributes[column].astype(float)

    return attributes, classifications


def plot_data_info(attributes, classifications):
    x, y = attributes, classifications
    feature_names = list(attributes)
    classes = list(set(classifications))
    x_pd = pd.DataFrame(x, columns=feature_names)
    correlation = FeatureCorrelation(method='mutual_info-classification',
                                     feature_names=feature_names, sort=True)
    correlation.fit(x_pd, y, random_state=0)
    correlation.poof(outpath='./results/{}/correlation.png'.format(args.dataset))
    _, _, y_train, y_test = train_test_split(x, y, test_size=0.2)
    balance = ClassBalance(labels=classes)
    balance.fit(y_train, y_test)
    balance.poof(outpath='./results/{}/balance.png'.format(args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--dataset', help='data to evaluate', choices=['car', 'mushrooms'],
                        default='car')
    subparsers = parser.add_subparsers(title='strategy', dest='strategy')
    tree_parser = subparsers.add_parser('tree', help='Decision Trees')
    nearest_parser = subparsers.add_parser('nearest', help='K-Nearest Neighbors')
    vector_parser = subparsers.add_parser('vector', help='Support Vector Machines')
    neural_parser = subparsers.add_parser('neural', help='Neural Networks')
    boosting_parser = subparsers.add_parser('boost', help='Boosting')
    args = parser.parse_args()

    if not args.strategy:
        parser.print_help()

    strategy = args.strategy
    path = './results/{}/{}'.format(args.dataset, strategy)

    args.attributes, args.classifications = load_data(args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)
    # plot_data_info(args.attributes, args.classifications)

    print('{}---------------------->{}'.format(args.dataset, args.strategy))
    CLASSIFIERS[strategy](**vars(args)).run()



