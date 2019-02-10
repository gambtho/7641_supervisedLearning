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
from sklearn.preprocessing import LabelEncoder
from printing import Printing

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
    dataset = pd.read_csv(f"./data/{data}.csv")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--dataset', help='data to evaluate', choices=['car', 'mushroom'],
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
    Printing.plot_data_info(args.attributes, args.classifications, args.dataset)

    print('{}---------------------->{}'.format(args.dataset, args.strategy))
    CLASSIFIERS[strategy](**vars(args)).run()

    args.dataset = '{}-slim'.format(args.dataset)
    path = './results/{}/{}'.format(args.dataset, strategy)
    args.attributes, args.classifications = load_data(args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)
        args.attributes, args.classifications = load_data(args.dataset)
    CLASSIFIERS[strategy](**vars(args)).run()



