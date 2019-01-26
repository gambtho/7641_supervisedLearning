"""Parse arguments and run analysis
"""

import argparse
from nearest import Nearest
from vector import Vector
from neural import Neural
from tree import Tree
# from boost import Boost
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSIFIERS = {
    'nearest': Nearest,
    'vector': Vector,
    'neural': Neural,
    'tree': Tree,
    # 'boosting': Boost
}


def load_data(data='car'):
    log.info('Loading: ' + data)
    dataset = pd.read_csv(f"./data/{data}.data")
    class_target = 'class'

    _feature_names = list(dataset)[:-1]
    _target = dataset[class_target].as_matrix()
    _data = dataset.drop(class_target, axis=1)

    return _data, _target, _feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--dataset', help='data to evaluate', choices=['car', 'titanic', 'iris', 'mushrooms'],
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
    path = './results/{}'.format(strategy)
    if not os.path.exists(path):
        os.makedirs(path)
    data, target, feature_names = load_data(args.dataset)
    log.info('Running %s', strategy)
    args.target = target
    args.data = data
    args.feature_names = feature_names
    CLASSIFIERS[strategy](**vars(args)).run()
