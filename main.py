#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSIFIERS = {
    'nearest': Nearest,
    'vector': Vector,
    'neural': Neural,
    'tree': Tree,
    'boosting': Boost
}


def load_data(data='car'):
    log.info('Loading: ' + data)
    dataset = pd.read_csv(f"./data/{data}.data")

    class_target = 'class'

    # get values of class_target
    values = dataset[class_target]

    # get all values excluding the classification column
    data_minus_values = dataset.drop(class_target, axis=1)

    return values.as_matrix(), data_minus_values.as_matrix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--data', help='data to evaluate', choices=['car', 'iris'],
                        default='car')
    subparsers = parser.add_subparsers(title='strategy', dest='strategy')
    knn_parser = subparsers.add_parser('nearest', help='K-Nearest Neighbors')
    svm_parser = subparsers.add_parser('vector', help='Support Vector Machines')
    ann_parser = subparsers.add_parser('neural', help='Neural Networks')
    dt_parser = subparsers.add_parser('tree', help='Decision Trees')
    boosting_parser = subparsers.add_parser('boost', help='Boosting')
    args = parser.parse_args()

    if not args.command:
        parser.print_help()

    strategy = args.strategy
    path = './results/{}/{}'.format(args.dataset, strategy)
    if not os.path.exists(path):
        log.info('Making results directory')
        os.makedirs(path)
    classifications, attributes = load_data(args.dataset)
    log.info('Running %s', strategy)
    args.classifications = classifications
    args.attributes = attributes
    CLASSIFIERS[strategy](**vars(args)).run()
