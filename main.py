#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module documentation goes here
   and here
   and ...
"""

import argparse
import clean
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


def load_data(dataset='absent'):
    if dataset == 'absent':
        log.info('Exploring absent dataset')
        dataset = pd.read_csv('./data/absent.csv')
        class_target = 'Absent'
        classifications = dataset[class_target]
        attributes = dataset.drop(class_target, axis=1)
        return classifications.as_matrix(), attributes.as_matrix()

    # else:
    #     log.info('Exploring credit card dataset')
    #     dataset = pd.read_csv('./data/credit-card-final.csv')
    #     class_target = 'default_payment_next_month'
    #     classifications = dataset[class_target]
    #     attributes = dataset.drop(class_target, axis=1)

    # return classifications.as_matrix(), attributes.as_matrix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-d', '--dataset', help='Which dataset to analyze', choices=['absent', 'dow'],
                        default='absent')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    cleaner_parser = subparsers.add_parser('clean',
                                           help='Perform data cleanup')
    knn_parser = subparsers.add_parser('nearest', help='Use k-nearest neighbors')
    svm_parser = subparsers.add_parser('vector', help='Use Support Vector Machines')
    ann_parser = subparsers.add_parser('neural', help='Use neural networks')
    dt_parser = subparsers.add_parser('tree', help='Use decision trees')
    boosting_parser = subparsers.add_parser('boost', help='Run boosting')
    args = parser.parse_args()

    if not args.command:
        parser.print_help()

    command = args.command
    if command == 'clean':
        log.info('Updating data')
        clean.create_final_datasets()
    else:
        path = './results/{}/{}'.format(args.dataset, command)
        if not os.path.exists(path):
            log.info('Making results directory')
            os.makedirs(path)
        classifications, attributes = load_data(args.dataset)
        log.info('Running %s', command)
        args.classifications = classifications
        args.attributes = attributes
        CLASSIFIERS[command](**vars(args)).run()
