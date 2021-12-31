# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import sys

import numpy as np
import pandas as pd

from itertools import product

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import ConvergenceWarning

from k_nearest_neighbors import KNearestNeighbors


def test_knn():
    print('Testing K-Nearest Neighbors Classification...')
    param_dict = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['l1', 'l2']}
    params_list = [dict(zip(param_dict.keys(), v)) for v in product(*param_dict.values())]
    params_len = len(params_list)

    knn_iris_results = pd.DataFrame(columns = ['  Model',
                                               '  # of Neighbors',
                                               '  Weight Function',
                                               '  Distance Metric',
                                               '  Accuracy Score'])

    for i, params in enumerate(params_list):
        sys.stdout.write('\r')
        sys.stdout.write(f" - Iris Dataset Progress:   [{'=' * int(60 * (i + 1) / params_len):60s}]"
                         f' {int(100 * (i + 1) / params_len)}%')
        sys.stdout.flush()

        sklearn_model = KNeighborsClassifier(n_neighbors = params['n_neighbors'],
                                             weights = params['weights'],
                                             algorithm = 'brute',
                                             p = 1 if params['metric'] == 'l1' else 2)
        sklearn_model.fit(X_train_i, y_train_i)
        y_pred = sklearn_model.predict(X_test_i)

        knn_iris_results.loc[len(knn_iris_results)] = ['Sklearn',
                                                       params['n_neighbors'],
                                                       params['weights'],
                                                       params['metric'],
                                                       round(accuracy_score(y_test_i, y_pred), 3)]

        my_model = KNearestNeighbors(params['n_neighbors'], params['weights'], params['metric'])
        my_model.fit(X_train_i, y_train_i)
        y_pred = my_model.predict(X_test_i)

        knn_iris_results.loc[len(knn_iris_results)] = ['My Model',
                                                       params['n_neighbors'],
                                                       params['weights'],
                                                       params['metric'],
                                                       round(accuracy_score(y_test_i, y_pred), 3)]

        knn_iris_results.loc[len(knn_iris_results)] = ['', '', '', '', '']

    print()

    knn_digits_results = pd.DataFrame(columns = ['  Model ',
                                                 '  # of Neighbors',
                                                 '  Weight Function',
                                                 '  Distance Metric',
                                                 '  Accuracy Score'])

    for i, params in enumerate(params_list):
        sys.stdout.write('\r')
        sys.stdout.write(f" - Digits Dataset Progress: [{'=' * int(60 * (i + 1) / params_len):60s}]"
                         f' {int(100 * (i + 1) / params_len)}%')
        sys.stdout.flush()

        sklearn_model = KNeighborsClassifier(n_neighbors = params['n_neighbors'],
                                             weights = params['weights'],
                                             algorithm = 'brute',
                                             p = 1 if params['metric'] == 'l1' else 2)
        sklearn_model.fit(X_train_d, y_train_d)
        y_pred = sklearn_model.predict(X_test_d)

        knn_digits_results.loc[len(knn_digits_results)] = ['Sklearn',
                                                           params['n_neighbors'],
                                                           params['weights'],
                                                           params['metric'],
                                                           round(accuracy_score(y_test_d, y_pred), 3)]

        my_model = KNearestNeighbors(params['n_neighbors'], params['weights'], params['metric'])
        my_model.fit(X_train_d, y_train_d)
        y_pred = my_model.predict(X_test_d)

        knn_digits_results.loc[len(knn_digits_results)] = ['My Model',
                                                           params['n_neighbors'],
                                                           params['weights'],
                                                           params['metric'],
                                                           round(accuracy_score(y_test_d, y_pred), 3)]

        knn_digits_results.loc[len(knn_digits_results)] = ['', '', '', '', '']

    print('\n')

    print('Exporting KNN Results to HTML Files...\n')
    knn_iris_results.to_html('knn_iris_results.html')
    knn_digits_results.to_html('knn_digits_results.html')

    print('Done Testing K-Nearest Neighbors Classification!\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Incorrect number of arguments have been specified.')

    np.random.seed(42)
    pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000)
    warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

    print('Loading the Iris and Digits Datasets...\n')
    data_i, data_d = load_iris(), load_digits()
    X_i, X_d = data_i.data, data_d.data
    y_i, y_d = data_i.target, data_d.target

    print('Splitting the Datasets into Train and Test Sets...\n')
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_i, y_i, test_size = 0.3, random_state = 42)
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size = 0.3, random_state = 42)

    print('Standardizing the Train and Test Datasets...\n')
    sc_i = StandardScaler()
    sc_d = StandardScaler()

    X_train_i = sc_i.fit_transform(X_train_i)
    X_test_i = sc_i.transform(X_test_i)
    X_train_d = sc_d.fit_transform(X_train_d)
    X_test_d = sc_d.transform(X_test_d)

    if sys.argv[1] == 'knn':
        test_knn()

    print('Program Finished! Exiting the Program...')
