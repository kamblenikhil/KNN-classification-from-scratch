# Auhtor - Nikhil Kamble
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

# to understand the concept of knn using sklearn, I reffered the official website of scikit-learn (cited below) and wikipedia page as well
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

import numpy as np
from sklearn import metrics
from utils import euclidean_distance, manhattan_distance

class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        # self._distance = euclidean_distance if metric == 'l2' else manhattan_distance
        self.metric = metric

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        prediction = []
        # X -> test data
        for i in range(len(X)):

            d = []
            a = []

            # _X -> train data
            for j in range(len(self._X)):
                # find distance: if l1 ->manhattan, else -> euclidean
                if self.metric == "l1":
                    distance = manhattan_distance(self._X[j], X[i])
                    # if(distance == 0):
                    #     print("MAN - zero", X[i], self._X[j])
                elif self.metric == "l2":
                    distance = euclidean_distance(self._X[j], X[i])
                    # if(distance == 0):
                    #     print("EU - zero", X[i], self._X[j])
                    
                d.append([distance, self._y[j]])
            
            # sorting the distance array
            d.sort()

            # print(d)
            
            # consider only n neighbors
            for x in range(0,self.n_neighbors):
                a.append(d[x])
            
            # print(a)

            temp = {}
            
            # weighing the class
            for dist, clust in a:
                # considering weights as uniform and distance 
                if self.weights == "uniform":
                    if clust not in temp:
                        temp[clust] = 1
                    else:
                        temp[clust] = temp[clust] + 1
                # the below code was suggested by stephen
                elif self.weights == "distance":
                    if clust not in temp:
                        temp[clust] = float(1/dist)
                    else:
                        temp[clust] = temp[clust] + float(1/dist)

            # appending the max from the given temp dictionary
            prediction.append(max(temp,key=temp.get))

        return prediction