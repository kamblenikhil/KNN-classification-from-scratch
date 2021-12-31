# Auhtor - Nikhil Kamble
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

from math import log2
import numpy as np

def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    # print("jojo_EUC")
    eu_distance = np.sqrt(((x1-x2)**2).sum())
    return eu_distance

def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    # print("jojo_MAN")
    man_distance = np.abs(x1 - x2).sum()
    return man_distance