"""
distance algorithms
"""

import numpy as np

def euclidean(x: np.array, y: np.array) -> np.array:
    """
    euclidean distance is quite simple and is the same as taught in algebra 1/2.
    it finds the literal distance between the two vectors
    """
    return np.sqrt(np.dot(x - y, x - y))

def cosine(x: np.array, y: np.array) -> np.array:
    """
    cosine similarity does not care about the vector's magnitude but just the angle.
    it always falls between [-1, 1]
    """
    return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
