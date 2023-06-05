import numpy as np

def euclidean(x: np.array, y: np.array) -> np.array:
    """
    literal distance between the two vectors
    """
    return np.sqrt(np.dot(x - y, x - y))

def cosine(x: np.array, y: np.array) -> np.array:
    """
    distance of angles (no magnitude); between [-1, 1]
    """
    return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
