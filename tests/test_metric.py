import numpy as np

from ir.metric import Metric


def test_cosine_similarity():
    vector_1 = np.array([1, 2, 3, 4, 5])
    vector_2 = np.array([1, 2, 3, 4, 5])
    score = Metric.cosine_similarity(vector_1, vector_2)
    assert score == 1.0


def test_euclidean_distance():
    vector_1 = np.array([1, 2, 3, 4, 5])
    vector_2 = np.array([1, 2, 3, 4, 5])
    score = Metric.euclidean_distance(vector_1, vector_2)
    assert score == 0.0