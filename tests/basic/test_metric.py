import numpy as np
import pytest

from ir.basic.metric import Metric


@pytest.fixture
def query_vector():
    query_vector = np.array([1, 2, 3, 4, 5])
    return query_vector


@pytest.fixture
def documents_vector_matrix_1():
    documents_vector_matrix = np.array([
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9],
    ])
    return documents_vector_matrix


def test_cosine_similarity_1(documents_vector_matrix_1, query_vector):
    score = Metric.cosine_similarity(documents_vector_matrix_1, query_vector)
    assert score[0] == 1.0


def test_euclidean_distance_1(documents_vector_matrix_1, query_vector):
    score = Metric.euclidean_distance(documents_vector_matrix_1, query_vector)
    assert score[0] == 0.0


@pytest.fixture
def documents_vector_matrix_2():
    documents_vector_matrix = np.array([
        [5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5],
    ])
    return documents_vector_matrix


def test_cosine_similarity_2(documents_vector_matrix_2, query_vector):
    score = Metric.cosine_similarity(documents_vector_matrix_2, query_vector)
    assert score[1] == 1.0


def test_euclidean_distance_2(documents_vector_matrix_2, query_vector):
    score = Metric.euclidean_distance(documents_vector_matrix_2, query_vector)
    assert score[1] == 0.0


@pytest.fixture
def documents_vector_matrix_3():
    documents_vector_matrix = np.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    return documents_vector_matrix


def test_cosine_similarity_3(documents_vector_matrix_3, query_vector):
    score = Metric.cosine_similarity(documents_vector_matrix_3, query_vector)
    assert score[0] == score[1] == 1.0


def test_euclidean_distance_3(documents_vector_matrix_3, query_vector):
    score = Metric.euclidean_distance(documents_vector_matrix_3, query_vector)
    assert score[0] == score[1] == 0.0
