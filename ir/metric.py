import numpy as np
from numpy.typing import NDArray


class Metric:
    def __init__(self):
        """a metric class to calculate the similarity between two vectors."""

    @staticmethod
    def cosine_similarity(matrix: NDArray[np.int64], vector: NDArray[np.int64]) -> NDArray:
        """
        calculate the cosine similarity between documents' vector matrix and query vector.
        (the larger the cosine, the more similar the vectors are)
        cosine = (V1 * V2) / (||V1|| * ||V2||)

        Args:
            vector_1 (NDArray): the M*N documents' vector matrix
            vector_2 (NDArray): the query vector
        """
        cosine = np.dot(matrix, vector) / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
        )
        return cosine

    @staticmethod
    def euclidean_distance(matrix: NDArray[np.int64], vector: NDArray[np.int64]) -> NDArray:
        """
        calculate the euclidean distance between documents' vector matrix and query vector.
        (the smaller the distance, the more similar the vectors are)
        euclidean = sqrt(sum((V1 - V2)^2))

        Args:
            vector_1 (NDArray): the M*N documents' vector matrix
            vector_2 (NDArray): the query vector
        """
        euclidean = np.sqrt(((matrix - vector) ** 2).sum(axis=1))
        return euclidean


if __name__ == "__main__":
    documents_matrix = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
    ])
    query_vector = np.array([1, 2, 3, 4, 5])
    print(Metric.cosine_similarity(documents_matrix, query_vector))
    print(Metric.euclidean_distance(documents_matrix, query_vector))
