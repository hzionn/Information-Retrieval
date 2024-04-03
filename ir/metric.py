import numpy as np
from numpy.typing import NDArray


class Metric:
    def __init__(self):
        """a metric class to calculate the similarity between two vectors."""

    @staticmethod
    def cosine_similarity(vector_1: NDArray[np.int64], vector_2: NDArray[np.int64]) -> float:
        """
        calculate the cosine similarity between two vectors
        (the larger the cosine, the more similar the vectors are)
        cosine = (V1 * V2) / (||V1|| * ||V2||)

        Args:
            vector_1(NDArray): the first vector
            vector_2(NDArray): the second vector
        """
        cosine = np.dot(vector_1, vector_2) / (
            np.linalg.norm(vector_1, axis=0) * np.linalg.norm(vector_2)
        )
        return float(cosine)

    @staticmethod
    def euclidean_distance(vector_1: NDArray[np.int64], vector_2: NDArray[np.int64]) -> float:
        """
        calculate the euclidean distance between two vectors.
        (the smaller the distance, the more similar the vectors are)
        euclidean = sqrt(sum((V1 - V2)^2))

        Args:
            vector_1(NDArray): the first vector
            vector_2(NDArray): the second vector
        """
        euclidean = np.sqrt(((vector_1 - vector_2) ** 2).sum(axis=0))
        return float(euclidean)


if __name__ == "__main__":
    vector_1 = np.array([1, 2, 3, 4, 5])
    vector_2 = np.array([1, 2, 3, 4, 5])
    print(Metric.cosine_similarity(vector_1, vector_2))
    print(Metric.euclidean_distance(vector_1, vector_2))
