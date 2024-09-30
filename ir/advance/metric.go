package advance

import (
	"math"
)

// Metric struct to calculate the similarity between two vectors
type Metric struct{}

// CosineSimilarity calculates the cosine similarity between documents' vector matrix and query vector
func (m *Metric) CosineSimilarity(matrix [][]float64, vector []float64) []float64 {
	cosine := make([]float64, len(matrix))
	for i := range matrix {
		dotProduct := 0.0
		normMatrix := 0.0
		normVector := 0.0
		for j := range matrix[i] {
			dotProduct += matrix[i][j] * vector[j]
			normMatrix += matrix[i][j] * matrix[i][j]
			normVector += vector[j] * vector[j]
		}
		cosine[i] = dotProduct / (math.Sqrt(normMatrix) * math.Sqrt(normVector))
	}
	return cosine
}

// EuclideanDistance calculates the euclidean distance between documents' vector matrix and query vector
func (m *Metric) EuclideanDistance(matrix [][]float64, vector []float64) []float64 {
	euclidean := make([]float64, len(matrix))
	for i := range matrix {
		sum := 0.0
		for j := range matrix[i] {
			sum += (matrix[i][j] - vector[j]) * (matrix[i][j] - vector[j])
		}
		euclidean[i] = math.Sqrt(sum)
	}
	return euclidean
}
