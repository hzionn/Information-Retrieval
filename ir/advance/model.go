package advance

import (
	"math"
	"strings"
	"sync"
)

// Model struct to represent a document weighting model
type Model struct {
	WeightingMethod     string
	DocumentsContent    []string
	VectorKeywordIndex  map[string]int
	DocumentsVector     [][]float64
	IDFCache            map[string]float64
	AvgDocumentLength   float64
	Mutex               sync.Mutex
}

// NewModel creates a new Model instance
func NewModel() *Model {
	return &Model{
		VectorKeywordIndex: make(map[string]int),
		IDFCache:           make(map[string]float64),
	}
}

// Weighting is a placeholder method to be overridden by subclasses
func (m *Model) Weighting(word string, words []string, documentsContent []string) float64 {
	return 0.0
}

// SetVectorKeywordIndex sets the vector keyword index {vector_keyword: index}
func (m *Model) SetVectorKeywordIndex(parser *Parser) {
	words := []string{}
	for _, documentContent := range m.DocumentsContent {
		tmpWords := parser.Tokenize(documentContent)
		words = append(words, tmpWords...)
	}
	for _, word := range unique(words) {
		if _, exists := m.VectorKeywordIndex[word]; !exists {
			m.VectorKeywordIndex[word] = len(m.VectorKeywordIndex)
		}
	}
}

// MakeVector builds a document vector with the weighting model
func (m *Model) MakeVector(documentContent string, parser *Parser) []float64 {
	vector := make([]float64, len(m.VectorKeywordIndex))
	words := unique(parser.Tokenize(documentContent))
	for _, word := range words {
		if index, exists := m.VectorKeywordIndex[word]; exists {
			vector[index] = m.Weighting(word, words, m.DocumentsContent)
		}
	}
	return vector
}

// PreCompute is a placeholder method to be overridden by subclasses
func (m *Model) PreCompute() {}

// MakeMatrix creates a matrix of document vectors
func (m *Model) MakeMatrix(documentsContent []string, parser *Parser) [][]float64 {
	m.DocumentsContent = documentsContent
	m.PreCompute()
	m.SetVectorKeywordIndex(parser)

	m.DocumentsVector = make([][]float64, len(documentsContent))
	for i, content := range documentsContent {
		m.DocumentsVector[i] = m.MakeVector(content, parser)
	}
	return m.DocumentsVector
}

// TFIDF struct that embeds Model
type TFIDF struct {
	*Model
}

// NewTFIDF creates a new TFIDF instance
func NewTFIDF() *TFIDF {
	return &TFIDF{
		Model: NewModel(),
	}
}

// Weighting calculates the term frequency-inverse document frequency
func (t *TFIDF) Weighting(word string, words []string, documentsContent []string) float64 {
	tf := float64(count(word, words)) / float64(len(words))
	idf := t.IDF(word, documentsContent)
	return tf * idf
}

// IDF calculates the inverse document frequency
func (t *TFIDF) IDF(word string, documentsContent []string) float64 {
	t.Mutex.Lock()
	defer t.Mutex.Unlock()

	if idf, exists := t.IDFCache[word]; exists {
		return idf
	}

	nContaining := 0
	for _, content := range documentsContent {
		if strings.Contains(content, word) {
			nContaining++
		}
	}
	idf := math.Log(float64(len(documentsContent)) / (1 + float64(nContaining)))
	t.IDFCache[word] = idf
	return idf
}

// BM25 struct that embeds Model
type BM25 struct {
	*Model
	K1 float64
	B  float64
}

// NewBM25 creates a new BM25 instance
func NewBM25(k1, b float64) *BM25 {
	return &BM25{
		Model: NewModel(),
		K1:    k1,
		B:     b,
	}
}

// Weighting calculates the BM25 weighting
func (b *BM25) Weighting(word string, words []string, documentsContent []string) float64 {
	tf := float64(count(word, words)) / float64(len(words))
	idf := b.IDF(word, documentsContent)
	return idf * ((tf * (b.K1 + 1)) / (tf + b.K1*(1-b.B+b.B*(float64(len(words))/b.AvgDocumentLength))))
}

// IDF calculates the inverse document frequency
func (b *BM25) IDF(word string, documentsContent []string) float64 {
	b.Mutex.Lock()
	defer b.Mutex.Unlock()

	if idf, exists := b.IDFCache[word]; exists {
		return idf
	}

	nContaining := 0
	for _, content := range documentsContent {
		if strings.Contains(content, word) {
			nContaining++
		}
	}
	idf := math.Log((float64(len(documentsContent)) - float64(nContaining) + 0.5) / (float64(nContaining) + 0.5))
	b.IDFCache[word] = idf
	return idf
}

// PreCompute computes the average document length
func (b *BM25) PreCompute() {
	totalWords := 0
	for _, content := range b.DocumentsContent {
		totalWords += len(strings.Fields(content))
	}
	b.AvgDocumentLength = float64(totalWords) / float64(len(b.DocumentsContent))
}

// Helper functions
func unique(words []string) []string {
	uniqueWords := make(map[string]bool)
	for _, word := range words {
		uniqueWords[word] = true
	}
	result := []string{}
	for word := range uniqueWords {
		result = append(result, word)
	}
	return result
}

func count(word string, words []string) int {
	count := 0
	for _, w := range words {
		if w == word {
			count++
		}
	}
	return count
}
