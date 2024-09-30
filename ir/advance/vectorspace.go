package advance

import (
	"log"
	"math"
	"os"
	"strings"
	"sync"
)

// VectorSpace struct for information retrieval with weighting
type VectorSpace struct {
	WeightingModel  *Model
	Parser          *Parser
	DocumentsVector [][]float64
	QueryVector     []float64
	IsBuilt         bool
	Scores          []float64
	Usage           string
	Logger          *log.Logger
}

// NewVectorSpace creates a new VectorSpace instance
func NewVectorSpace(weightingModel *Model, parser *Parser, logger *log.Logger) *VectorSpace {
	return &VectorSpace{
		WeightingModel: weightingModel,
		Parser:         parser,
		IsBuilt:        false,
		Logger:         logger,
	}
}

// Build builds the vector space model
func (vs *VectorSpace) Build(documentsContent []string) {
	vs.DocumentsVector = vs.WeightingModel.MakeMatrix(documentsContent, vs.Parser)
	vs.IsBuilt = true
	vs.Logger.Println("Vector Space Built")
}

// Related finds documents related to the document indexed by docIndex
func (vs *VectorSpace) Related(metric string, docIndex int) []float64 {
	if !vs.IsBuilt {
		vs.Logger.Fatal("The vector space model is not built yet.")
	}
	vs.Logger.Println("Finding related documents")
	switch metric {
	case "cosine":
		vs.Scores = vs.WeightingModel.CosineSimilarity(vs.DocumentsVector, vs.DocumentsVector[docIndex])
	case "euclidean":
		vs.Scores = vs.WeightingModel.EuclideanDistance(vs.DocumentsVector, vs.DocumentsVector[docIndex])
	default:
		vs.Logger.Fatal("Invalid metric, choose either 'cosine' or 'euclidean'")
	}
	vs.Usage = "related"
	return vs.Scores
}

// Search finds documents that match the query string
func (vs *VectorSpace) Search(query string, metric string) []float64 {
	if !vs.IsBuilt {
		vs.Logger.Fatal("The vector space model is not built yet.")
	}
	vs.Logger.Println("Searching documents with query:", query)
	vs.QueryVector = vs.WeightingModel.MakeVector(query, vs.Parser)
	switch metric {
	case "cosine":
		vs.Scores = vs.WeightingModel.CosineSimilarity(vs.DocumentsVector, vs.QueryVector)
	case "euclidean":
		vs.Scores = vs.WeightingModel.EuclideanDistance(vs.DocumentsVector, vs.QueryVector)
	default:
		vs.Logger.Fatal("Invalid metric, choose either 'cosine' or 'euclidean'")
	}
	vs.Usage = "search"
	return vs.Scores
}

// Rank ranks the documents based on the scores
func (vs *VectorSpace) Rank(topK int) []int {
	vs.Logger.Println("Ranking documents")
	if vs.Usage == "related" || vs.Usage == "search" {
		return argsort(vs.Scores, topK)
	}
	vs.Logger.Fatal("You need to call Related() or Search() first.")
	return nil
}

// Documents struct for handling documents
type Documents struct {
	Directory        string
	Parser           *Parser
	SampleSize       int
	ToSort           bool
	DocumentNames    []string
	DocumentContents []string
	Logger           *log.Logger
}

// NewDocuments creates a new Documents instance
func NewDocuments(directory string, parser *Parser, sampleSize int, toSort bool, logger *log.Logger) *Documents {
	return &Documents{
		Directory:  directory,
		Parser:     parser,
		SampleSize: sampleSize,
		ToSort:     toSort,
		Logger:     logger,
	}
}

// Load loads the documents from the directory
func (d *Documents) Load() {
	d.Logger.Println("Loading documents")
	// Implement loading logic here
}

// Clean cleans the documents
func (d *Documents) Clean() {
	d.Logger.Println("Cleaning documents")
	// Implement cleaning logic here
}

// Sort sorts the documents by size
func (d *Documents) Sort() {
	d.Logger.Println("Sorting documents by size")
	// Implement sorting logic here
}

// Helper function to sort and get top K indices
func argsort(scores []float64, topK int) []int {
	type kv struct {
		Key   int
		Value float64
	}
	var ss []kv
	for i, v := range scores {
		ss = append(ss, kv{i, v})
	}
	sort.Slice(ss, func(i, j int) bool {
		return ss[i].Value > ss[j].Value
	})
	var indices []int
	for i := 0; i < topK && i < len(ss); i++ {
		indices = append(indices, ss[i].Key)
	}
	return indices
}
