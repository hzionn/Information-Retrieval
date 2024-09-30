package advance

import (
	"strings"
	"unicode"
)

// Parser struct to tokenize, remove stopwords, and stem words
type Parser struct {
	Stopwords map[string]bool
	Stemmer   func(string) string
}

// NewParser creates a new Parser instance
func NewParser(stopwords []string, stemmer func(string) string) *Parser {
	stopwordsMap := make(map[string]bool)
	for _, word := range stopwords {
		stopwordsMap[word] = true
	}
	return &Parser{
		Stopwords: stopwordsMap,
		Stemmer:   stemmer,
	}
}

// Tokenize splits the document content into words
func (p *Parser) Tokenize(content string) []string {
	words := strings.FieldsFunc(content, func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	})
	return words
}

// RemoveStopwords removes stopwords from the list of words
func (p *Parser) RemoveStopwords(words []string) []string {
	filteredWords := []string{}
	for _, word := range words {
		if !p.Stopwords[word] {
			filteredWords = append(filteredWords, word)
		}
	}
	return filteredWords
}

// StemWords applies the stemmer function to each word
func (p *Parser) StemWords(words []string) []string {
	stemmedWords := make([]string, len(words))
	for i, word := range words {
		stemmedWords[i] = p.Stemmer(word)
	}
	return stemmedWords
}
