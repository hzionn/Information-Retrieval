"""
create a document weighting class for vector space model.
"""

from collections import Counter
from math import log
from typing import List

from tqdm import trange


class TFIDF:
    def __init__(self):
        self.documents_content = []
        self.vector_keyword_index = {}
        self.documents_vector = []
        self._idf_cache = {}

    def n_containing(self, word: str, documents_content: List[str], document_content_index: int) -> int:
        return sum(1 for document_content in documents_content[document_content_index:] if word in document_content)

    def _tf(self, word: str, words: List[str]) -> float:
        """
        term frequency
        """
        return Counter(words)[word] / len(words)

    def _idf(self, word: str, documents_content: List[str], document_content_index: int) -> float:
        """
        inverse document frequency
        """
        if word not in self._idf_cache:
            self._idf_cache[word] = log(
                len(documents_content) / (1 + self.n_containing(word, documents_content, document_content_index))
            )
        return self._idf_cache[word]

    def _tfidf(self, word: str, words: List[str], documents_content: List[str], document_content_index: int) -> float:
        """
        term frequency-inverse document frequency
        """
        return self._tf(word, words) * self._idf(word, documents_content, document_content_index)

    def _set_vector_keyword_index(self, parser):
        """set vector keyword index {vector_keyword: index}"""
        words = []
        for document_content in self.documents_content:
            tmp_words = list(parser.tokenise(document_content))
            words.extend(tmp_words)

        for _, word in enumerate(set(words)):
            if word not in self.vector_keyword_index:
                self.vector_keyword_index[word] = len(self.vector_keyword_index)

    def _make_vector(self, document_content: str, document_content_index: int, parser) -> List[float]:
        """build document vector with weighting model"""
        vector = [0.0] * len(self.vector_keyword_index)
        words = list(set(parser.tokenise(document_content)))
        for word in words:
            if word not in self.vector_keyword_index:
                raise ValueError(f"word '{word}' not in vector keyword index")
            else:
                vector[self.vector_keyword_index[word]] = self._tfidf(
                    word,
                    words,
                    self.documents_content,
                    document_content_index,
                )
        return vector

    def compute(self, documents_content: List[str], parser):
        self.documents_content = documents_content
        if len(self.vector_keyword_index) == 0:
            self._set_vector_keyword_index(parser=parser)

        self.documents_vector = [
            self._make_vector(documents_content[i], i, parser=parser)
            for i in trange(len(documents_content), desc="Computing tf-idf", ncols=90)
        ]
        return self.documents_vector
