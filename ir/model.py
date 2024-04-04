"""
create a document weighting class for vector space model.
"""

from collections import Counter
from math import log
from typing import List

from tqdm import trange


class Model:
    def __init__(self):
        self.WEIGHTING_METHOD = ""
        self.vector_keyword_index = {}
    
    def _set_vector_keyword_index(self, parser):
        """set vector keyword index {vector_keyword: index}"""
        words = []
        for document_content in self.documents_content:
            tmp_words = list(parser.tokenise(document_content))
            words.extend(tmp_words)
        for _, word in enumerate(set(words)):
            if word not in self.vector_keyword_index:
                self.vector_keyword_index[word] = len(self.vector_keyword_index)
    
    def _make_vector(self, document_content: str, parser) -> List[float]:
        """build document vector with weighting model"""
        vector = [0.0] * len(self.vector_keyword_index)
        words = list(set(parser.tokenise(document_content)))
        for word in words:
            if word not in self.vector_keyword_index:
                raise ValueError(f"word '{word}' not in vector keyword index")
            else:
                vector[self.vector_keyword_index[word]] = self._weighting(
                    word=word,
                    words=words,
                    documents_content=self.documents_content,
                )
        return vector
    
    def _weighting(self, *args, **kwargs):
        """placeholder method to be overridden by subclasses"""
        raise NotImplementedError("Subclass must implement this method")

    def compute(self, documents_content: List[str], parser):
        self.documents_content = documents_content
        if len(self.vector_keyword_index) == 0:
            self._set_vector_keyword_index(parser=parser)

        self.documents_vector = [
            self._make_vector(documents_content[i], parser=parser)
            for i in trange(len(documents_content), desc=f"Computing {self.WEIGHTING_METHOD}", ncols=90)
        ]
        return self.documents_vector


class TFIDF(Model):
    def __init__(self):
        super().__init__()
        self.WEIGHTING_METHOD = "TFIDF"
        self.documents_content = []
        self.documents_vector = []
        self._idf_cache = {}

    def _tf(self, word: str, words: List[str]) -> float:
        """
        term frequency
        """
        return Counter(words)[word] / len(words)

    def _n_containing(self, word: str, documents_content: List[str]) -> int:
        if word not in self._idf_cache:
            self._idf_cache[word] = sum(1 for document_content in documents_content if word in document_content)
        return self._idf_cache[word]

    def _idf(self, word: str, documents_content: List[str]) -> float:
        """
        inverse document frequency
        """
        return log(len(documents_content) / (1 + self._n_containing(word, documents_content)))

    def _weighting(self, word: str, words: List[str], documents_content: List[str]) -> float:
        """
        term frequency-inverse document frequency
        """
        return self._tf(word, words) * self._idf(word, documents_content)


class BM25(Model):
    def __init__(self):
        super().__init__()
        self.WEIGHTING_METHOD = "BM25"
    
    def _weighting(self, *args, **kwargs):
        pass