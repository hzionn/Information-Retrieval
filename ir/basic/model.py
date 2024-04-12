"""
create a document weighting class for vector space model.
"""

from collections import Counter
from math import log
from typing import List

from tqdm import trange

from .log import setup_logger


class Model:
    """A template for all weighting models."""
    def __init__(self, logging_level="INFO"):
        self._logger = setup_logger(
            filename=__file__,
            classname=self.__class__.__name__,
            level=logging_level.upper(),
        )
        self.WEIGHTING_METHOD = ""
        self.documents_content = []
        self.vector_keyword_index = {}
        self.documents_vector = [[]]

    def weighting(self, *args, **kwargs):
        """placeholder method to be overridden by subclasses"""
        raise NotImplementedError("Subclass must implement this method")

    def _set_vector_keyword_index(self, parser):
        """set vector keyword index {vector_keyword: index}"""
        words = []
        for document_content in self.documents_content:
            tmp_words = list(parser.tokenise(document_content))
            words.extend(tmp_words)
        for _, word in enumerate(set(words)):
            if word not in self.vector_keyword_index:
                self.vector_keyword_index[word] = len(self.vector_keyword_index)

    def make_vector(self, document_content: str, parser) -> List[float]:
        """build document vector with weighting model"""
        vector = [0.0] * len(self.vector_keyword_index)
        words = list(set(parser.tokenise(document_content)))
        for word in words:
            if word in self.vector_keyword_index:
                vector[self.vector_keyword_index[word]] = self.weighting(
                    word=word,
                    words=words,
                    documents_content=self.documents_content,
                )
        return vector

    def pre_compute(self):
        """placeholder method to be overridden by subclasses"""
        return NotImplementedError("Subclass must implement this method") 

    def make_matrix(self, documents_content: List[str], parser):
        self.documents_content = documents_content
        self.pre_compute()
        self._set_vector_keyword_index(parser=parser)

        self.documents_vector = [
            self.make_vector(documents_content[i], parser=parser)
            for i in trange(len(documents_content), desc=f"Computing {self.WEIGHTING_METHOD}", ncols=90)
        ]
        return self.documents_vector

    def __str__(self):
        if not self.WEIGHTING_METHOD:
            raise NotImplementedError("Subclass should set self.WEIGHTING_METHOD attribute")
        return f"{self.WEIGHTING_METHOD}"


class TFIDF(Model):
    """
    Term frequency-inverse document frequency (TFIDF) weighting model.

    (Wikipedia) In information retrieval, `tf-idf` (also TF*IDF, TFIDF, TF-IDF, or Tf-idf),
    short for term frequency-inverse document frequency, is a measure of importance
    of a word to a document in a collection or corpus, adjusted for the fact that some
    words appear more frequently in general.
    """
    def __init__(self):
        super().__init__()
        self.WEIGHTING_METHOD = "TFIDF"
        self.documents_content = []
        self._idf_cache = {}
        self._logger.critical(f"Choosing model {self.WEIGHTING_METHOD}")

    def _tf(self, word: str, words: List[str]) -> float:
        """
        term frequency
        """
        return Counter(words)[word] / len(words)

    def _n_containing(self, word: str, documents_content: List[str]) -> int:
        if word not in self._idf_cache:
            self._idf_cache[word] = sum(
                1 for document_content in documents_content
                if word in document_content
            )
        return self._idf_cache[word]

    def _idf(self, word: str, documents_content: List[str]) -> float:
        """
        inverse document frequency
        """
        return log(len(documents_content) / (1 + self._n_containing(word, documents_content)))

    def weighting(self, word: str, words: List[str], documents_content: List[str]) -> float:
        """
        term frequency-inverse document frequency
        """
        return self._tf(word, words) * self._idf(word, documents_content)


class BM25(Model):
    """
    Okapi BM25 weighting model.

    (Wikipedia) In information retrieval,
    Okapi `BM25` (BM is an abbreviation of best matching) is a ranking function
    used by search engines to estimate the relevance of documents to a given search query.
    """
    def __init__(self, k1=1.5, b=0.75):
        """
        Args:
            k1 (float, optional): Turing parameter that calibrates the document term frequency
            scaling. Defaults to 1.5.
            b (float, optional): Turing parameter between 0 and 1 that determines the scaling
            by document length. Defaults to 0.75.
        """
        super().__init__()
        self.WEIGHTING_METHOD = "Okapi BM25"
        self.k1 = k1
        self.b = b
        self.documents_content = []
        # self.documents_vector = []
        self._idf_cache = {}
        self.avgdl = 0
        self._logger.critical(f"Choosing model {self.WEIGHTING_METHOD}")

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
        n_containing = self._n_containing(word, documents_content)
        return log((len(documents_content) - n_containing + 0.5) / (n_containing + 0.5))

    def _compute_avgdl(self):
        """
        compute average document length
        """
        total_words = sum(len(document_content.split()) for document_content in self.documents_content)
        self.avgdl = total_words / len(self.documents_content)

    def weighting(self, word: str, words: List[str], documents_content: List[str]) -> float:
        """
        term frequency-inverse document frequency
        """
        tf = self._tf(word, words)
        idf = self._idf(word, documents_content)
        return idf * ((tf * (self.k1 + 1)) / (
            tf + self.k1 * (1 - self.b + self.b * (len(words) / self.avgdl))
        ))

    def pre_compute(self):
        self._compute_avgdl()
