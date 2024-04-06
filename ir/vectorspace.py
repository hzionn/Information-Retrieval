"""
a vector space model for information retrieval with weighting.
"""

import logging
import os
from random import sample
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .myparser import Parser
from .metric import Metric


def setup_logger(filename, classname, level):
    logging.basicConfig(level=level.upper())
    logger = logging.getLogger(f"{filename}.{classname}")
    logger.propagate = False  # to not process the log message in the root logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class VectorSpace:
    """a vector space model for information retrieval with weighting."""

    def __init__(self, weighting_model, parser=Parser(), logging_level="INFO"):
        """
        Args:
            weighting_model: a document weighting model
            parser: a custom document parser
            logging_level (str): logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self._logger = setup_logger(
            filename=__file__,
            classname=self.__class__.__name__,
            level=logging_level.upper(),
        )
        self.weighting_model = weighting_model
        self.parser = parser
        self.docs = None
        self.documents_vector = [[]]
        self.query_vector = []
        self._is_built = False
        self._usage = ""
        self.scores: NDArray

        self._logger.info("Vector Space Initailized")

    def build(self, documents_directory: str, sample_size: int = -1, to_sort: bool = True):
        """
        a pipeline to build our vector space model

        Args:
            documents_directory (str): the directory containing all documents
            sample_size (int): the number of documents to sample
            to_sort (bool): whether to sort the documents by size or not
        """
        self.docs = Documents(directory=documents_directory, parser=self.parser, sample_size=sample_size, to_sort=to_sort)
        self.docs.clean_all_documents()
        self.docs.sort_documents_by_size()
        self.documents_vector = self.weighting_model.make_matrix(
            documents_content=self.docs.document_contents,
            parser=self.parser,
        )
        self._is_built = True
        self._logger.info("Vector Space Built")

    def related(self, doc_index: int = -1):
        """find documents that are related to the document indexed by passed index within the documents' vector."""
        if not self._is_built:
            raise Exception("The vector space model is not built yet.")
        self._logger.info("Finding related documents")
        self.scores = Metric.cosine_similarity(np.array(self.documents_vector), np.array(self.documents_vector[doc_index]))
        self._usage = "related"
        return self.scores

    def search(self, query: str):
        """given a query, find documents that match based on the query string."""
        if not self._is_built:
            raise Exception("The vector space model is not built yet.")
        self._logger.info(f"Searching documents with query: {query}")
        self.query_vector = self.weighting_model.make_vector(query, parser=self.parser)
        self._logger.debug("Query Vector: %s", self.query_vector)
        self.scores = Metric.cosine_similarity(np.array(self.documents_vector), np.array(self.query_vector))
        self._usage = "search"
        return self.scores

    def rank(self, top_k: int = 10):
        self._logger.info("Ranking documents")
        if self._usage == "related":
            top_k_index = np.argsort(self.scores)[-top_k:][::-1]
        elif self._usage == "search":
            top_k_index = np.argsort(self.scores)[-top_k:][::-1]
        else:
            raise Exception("You need to call related() or search() first.")
        return [(self.docs.document_names[i], self.scores[i]) for i in top_k_index if self.docs is not None]


class Documents:
    def __init__(self, directory: str, parser, sample_size: int = -1, to_sort: bool = True, logging_level: str = "INFO") -> None:
        self._logger = setup_logger(
            filename=__file__,
            classname=self.__class__.__name__,
            level=logging_level.upper(),
        )
        self._directory = directory
        self._parser = parser
        self._sample_size = sample_size
        self._to_sort = to_sort
        self._info = self._get_documents_name_content()
        self._document_names = self._get_document_names()
        self._document_contents = self._get_document_contents()

        self._logger.info("Documents Initialized")

    def _get_documents_name_content(self) -> Dict[str, str]:
        """
        get all documents' name and content then map them

        Args:
            sample_size (int): the number of documents to sample (-1 means all documents)
        """
        self._logger.info("Getting documents' name and content")
        name_content = {}
        path = os.path.join(os.path.dirname(__file__), self._directory)
        only_text_files = [f for f in os.listdir(path) if f.endswith(".txt")]
        if self._sample_size == -1:
            names = only_text_files
        else:
            names = sample(only_text_files, self._sample_size)
        contents = []
        for document in names:
            with open(os.path.join(path, document), "r") as f:
                content = " ".join(f.readlines())
                contents.append(content)
        assert len(names) == len(contents)
        for name, content in zip(names, contents):
            name_content[name] = content
        return name_content

    def _get_document_names(self) -> List[str]:
        self._logger.info("Getting documents' names")
        return list(self._info.keys())

    def _get_document_contents(self) -> List[str]:
        self._logger.info("Getting documents' contents")
        return list(self._info.values())

    def _clean_single_document(self, document_content: str) -> str:
        """clean single document content"""
        cleaned_content = self._parser.tokenise(document_content)
        cleaned_content = self._parser.remove_stopwords(cleaned_content)
        return " ".join(cleaned_content)

    def _update(self):
        self._document_names = self._get_document_names()
        self._document_contents = self._get_document_contents()
        self._logger.info("Documents Updated")

    def clean_all_documents(self):
        """clean all documents content"""
        self._logger.info("Cleaning all documents")
        for name, content in tqdm(self._info.items(), desc="Cleaning documents", ncols=90):
            self._info[name] = self._clean_single_document(content)
        self._update()

    def sort_documents_by_size(self):
        """sort documents_name_content by content size (this can greatly speed up the building process)"""
        if not self._to_sort:
            self._logger.info("Documents will not sort by size")
        else:
            self._logger.info("Sorting documents by size")
            name_content = self._info
            sorted_name_content = {}
            name_len = {name: len(name_content[name]) for name in name_content}
            name_len_sorted = dict(sorted(name_len.items(), key=lambda x: x[1], reverse=True))
            new_name_content = {name: name_content[name] for name in name_len_sorted.keys()}
            for doc, news in tqdm(zip(list(new_name_content.keys()), list(new_name_content.values())), desc="Sorting documents by size", ncols=90):
                sorted_name_content[doc] = news
            self._info = sorted_name_content
        self._update()

    @property
    def info(self) -> Dict[str, str]:
        return self._info

    @property
    def document_names(self) -> List[str]:
        return self._document_names

    @property
    def document_contents(self) -> List[str]:
        return self._document_contents

    def __str__(self):
        return f"{self.info}"


def main():
    from nltk.stem.porter import PorterStemmer
    from ir.model import TFIDF
    directory = os.path.join(os.path.dirname(__file__), "sample_data", "EnglishNews")
    vs = VectorSpace(weighting_model=TFIDF(), parser=Parser(stemmer=PorterStemmer()), logging_level="INFO")
    vs.build(documents_directory=directory, sample_size=50, to_sort=True)

    related_scores = vs.related(doc_index=40)
    print(related_scores)
    search_scores = vs.search("coronavirus is a pandemic")
    print(search_scores)

    for doc, score in vs.rank(top_k=5):
        print(doc, score)


if __name__ == "__main__":
    main()
