"""
a vector space model for information retrieval with weighting.
"""

import os
from random import sample
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .myparser import Parser


class VectorSpace:
    """a vector space model for information retrieval with weighting."""

    def __init__(self, weighting_model, parser=Parser()):
        """
        Args:
            weighting_model: a document weighting model
            parser: a custom document parser
        """
        self.weighting_model = weighting_model
        self.parser = parser
        self.docs = None
        self.documents_vector = [[]]


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


class Documents:
    def __init__(self, directory: str, parser, sample_size: int = -1, to_sort: bool = True) -> None:
        self._directory = directory
        self._parser = parser
        self._sample_size = sample_size
        self._to_sort = to_sort
        self._info = self._get_documents_name_content()
        self._document_names = self._get_document_names()
        self._document_contents = self._get_document_contents()
    def _get_documents_name_content(self) -> Dict[str, str]:
        """
        get all documents' name and content then map them

        Args:
            sample_size (int): the number of documents to sample (-1 means all documents)
        """
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
        return list(self._info.keys())

    def _get_document_contents(self) -> List[str]:
        return list(self._info.values())

    def _clean_single_document(self, document_content: str) -> str:
        """clean single document content"""
        cleaned_content = self._parser.tokenise(document_content)
        cleaned_content = self._parser.remove_stopwords(cleaned_content)
        return " ".join(cleaned_content)

    def _update(self):
        self._document_names = self._get_document_names()
        self._document_contents = self._get_document_contents()

    def clean_all_documents(self):
        """clean all documents content"""
        for name, content in tqdm(self._info.items(), desc="Cleaning documents", ncols=90):
            self._info[name] = self._clean_single_document(content)
        self._update()

    def sort_documents_by_size(self):
        """sort documents_name_content by content size (this can greatly speed up the building process)"""
        if not self._to_sort:
        else:
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

    def __str__(self) -> str:
        return f"{self.info}"


def main():
    from nltk.stem.porter import PorterStemmer
    from ir.model import TFIDF
    directory = os.path.join(os.path.dirname(__file__), "sample_data", "EnglishNews")
    vs = VectorSpace(weighting_model=TFIDF(), parser=Parser(stemmer=PorterStemmer()))
    vs.build(documents_directory=directory, sample_size=10, to_sort=True)


if __name__ == "__main__":
    main()
