"""
a vector space model for information retrieval with weighting.
"""

import os
from random import sample
from typing import Dict

from tqdm import tqdm

from .model import TFIDF
from .myparser import Parser


class VectorSpace:
    """a vector space model for information retrieval with weighting."""

    def __init__(self, weighting_model=TFIDF(), parser=Parser()):
        """
        Args:
            weighting_model: a document weighting model
            parser: a custom document parser
        """
        self.weighting_model = weighting_model
        self.parser = parser
        self.documents_directory = ""
        self.documents_name_content = {}  # TODO: this can be a stand alone class
        print("Vector Space Initailized")

    def build(self, documents_directory: str, sample_size: int = -1, to_sort: bool = True):
        """
        a pipeline to build our vector space model

        Args:
            documents_directory(str): the directory containing all documents
            sample_size(int): the number of documents to sample
            to_sort(bool): whether to sort the documents by size or not
        """
        self.documents_directory = documents_directory
        self.documents_name_content = self._get_documents_name_content(sample_size=sample_size)
        self.documents_name_content = self._clean_all_documents()
        self.documents_name_content = self._sort_documents_by_size(to_sort=to_sort)
        self.weighting_model.compute(
            documents_content=list(self.documents_name_content.values()),
            parser=self.parser,
        )
        print("Vector Space Built")

    def _clean_single_document(self, document_content: str) -> str:
        """clean single document content"""
        cleaned_content = self.parser.tokenise(document_content)
        cleaned_content = self.parser.remove_stopwords(cleaned_content)
        return " ".join(cleaned_content)

    def _clean_all_documents(self) -> Dict[str, str]:
        """clean all documents content"""
        for name, content in tqdm(self.documents_name_content.items(), desc="Cleaning documents", ncols=90):
            self.documents_name_content[name] = self._clean_single_document(content)
        return self.documents_name_content

    def _get_documents_name_content(self, sample_size: int = -1) -> Dict[str, str]:
        """
        get all documents' name and content then map them

        Args:
            sample_size(int): the number of documents to sample (<=0 means all documents)
        """
        name_content = {}
        path = os.path.join(os.path.dirname(__file__), self.documents_directory)
        only_text_files = [f for f in os.listdir(path) if f.endswith(".txt")]
        names = sample(only_text_files, sample_size) if sample_size > 0 else only_text_files
        contents = []
        for document in names:
            with open(os.path.join(path, document), "r") as f:
                content = " ".join(f.readlines())
                contents.append(content)
        assert len(names) == len(contents)
        for name, content in zip(names, contents):
            name_content[name] = content
        return name_content

    def _sort_documents_by_size(self, to_sort: bool) -> Dict[str, str]:
        """sort documents_name_content by content size (this can greatly speed up the building process)"""
        if not to_sort:
            return self.documents_name_content
        name_content = self.documents_name_content
        sorted_name_content = {}
        name_len = {name: len(name_content[name]) for name in name_content}
        name_len_sorted = dict(sorted(name_len.items(), key=lambda x: x[1], reverse=True))
        new_name_content = {name: name_content[name] for name in name_len_sorted.keys()}
        for doc, news in tqdm(zip(list(new_name_content.keys()), list(new_name_content.values())), desc="Sorting documents by size", ncols=90):
            sorted_name_content[doc] = news
        return sorted_name_content


class Documents:
    # TODO:
    def __init__(self) -> None:
        self.directory = ""
        self.filenames = []
        self.filecontents = []


def main():
    pass


if __name__ == "__main__":
    main()
