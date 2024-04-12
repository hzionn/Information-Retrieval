import os

import pytest
from nltk.stem import PorterStemmer

from ir.basic.model import BM25, TFIDF
from ir.basic.myparser import Parser
from ir.basic.vectorspace import VectorSpace


@pytest.fixture
def file_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "EnglishNews")


@pytest.fixture
def tfidf():
    vs = VectorSpace(
        weighting_model=TFIDF(),
        parser=Parser(stemmer=PorterStemmer(), language="english"),
    )
    return vs


@pytest.fixture
def bm25():
    vs = VectorSpace(
        weighting_model=BM25(),
        parser=Parser(stemmer=PorterStemmer(), language="english"),
    )
    return vs


def test_tfidf_sample_size(tfidf, file_path):
    sample_size = 20
    tfidf.build(documents_directory=file_path, sample_size=sample_size)
    assert len(tfidf.docs.info) == sample_size


def test_bm25_sample_size(bm25, file_path):
    sample_size = 20
    bm25.build(documents_directory=file_path, sample_size=sample_size)
    assert len(bm25.docs.info) == sample_size


def test_tfidf_sort_document_by_size_english(tfidf, file_path):
    tfidf.build(documents_directory=file_path, sample_size=-1, to_sort=True)
    documents_length = []
    for i in [0, 50, 70, -1]:
        documents_length.append(len(tfidf.docs.document_contents[i]))
    assert documents_length[0] >= documents_length[1] >= documents_length[2] >= documents_length[3]


def test_bm25_sort_document_by_size_english(bm25, file_path):
    bm25.build(documents_directory=file_path, sample_size=-1, to_sort=True)
    documents_length = []
    for i in [0, 50, 70, -1]:
        documents_length.append(len(bm25.docs.document_contents[i]))
    assert documents_length[0] >= documents_length[1] >= documents_length[2] >= documents_length[3]


def test_tfidf_not_sort_document_by_size_english(tfidf, file_path):
    tfidf.build(documents_directory=file_path, sample_size=-1, to_sort=False)
    documents_length = []
    for i in [0, 50, 70, -1]:
        documents_length.append(len(tfidf.docs.document_contents[i]))
    assert not (documents_length[0] >= documents_length[1] >= documents_length[2] >= documents_length[3])


def test_bm25_not_sort_document_by_size_english(bm25, file_path):
    bm25.build(documents_directory=file_path, sample_size=-1, to_sort=False)
    documents_length = []
    for i in [0, 50, 70, -1]:
        documents_length.append(len(bm25.docs.document_contents[i]))
    assert not (documents_length[0] >= documents_length[1] >= documents_length[2] >= documents_length[3])


@pytest.fixture
def tfidf_with_sort(file_path):
    vs = VectorSpace(
        weighting_model=TFIDF(),
        parser=Parser(stemmer=PorterStemmer(), language="english"),
    )
    vs.build(documents_directory=file_path, sample_size=-1, to_sort=True)
    return vs


@pytest.fixture
def bm25_with_sort(file_path):
    vs = VectorSpace(
        weighting_model=BM25(),
        parser=Parser(stemmer=PorterStemmer(), language="english"),
    )
    vs.build(documents_directory=file_path, sample_size=-1, to_sort=True)
    return vs


def test_tfidf_related(tfidf_with_sort):
    pass


def test_bm25_related(bm25_with_sort):
    pass
