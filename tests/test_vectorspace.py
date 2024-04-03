import os

from nltk.stem import PorterStemmer

from ir.vectorspace import VectorSpace
from ir.myparser import Parser


def test_sort_document_by_size_english():
    documents_length = []
    vs = VectorSpace(parser=Parser(stemmer=PorterStemmer(), language="english"))
    file_path = os.path.join(os.path.dirname(__file__), "sample_data", "EnglishNews")
    vs.build(documents_directory=file_path, sample_size=0, to_sort=True)
    for i in (0, 50, 70, -1):
        documents_length.append(len(vs.documents_name_content[list(vs.documents_name_content.keys())[i]]))
    assert documents_length[0] >= documents_length[1] >= documents_length[2] >= documents_length[3]


def test_not_to_sort_document_by_size_english():
    documents_length = []
    vs = VectorSpace(parser=Parser(stemmer=PorterStemmer(), language="english"))
    file_path = os.path.join(os.path.dirname(__file__), "sample_data", "EnglishNews")
    vs.build(documents_directory=file_path, sample_size=0, to_sort=False)
    for i in (0, 50, 70, -1):
        documents_length.append(len(vs.documents_name_content[list(vs.documents_name_content.keys())[i]]))
    assert not (documents_length[0] >= documents_length[1] >= documents_length[2] >= documents_length[3])
