import os

from nltk.stem import PorterStemmer

from ir.vectorspace import VectorSpace
from ir.myparser import Parser

def main():
    sample_size = 1000
    vs = VectorSpace(parser=Parser(stemmer=PorterStemmer()))
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=sample_size, to_sort=True)
    # print(vs.documents_name_content)
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[0]]))
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[100]]))
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[200]]))
    # print(len(vs.documents_name_content[list(vs.documents_name_content.keys())[-1]]))


if __name__ == "__main__":
    main()
