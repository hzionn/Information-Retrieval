import argparse
import os

from nltk.stem import PorterStemmer

from ir.model import BM25, TFIDF
from ir.myparser import Parser
from ir.vectorspace import VectorSpace


def main(sample_size):
    vs = VectorSpace(
        weighting_model=TFIDF(),
        parser=Parser(stemmer=PorterStemmer()),
    )
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=sample_size, to_sort=True)
    print()
    vs = VectorSpace(
        weighting_model=BM25(),
        parser=Parser(stemmer=PorterStemmer()),
    )
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=sample_size, to_sort=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Vector Space Model from a directory of documents"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size of the documents to be used for building model (-1 for all documents)",
    )

    args = parser.parse_args()
    main(sample_size=args.sample_size)
