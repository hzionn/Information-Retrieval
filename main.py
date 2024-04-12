import argparse
import os
from typing import List, Tuple

from nltk.stem import PorterStemmer, SnowballStemmer

from ir.basic.model import TFIDF, BM25
from ir.basic.myparser import Parser
from ir.basic.vectorspace import VectorSpace


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build a Vector Space Model from a directory of documents."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Sample size of the documents to be used for building model (-1 for all documents)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Youtube Taiwan COVID-19",
        help="The query to search for in the documents",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser


def content_block(func):
    def warp(*args, **kwargs):
        print("#" * 30)
        print(f"{'NewsID': <15}", f"{'score': >8}")
        print("-" * 30)
        func(*args, **kwargs)
        print("#" * 30)
    return warp


@content_block
def print_ranking(ranking: List[Tuple]):
    for document, score in ranking:
        print(f"{document: <15}", f"{round(score, 7): >12}")


def main(sample_size: int, query: str, logging_level: str):
    vs = VectorSpace(
        weighting_model=TFIDF(),
        parser=Parser(stemmer=PorterStemmer()),
        logging_level=logging_level,
    )
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=sample_size, to_sort=True)
    vs.search(query=query, metric="cosine")
    ranking = vs.rank(top_k=10)
    print_ranking(ranking)

    vs = VectorSpace(
        weighting_model=TFIDF(),
        parser=Parser(stemmer=PorterStemmer()),
        logging_level=logging_level,
    )
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=sample_size, to_sort=True)
    vs.search(query=query, metric="euclidean")
    ranking = vs.rank(top_k=10)
    print_ranking(ranking)

    vs = VectorSpace(
        weighting_model=BM25(),
        parser=Parser(stemmer=SnowballStemmer()),
        logging_level=logging_level,
    )
    files_path = os.path.join(os.path.dirname(__file__), "data", "EnglishNews")
    vs.build(documents_directory=files_path, sample_size=sample_size, to_sort=True)
    vs.search(query=query, metric="euclidean")
    ranking = vs.rank(top_k=10)
    print_ranking(ranking)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(
        sample_size=args.sample_size,
        query=args.query,
        logging_level=args.logging_level.upper()
    )
