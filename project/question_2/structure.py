from os import listdir
from random import sample
from time import time
from VectorSpace import VectorSpace


def read_documents(path: str = "../EnglishNews/", n_sample: int = 8000):
    print(f"reading {n_sample} documents...")
    sample_news = sample(list(listdir(path)), n_sample)
    documents: list[str] = []
    for sample_new in sample_news:
        with open(f"{path}{sample_new}", "r") as f:
            f: list[str] = f.readlines()
            f: str = " ".join(f)
            documents.append(f)
    return documents, sample_news

def build_vector_space(documents: list[str], sample_news: list[str]):
    print("building vector space...")
    start = time()
    vector_space = VectorSpace(documents = documents, sample_news=sample_news)
    sample_news = vector_space.sample_news
    print(f"time used in building vector space: {round(time() - start, 3)}(s)")
    return vector_space, sample_news

def similarity(vector_space, query: str, distance: str, sample_news: list):
    print()
    print("calculating similarity...")
    start = time()
    # first retrieve
    ratings, old_query_vector = vector_space.search(query.split(), distance)
    # second retrieve with pseudo feedback
    new_ratings = vector_space.feedback(ratings, sample_news, old_query_vector)
    print(f"time used in calculating similarity: {round(time() - start, 3)}(s)")
    return new_ratings

def use_related(documents: list, sample_news: list, query: str):
    documents.append(query)
    sample_news.append("query itself")
    assert documents[-1] == query

    vector_space, sample_news = build_vector_space(documents, sample_news)

    print("calculating cosine similarity...")
    start = time()
    ratings = vector_space.related(-1)
    print(f"time used in calculating cosine similarity: {round(time() - start, 3)}(s)")
    return ratings, sample_news