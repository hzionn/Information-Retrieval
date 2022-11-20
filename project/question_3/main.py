from numpy import argsort
from structure import read_documents, build_vector_space, similarity, use_related
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="my descriptions")
    parser.add_argument("--query", type=str, nargs="+",default="烏克蘭 大選")
    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    query = "".join(args.query)
    print(f"query: {query}")

    # there are only 1999 chinese news in total
    n_sample = 1999
    documents, sample_news = read_documents(path="ChineseNews/", n_sample=n_sample)
    assert len(documents) == n_sample
    
    use_search = True
    if use_search:
        vector_space, sample_news = build_vector_space(documents, sample_news)
        ratings = similarity(vector_space, query=query, distance="cosine")
    elif not use_search:
        ratings, sample_news = use_related(documents, sample_news, query=query)

    # Print Top 10 results
    print("#" * 50)
    print("TF-IDF Weighting + Cosine Similarity")
    top_ratings_index = argsort(ratings)[-10:][::-1]
    print(f"{'NewsID': <15}", f"{'score': >8}")
    for index in top_ratings_index:
        print(f"{sample_news[index]: <15}", f"{round(ratings[index], 7): >12}")
