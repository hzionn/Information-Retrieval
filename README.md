# Information Retrieval

## Notice

- some optimisation techniques are applied to speed up the process:
  - cache word frequency
  - use `numpy` for vector operations
  - **process larger documents first** (the remaining documents for computation will be smaller and smaller)

## Installation

Install all the dependencies using `pip` within virtual environment:

```terminal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## My Environment

- OS: MacOS
- RAM: 8G
- Python: 3.11

## Question 1

TODO:

question 1 is to build a **Vector Space Model** with **TF-IDF** weighting, and search for potential relevent documents using **cosine similarity**.

you may feed a specific query to the model by just adding `--query <your query>` right after the execution file.
if you leave `--query` blank, the query will set to be default query as "Youtube Taiwan COVID-19".
full documents size is 8000, but you can set a smaller size (not too small) for testing by adding `--n_sample <number>`.

```terminal
cd question_1
python main.py --query Youtube Taiwan COVID-19 --n_sample 500
```

last two sections of the output will be rankings (Top 10) with using (**TF-IDF** + **Cosine Similarity**) or (**TF-IDF** + **Euclidean Distance**)

## Question 2

TODO:

in question 2, we need to re-score the documents with a simple relevance feedback technique (**pseudo feedback**).

in `search`, only if we have give `old_query_vector` argurment then it will re-score the documents with new query vector.

```python
def search(self, searchList: list, distance: str, old_query_vector: list[float] = None) -> list:
    """search for documents that match based on a list of terms"""
    assert distance in ("cosine", "euclidean")
    
    print("...searching relevent documents...")
    queryVector = self.buildQueryVector(searchList)
    
    # for second retrieve
    if old_query_vector:
        # re-weighting with old query vector and new one to form a new query vector
        new_query_vector = (1 * np.array(old_query_vector)) + (0.5 * np.array(queryVector))
        ratings = util.cosine(np.array(self.documentVectors), new_query_vector)
        return ratings

    # for first retrieve
    if not old_query_vector:
        if distance == "cosine":
            ratings = util.cosine(np.array(self.documentVectors), np.array(queryVector))
        elif distance == "euclidean":
            ratings = util.euclidean(np.array(self.documentVectors), np.array(queryVector))
        return ratings, queryVector
```
