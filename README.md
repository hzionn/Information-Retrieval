# Information Retrieval - Project 1

## File Structure

in my code directory for question 1 and question 2, `/EnglishNews/` should at this directory.

and others will stay at their own question folder

```terminal
project
├── EnglishNews
│   └── ...
├── question_1
│   └── ...
├── question_2
│   └── ...
├── question_3
│   └── ...
├── question_4
│   └── ...
├── requirements.txt
└── README.md
```

## Notice

- all the file's paths I wrote are fixed, and there will be instructions for where they should be.
- for each problem set, I have them all in 4 separate directories. (`./question_1`, `./question_2`, `question_3`, `question_4`)
- you **MUST** execute any `main.py` in their own directory!
- they **DO NOT** share any of the files.

## Installation

Install all the dependencies using `pip`:

```terminal
pip install -r requirements.txt
```

## My Environment

- OS: MacOS
- RAM: 8G

## Question 1

question 1 aim to build a **Vector Space Model** with **TF-IDF** weighting, and search for potential relevent documents using **cosine similarity**.

you may feed a specific query to the model by just adding `--query <your query>` right after the execution file. FYI, if you leave `--query` blank, the query will set to be default query as "Youtube Taiwan COVID-19".

```terminal
python main.py --query Youtube Taiwan COVID-19
```

the output will pretty much look the same as below.
(some informations are provided to track the modelling process)

last two sections of the output will be rankings (Top 10) with using (**TF-IDF** + **Cosine Similarity**) or (**TF-IDF** + **Euclidean Distance**)

```terminal
query: Youtube Taiwan COVID-19
reading 8000 documents...
building vector space...
...sort out documents size from biggest to smallest...
...cleaning documents (tokenise, remove stopwords)...
100%|███████████████████████████████| 8000/8000 [01:03<00:00, 126.22it/s]
...get vector keyword index...(found 101192 vector keywords)...
...generating vectors for each document...
100%|████████████████████████████████| 8000/8000 [13:47<00:00,  9.67it/s]
time used in building vector space: 891.514(s)

calculating similarity...
time used in calculating similarity: 95.073(s)
##################################################
TF-IDF Weighting + Cosine Similarity
NewsID             score
News2230.txt        0.421779
News668.txt        0.4103842
News1240.txt       0.4083983
News1679.txt       0.4059166
News623.txt        0.3643374
News7403.txt        0.345974
News796.txt        0.3361785
News7570.txt       0.3352863
News2401.txt       0.3352863
News820.txt        0.3326828

calculating similarity...
time used in calculating similarity: 100.337(s)
##################################################
TF-IDF Weighting + Euclidean Distance
NewsID             score
News7207.txt        5.637092
News2424.txt        5.637092
News7467.txt       5.6335065
News2039.txt       5.5376266
News7513.txt       5.5259606
News2180.txt       5.5259606
News1292.txt       5.4798343
News2761.txt       5.4564449
News1048.txt       5.4377753
News7586.txt       5.4307282
```

unix command (`time`) come in handy for showing a total running time:

```terminal
time python main.py --query Youtube Taiwan COVID-19
```

```terminal
query: Youtube Taiwan COVID-19
reading 8000 documents...
building vector space...
...sort out documents size from biggest to smallest...
...cleaning documents (tokenise, remove stopwords)...
.
.
.
python main.py  1055.77s user 76.35s system 97% cpu 19:25.90 total
```

## Question 2

in question 2, we need to re-score the documents with a simple relevance feedback technique (**pseudo feedback**).

since question 2 does not provides sample output, here are some block sections to show how's the works done.

AGAIN, they **DO NOT** share any of the files.

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

## Question 3

AGAIN, they **DO NOT** share any of the files.

seem like 9 out of 10 matched with TA's result.

```terminal
query: 烏克蘭 大選
reading 1999 documents...
building vector space...
...sort out documents size from biggest to smallest...
...cleaning documents (tokenise, remove stopwords)...
100%|█████████████████████████| 1999/1999 [00:08<00:00, 229.03it/s]
...get vector keyword index...(found 49956 vector keywords)...
...generating vectors for each document...
100%|█████████████████████████| 1999/1999 [00:19<00:00, 103.54it/s]
time used in building vector space: 32.536(s)

calculating similarity...
time used in calculating similarity: 9.222(s)
##################################################
TF-IDF Weighting + Cosine Similarity
NewsID             score
News200049.txt     0.1755873
News200847.txt       0.16763
News200892.txt     0.1658111
News200908.txt     0.1544334
News200056.txt     0.1487441
News200137.txt     0.1473342
News200565.txt     0.1400956
News200071.txt      0.134985
News200898.txt     0.1291964
News200000.txt     0.1235518
```

## Question 4

AGAIN, they **DO NOT** share any of the files.

```terminal
reading 1460 documents...
building vector space...
...sort out documents size from biggest to smallest...
...cleaning documents (tokenise, remove stopwords)...
100%|█████████████████████████| 1460/1460 [00:02<00:00, 493.02it/s]
...get vector keyword index...(found 7944 vector keywords)...
...generating vectors for each document...
100%|█████████████████████████| 1460/1460 [00:04<00:00, 352.18it/s]
time used in building vector space: 7.423(s)
100%|██████████████████████████████| 76/76 [01:13<00:00,  1.03it/s]

TF-IDF retrieve
##################################################
MRR@10 0.032042995692190025
MAP@10 0.11247444759208314
RECALL@10 0.04125495333377129
```
