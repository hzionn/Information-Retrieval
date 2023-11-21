from numpy import argsort
from os import listdir
import pandas as pd
from tqdm import tqdm
from structure import read_documents, build_vector_space, similarity

def open_file(file: str):
    with open(file, "r") as f:
        return f.read()


if __name__ == "__main__":

    df_rel = pd.read_csv("smaller_dataset/rel.tsv", delimiter="\t", header=None)

    # there are 1460 in total
    n_sample = 1460
    documents, sample_news = read_documents(path="smaller_dataset/collections/", n_sample=n_sample)
    assert len(documents) == n_sample
    
    queries_order = listdir("smaller_dataset/queries/")
    queries = [open_file(f"smaller_dataset/queries/{query}") for query in queries_order]
    assert len(queries_order) == len(queries) == 76

    vector_space, sample_news = build_vector_space(documents, sample_news)

    n_queries = 76
    queries_results = []

    print("calculating similarity...")
    for query in tqdm(queries[:n_queries]):
        each_result = []
        ratings = similarity(vector_space, query=query, distance="cosine")

        top_ratings_index = argsort(ratings)[-10:][::-1]
        for index in top_ratings_index:
            each_result.append(sample_news[index]) # (filename, its rating)
        queries_results.append(each_result) # [all filenames, their ratings]



    all_avg_precision = []
    all_avg_recall = []
    all_rrk = []
    for qr, q, qo in zip(queries_results, queries[:n_queries], queries_order[:n_queries]):
        #print(qo)
        #print(qr)

        rel_str = df_rel[df_rel[0] == f"{qo[:-4]}"][1].values[0]
        rel_str = rel_str.replace("[", "")
        rel_str = rel_str.replace("]", "")
        rel_list = rel_str.split(", ")
        #print(rel_list)

        count = 0
        qr_precision = []
        qr_recall = []
        qr_rrk = []
        for i, r in enumerate(qr):
            r_num = r[1:-4]
            if r_num in rel_list:
                count += 1
                if count == 1:
                    rrk = 1/(i+1) # 第一個出現正確答案的位置
                    qr_rrk.append(rrk)
                precision = count / (i+1)
                recall = count / len(rel_list)
                qr_precision.append(precision)
                qr_recall.append(recall)

        
        #print(count)

        avg_precision = sum(qr_precision) / len(rel_list)
        #print(f"average precision: {avg_precision}")
        all_avg_precision.append(avg_precision)

        avg_recall = sum(qr_recall) / len(rel_list)
        #print(f"average recall: {avg_recall}")
        all_avg_recall.append(avg_recall)

        avg_rrk = sum(qr_rrk) / len(rel_list)
        all_rrk.append(avg_rrk)

    print()
    print("TF-IDF retrieve")
    print("#" * 50)
    print(f"MRR@10 {sum(all_rrk) / 76}")
    print(f"MAP@10 {sum(all_avg_precision) / 76}")
    print(f"RECALL@10 {sum(all_avg_recall) / 76}")