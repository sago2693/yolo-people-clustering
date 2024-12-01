import faiss
from sklearn.cluster import AgglomerativeClustering
import time
import numpy as np
import pandas as pd
import re 


# Compute distances

def compute_distances(embeddings, metric):
    start_time = time.time()
    if metric == 'cosine':
        index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    elif metric == 'euclidean':
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        raise ValueError('Invalid metric')
    

    index.add(embeddings)

    D, I = index.search(embeddings, len(embeddings))
    reordered_distances = np.zeros_like(D)
    rows = np.arange(len(embeddings))[:, np.newaxis]
    reordered_distances[rows, I] = D

    distances = reordered_distances if metric == 'euclidean' else 1 - reordered_distances
    end_time = time.time()
    execution_time = end_time - start_time
    print('Execution time: ', execution_time)
    return distances