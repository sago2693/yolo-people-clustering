import time
import faiss
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
import glob
# Compute distances

def compute_distances(embeddings, metric,print_time=False):
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
    if print_time:
        print('Execution time: ', execution_time)
    return distances


def create_clusters(distances,embedding_keys, clustering_threshold=0.02,print_time=False):
    # Measure time for Agglomerative clustering
    start_time = time.time()
    predicted_clusters = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average',
                                        distance_threshold=clustering_threshold)
    predicted_clusters.fit(distances)
    print(predicted_clusters)
    end_time = time.time()
    execution_time = end_time - start_time

    # Retrieve the pictures that belong to each cluster
    clusters = {}
    for i, cluster in enumerate(predicted_clusters.labels_):
        if cluster not in clusters:
            clusters[cluster] = []
        try:
            clusters[cluster].append(embedding_keys[i])
        except Exception as e:
            print(f"Error encountered at index {i}: {e}")
            print('cluster labels:', predicted_clusters.labels_)
            print('embedding keys labels:',embedding_keys)
            break

    if print_time:
        print('Execution time: ', execution_time)
    return clusters


def get_mention_assignments(inp_clusters, out_clusters):
    mention_cluster_ids = {}
    out_dic = {}
    for i, c in enumerate(out_clusters):
        for m in c:
            out_dic[m] = i

    for ic in inp_clusters:
        for im in ic:
            if im in out_dic:
                mention_cluster_ids[im] = out_dic[im]

    return mention_cluster_ids

def lea(input_clusters, output_clusters, mention_to_gold):
    num, den = 0, 0

    for c in input_clusters:
        all_links = 1 if len(c)==1 else len(c) * (len(c) - 1) / 2.0
        common_links = 0

        if len(c) == 1:
            all_links = 1 #max possible links
            if c[0] in mention_to_gold and len(
                    output_clusters[mention_to_gold[c[0]]]) == 1:
                common_links = 1
        else:
            for i, m in enumerate(c):
                if m in mention_to_gold:
                    for m2 in c[i + 1:]:
                        if m2 in mention_to_gold and mention_to_gold[
                                m] == mention_to_gold[m2]:
                            common_links += 1

        num += len(c) * common_links / float(all_links)
        den += len(c)

    return num, den


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return (0 if p + r == 0
            else (1 + beta * beta) * p * r / (beta * beta * p + r))



def extract_img_name(path):
    return path.split("\\")[-1].split("/")[-1].split(".")[0]


def compute_f1_lea(embeddings,embedding_keys, distance_type,clustering_threshold,ground_truth_dict):
    distance = compute_distances(embeddings, distance_type)
    sys_clusters = {k: [extract_img_name(v) for v in values] for k, values in create_clusters(distance,embedding_keys,clustering_threshold).items()}

    sys_mention_key_clusters =  get_mention_assignments(sys_clusters.values(),ground_truth_dict.values())
    key_mention_sys_clusters = get_mention_assignments(ground_truth_dict.values(),sys_clusters.values())

    p_num, p_den = lea(list(sys_clusters.values()), list(ground_truth_dict.values()), sys_mention_key_clusters)
    r_num, r_den = lea(list(ground_truth_dict.values()), list(sys_clusters.values()), key_mention_sys_clusters)

    return f1(p_num, p_den, r_num, r_den, beta=1)


#Search best parameters


def tune_parameters(model, layers_list, distance_metrics, clustering_thresholds, crops_path,ground_truth_dict):

    
    # Initialize variables to store the best metric, threshold, and F1 score
    best_metric = None
    best_threshold = None
    best_f1 = -float('inf')  # Set to negative infinity to ensure any F1 score will be higher
    best_layer_id = None

    crops_keys = glob.glob(f"{crops_path}/*.jpg")


    # Loop through YOLO layers
    for layer in layers_list:
        try:
            # Compute embeddings
            results_on_crops = model.predict(crops_path, classes=[0], embed=[layer], project='discard')
        except Exception as e:
            print(f"Error encountered at layer {layer}: {e}")
            continue  # Skip to the next layer

        # Convert the values (vectors) into a NumPy array
        embeddings = []
        for result in results_on_crops:
            if isinstance(result, torch.Tensor):  # Ensure it's a tensor
                embeddings.append(result.flatten().numpy())
            else:
                continue
        # Convert list of embeddings to a NumPy array
        if embeddings:
            try:
                embeddings = np.array(embeddings)
                faiss.normalize_L2(embeddings)
            except Exception as e:
                print(f"Error encountered at layer {layer}: {e}")
                continue

            for metric in distance_metrics:
                for threshold in clustering_thresholds:
                    # Compute F1 score for the current combination of metric and threshold
                    f1_score = compute_f1_lea(embeddings, crops_keys, metric, threshold,ground_truth_dict)
                    
                    # Update best parameters if the current F1 score is higher
                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_metric = metric
                        best_threshold = threshold
                        best_layer_id = layer
                        print(f"New Best - Metric: {metric}, Threshold: {threshold}, Layer: {layer}, F1 Score: {best_f1}")


    return best_f1, best_metric, best_threshold, best_layer_id