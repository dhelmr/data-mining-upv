import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np

from k_means import K_means
from cluster_eval import *

from pyclustertend import hopkins, vat, ivat

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


##### Evaluation 1: Cluster Tendency #####

print("####### Results from Evaluation 1: Cluster Tendency ########")

original_X = pd.read_csv("resources/small/clean.csv")
k_means_obj = pickle.load(open("resources/small/clustering/k_2_m_2.0_1572688563.3661783","rb"))

X = np.vstack(k_means_obj.instances)
hop = hopkins(X, len(X))

print(f"The Hopkins Score: {hop}" )

#vat(X)

#ivat(X)


##### Evaluation 2: External Criteria
print("#######  Results from Evaluation 2: External Criteria #########")

partition_labels = list(original_X.label.values)
clustering_labels = list(map(int, k_means_obj.cluster_mapping))

adj_info = adjusted_mutual_info_score(partition_labels, clustering_labels)
adj_rand = adjusted_rand_score(partition_labels, clustering_labels)

print(f"Adjusted Mutual Information Score: {adj_info}")
print(f"Adjusted Rand Score: {adj_rand}")

# Idea: Matrix/Table with class vs cluster, each entry having a kind of "percentile"
# or jaccard similarity between each class/cluster

def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def percentage_coverage(original_X, k_means_obj):

    for cluster_i in k_means.instances_by_cluster:

        indices_cluster_i = k_means.instances_by_cluster[cluster_i]


        #percentage_cov =

    return percentage_cov


##### Evaluation 3: Internal Criteria #####

#ToDo: loading multiple runs of the clustering models for comparison

#plot_clusters_vs_see(X_std, 10)
#plot_silhouettes(X_std, 4, 4)
