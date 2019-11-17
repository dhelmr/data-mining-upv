import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np

from k_means import K_means
from cluster_eval import *
from sklearn import datasets
from sklearn.cluster import KMeans


"""

### Single Test Run data #### 

original_X = pd.read_csv("resources/small/clean.csv")
#dir_path = "resources/small/clustering/"
#give_external_eval(original_X, original_X, 14, 2.0, dir_path, "single_run_m2_k13")

#k_means_obj = pickle.load(open("resources/small/clustering/m_2.0/k_2_m_2.0_1572688563.3661783","rb"))
k_means_obj = pickle.load(open("resources/small/clustering/m_2.0/k_13_m_2.0_1572698575.1576815","rb"))

give_tendency_eval(k_means_obj)
best_run = table_class_vs_cluster(original_X, k_means_obj)

#external_eval_all_files("resources/small/clustering")

"""

###### Complete run with plots over all directories ########

dirs_to_test = ["resources/results/tweets_test_vecs600.vec", "resources/results/tweets_test_vecs30.vec", "resources/results/10epoch"]

for path in dirs_to_test:

    print(f"_____ Evaluation for: {path} _______ ")
    external_eval_all_files(path)

###### Possible Check for single clustering result ########

k_means_obj = pickle.load(open("resources/results/tweets_test_vecs600.vec/m_2/init_1/k=50_m=2.0_init=1_1573629301.038218.result","rb"))
original_X = pd.read_csv("resources/small/clean.csv")
give_tendency_eval()
best_run = table_class_vs_cluster(original_X, k_means_obj)


#result_df = table_class_vs_cluster(original_X, k_means_obj, 0)
#test = give_internal_eval(original_X)

#give_tendency_eval(k_means)
#give_external_eval(Y_df, X, 25, 2, True)

