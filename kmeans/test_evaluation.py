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
##Iris run ###
iris = datasets.load_iris()
X = iris.data
Y = iris.target

Y_df = pd.DataFrame(Y)

give_external_eval(Y, X, 14, 2.0, True)

k_means = K_means(k=8,m=2)
k_means.run(X)
give_tendency_eval(k_means)
best_run = table_class_vs_cluster(Y, k_means, True)



### Test Run data #### 

original_X = pd.read_csv("resources/small/clean.csv")
dir_path = "resources/small/clustering/"
give_external_eval(original_X, original_X, 14, 2.0, dir_path)

#k_means_obj = pickle.load(open("resources/small/clustering/m_2.0/k_2_m_2.0_1572688563.3661783","rb"))
k_means_obj = pickle.load(open("resources/small/clustering/m_2.0/k_13_m_2.0_1572698575.1576815","rb"))

give_tendency_eval(k_means_obj)
best_run = table_class_vs_cluster(original_X, k_means_obj)

"""

external_eval_all_files("resources/without-100000")


#result_df = table_class_vs_cluster(original_X, k_means_obj, 0)
#test = give_internal_eval(original_X)

#give_tendency_eval(k_means)
#give_external_eval(Y_df, X, 25, 2, True)

