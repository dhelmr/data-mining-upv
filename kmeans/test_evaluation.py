import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np

from k_means import K_means
from cluster_eval import *
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Y_df = pd.DataFrame(Y)

k_means = K_means(k=3,m=2)
k_means.run(X)


original_X = pd.read_csv("resources/small/clean.csv")
k_means_obj = pickle.load(open("resources/small/clustering/m_2.0/k_2_m_2.0_1572688563.3661783","rb"))
#k_means_obj = pickle.load(open("resources/small/clustering/m_2.0/k_6_m_2.0_1572690149.6818223","rb"))

give_tendency_eval(k_means)
give_external_eval(original_X, original_X, 6, 1.5)
#result_df = table_class_vs_cluster(original_X, k_means_obj, 0)
#test = give_internal_eval(original_X)

#give_tendency_eval(k_means)
#give_external_eval(Y_df, X, 25, 2, True)
