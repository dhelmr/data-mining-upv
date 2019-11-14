import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle
import numpy as np

from k_means import K_means
from cluster_eval import *
from sklearn import datasets
from sklearn.cluster import KMeans


##Iris run ###
iris = datasets.load_iris()
X = iris.data
Y = iris.target

Y_df = pd.DataFrame(Y)

give_external_eval(Y, X, 14, 2.0,print_message = "iris_run_m2_k12",iris= True, )

k_means = K_means(k=14,m=2)
k_means.run(X)
give_tendency_eval(k_means)
best_run = table_class_vs_cluster(Y, k_means, iris=True)

total_nbr_instances = sum(sum(best_run.iloc[:, :3].values))
total_nbr_instances