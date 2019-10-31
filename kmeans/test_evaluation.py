from k_means import K_means
from cluster_eval import calc_see, plot_clusters_vs_see, plot_silhouettes
import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.preprocessing import StandardScaler


data = list()
for i in range(100):
    data.append([random.randint(0, 100), random.randint(0, 100)])


def plot_k_means(k_means):
    # plt.waitforbuttonpress()
    # plt.plot(k_means.instance_map)
    plt.clf()
    colors = 10 * ["r", "g", "c", "b", "k", "y"]

    for cluster_i in k_means.instances_by_cluster:
        color = colors[cluster_i]
        centroid = k_means.centroids[cluster_i]
        plt.scatter(centroid[0], centroid[1], marker="X", s=50)
        for i in k_means.instances_by_cluster[cluster_i]:
            instance = k_means.instances[i]
            plt.scatter(instance[0], instance[1], color=color, s=30)
    plt.show()


##### Evaluation 1: intra cluster cohesion with elbow method #####

# Standardize the data
X_std = StandardScaler().fit_transform(data)

#plot_clusters_vs_see(X_std, 10)

plot_silhouettes(X_std, 4, 4)