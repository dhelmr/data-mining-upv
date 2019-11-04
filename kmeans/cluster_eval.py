from k_means import K_means
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

def calc_see(k_means):

    """
    first working draft with external function, not validated yet.

    :param k_means:
    :return:
    """

    cohesion_partition = []

    for cluster_i in k_means.instances_by_cluster:

        cohesions_cluster = []

        centroid_i = k_means.centroids[cluster_i]

        for index_i in k_means.instances_by_cluster[cluster_i]:

            instance = k_means.instances[index_i]
            distance_list = []
            distance_list.append((k_means.distance(instance, centroid_i)) ** 2)

        cohesions_cluster.append(np.sum(distance_list))

    cohesion_partition = np.sum(cohesions_cluster)

    return cohesion_partition






def plot_clusters_vs_see(X_std, max_range):
    """
    execute k-means clustering with multiple iterations of increasing k and plots the error against the
    number of k clusters


    :param X_std: standardized dataset
    :param max_range: maximum k
    :return:
    """

    list_sse = []
    list_k = list(range(1, max_range))

    for k_i in list_k:
        k_means = K_means(k=k_i, m=1)
        k_means.run(X_std)
        list_sse.append(calc_see(k_means))

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, list_sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.show()


def plot_silhouettes(X_std, k, max_range):

    """
    "translated" from https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
    
    :param X_std: 
    :param max_range: 
    :return: 
    """

    list_k = list(range(2, max_range+1))

    for i, k in enumerate(list_k):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Run the Kmeans algorithm
        k_means = K_means(k=k, m=1)
        k_means.run(X_std)
        labels = k_means.cluster_mapping
        centroids = np.vstack(k_means.centroids)

        # Get silhouette samples
        silhouette_vals = silhouette_samples(X_std, labels)

        # Silhouette plot
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        avg_score = np.mean(silhouette_vals)
        ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
        ax1.set_yticks([])
        ax1.set_xlim([-0.1, 1])
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster labels')
        ax1.set_title('Silhouette plot for the various clusters', y=1.02);

        # Scatter plot of data colored with labels
        ax2.scatter(X_std[:, 0], X_std[:, 1], c=labels)
        ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
        ax2.set_xlim([-2, 2])
        ax2.set_xlim([-2, 2])
        ax2.set_xlabel('Eruption time in mins')
        ax2.set_ylabel('Waiting time to next eruption')
        ax2.set_title('Visualization of clustered data', y=1.02)
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.suptitle(f'Silhouette analysis using k = {k}',
                     fontsize=16, fontweight='semibold', y=1.05)\

        plt.show();
