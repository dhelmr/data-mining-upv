import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import sample
from numpy.random import uniform
from math import isnan
import os

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_score
from sklearn.neighbors import NearestNeighbors

import pickle

from kmeans.k_means import K_means
from pyclustertend import hopkins, vat, ivat


def give_tendency_eval(origin_path):

    path_to_indexes_vectors = origin_path + "/tweets_training_vecs600.vec"
    original_X = pickle.load(open(path_to_indexes_vectors, "rb"))
    X_stacked = np.vstack(X.vectors.values)

    hop = hopkins(X_stacked, len(X))

    print(f"The Hopkins Score for the training data: {hop}" )


def table_class_vs_cluster(matrix_indexes, k_means_obj, iris=False):

    original_X = matrix_indexes[:len(k_means_obj.cluster_mapping)]

    if iris is True:

    ## Test for Iris data

        collect_lst = []

        original_X = pd.DataFrame(original_X)

        indices_class_2 = original_X[original_X[0] == 2].index.values
        indices_class_1 = original_X[original_X[0] == 1].index.values
        indices_class_0 = original_X[original_X[0] == 0].index.values

        for cluster_i in k_means_obj.instances_by_cluster:

            indices_cluster_i = list(k_means_obj.instances_by_cluster[cluster_i])

            tot_nbr_2 = len(np.intersect1d(indices_class_2, indices_cluster_i))
            tot_nbr_1 = len(np.intersect1d(indices_class_1, indices_cluster_i))
            tot_nbr_0 = len(np.intersect1d(indices_class_0, indices_cluster_i))

            if sum([tot_nbr_0, tot_nbr_1, tot_nbr_2]) == 0:

                perc_of_biggest_cluster = 0.0
                weight_of_cluster = 0.0

            else:

                perc_of_biggest_cluster = max(tot_nbr_0, tot_nbr_1, tot_nbr_2)/ sum([tot_nbr_0, tot_nbr_1, tot_nbr_2])
                weight_of_cluster = sum([tot_nbr_0, tot_nbr_1, tot_nbr_2]) / len(original_X)

            collect_lst.append([tot_nbr_0, tot_nbr_1, tot_nbr_2, perc_of_biggest_cluster, weight_of_cluster])

        comparison_df = pd.DataFrame(collect_lst, columns=["class_0","class_1", "class_2","match_percentage","cluster_weight"])

    else:

        collect_lst = []

        indices_class_1 = list(original_X[original_X.label == 1.0].index.values)
        indices_class_0 = list(original_X[original_X.label == 0.0].index.values)

        for cluster_i in k_means_obj.instances_by_cluster:

            indices_cluster_i = list(k_means_obj.instances_by_cluster[cluster_i])

            tot_nbr_1 = len(np.intersect1d(indices_class_1, indices_cluster_i))
            tot_nbr_0 = len(np.intersect1d(indices_class_0, indices_cluster_i))


            if sum([tot_nbr_0, tot_nbr_1]) == 0:

                perc_of_biggest_cluster = 0.0
                weight_of_cluster = 0.0

            else:
                perc_of_biggest_cluster = max(tot_nbr_0, tot_nbr_1) / sum([tot_nbr_0, tot_nbr_1])
                weight_of_cluster = sum([tot_nbr_0, tot_nbr_1]) / len(original_X)

            collect_lst.append([tot_nbr_0, tot_nbr_1, perc_of_biggest_cluster, weight_of_cluster])

        comparison_df = pd.DataFrame(collect_lst, columns=["class_0", "class_1", "match_percentage", "cluster_weight"])

    return comparison_df


def give_external_eval(original_X, X=None, max_range=10, m=2.0, dir_path=None, plot_title=None, iris=False):


    if iris is True:

        ### This validation is creating new k-means objects for increasing k

        avrg_perc_lst = []
        list_k = list(range(2, max_range))

        for k in list_k:

            k_means = K_means(k=k, m=m)
            #With X as the instances from the iris dataset and original_X as the list of labels
            k_means.run(X)

            result_df = table_class_vs_cluster(original_X, k_means, True)
            weights = result_df


            avr_perc = np.average(result_df.match_percentage, weights=result_df.cluster_weight)
            avrg_perc_lst.append(avr_perc)

        plt.figure(figsize=(6, 6))
        plt.plot(list_k, avrg_perc_lst, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Weighted average % of match: cluster vs class')
        plt.show()

    else:

        avrg_perc_lst = []
        files = os.listdir(f"{dir_path}")

        if ".DS_Store" in files:
            files.remove(".DS_Store")

        if plot_title + ".png" in files:
            files.remove(plot_title + ".png")

        list_k = sorted([int(file.split("_")[0].split("=")[1]) for file in files])
        sorted_files =  sorted(files, key=lambda s: int(s.split("_")[0].split("=")[1]))

        for k, filename in zip(list_k, sorted_files):

            file = f"{dir_path}/" + filename

            if os.stat(file).st_size == 0:
                avr_perc = 0
                avrg_perc_lst.append(avr_perc)
                continue

            k_means = pickle.load(open(file, "rb"))
            result_df = table_class_vs_cluster(original_X, k_means)

            avr_perc = np.average(result_df.match_percentage,weights=result_df.cluster_weight)
            avrg_perc_lst.append(avr_perc)

        x_axis_labels = list_k

        plt.figure(figsize=(6, 6))
        plt.plot(x_axis_labels, avrg_perc_lst, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Weighted average % of match: cluster vs class')
        plt.title(plot_title)

        plt.savefig(dir_path + "/" + plot_title)

        print(f"Results for {plot_title} saved to {dir_path}")


def external_eval_all_files(origin_path):

    path_to_indexes_vectors = origin_path + "/tweets_training_vecs600.vec"
    original_X = pickle.load(open( path_to_indexes_vectors, "rb"))
    dirs_1 = os.listdir(origin_path)

    if ".DS_Store" in dirs_1:
        dirs_1.remove(".DS_Store")

    if "tweets_training_vecs600.vec" in dirs_1:
        dirs_1.remove("tweets_training_vecs600.vec")

    for dir_1 in dirs_1:

        origin_path_2 = origin_path + "/" + dir_1
        dirs_2 =  os.listdir(origin_path_2)

        if ".DS_Store" in dirs_2:
            dirs_2.remove(".DS_Store")

        for dir_2 in dirs_2:

            dir_path = origin_path_2 + "/" + dir_2
            plot_title = f"Agreement_plot_{dir_1}_{dir_2}"
            give_external_eval(original_X, dir_path = dir_path, plot_title= plot_title)


################## Unused-Depreciated-Runtime Issues #################

def plot_silhouettes(instance_matrix, origin_path, plot_title):
    """
    "translated" from https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

    :param X_std:
    :param max_range:
    :return:
    """
    files = os.listdir(f"{origin_path}")

    if ".DS_Store" in files:
        files.remove(".DS_Store")

    if plot_title + ".png" in files:
        files.remove(plot_title + ".png")

    sorted_files = sorted(files, key=lambda s: int(s.split("_")[0].split("=")[1]))
    #sorted_files = sorted(os.listdir("resources/small/clustering/m_1.5"), key=str.lower)
    not_odd_files = sorted_files[0::2]  # return just every 2nd item
    print(not_odd_files)

    list_k = sorted([int(file.split("_")[0].split("=")[1]) for file in not_odd_files])

    for k, filename in zip(list_k, not_odd_files):

        file = origin_path + "/" + filename

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Kmeans object
        k_means = pickle.load(open(file, "rb"))

        labels = k_means.cluster_mapping
        centroids = np.vstack(k_means.centroids)
        #X = np.vstack(k_means.instances)

        instance_matrix_less = instance_matrix[:len(labels)]

        # Get silhouette samples
        silhouette_vals = silhouette_samples(instance_matrix_less, labels)

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
        ax2.scatter(instance_matrix[:, 0], instance_matrix[:, 1], c=labels)
        ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='r', s=250)
        ax2.set_xlim([-2, 2])
        ax2.set_xlim([-2, 2])
        ax2.set_xlabel('Eruption time in mins')
        ax2.set_ylabel('Waiting time to next eruption')
        ax2.set_title('Visualization of clustered data', y=1.02)
        ax2.set_aspect('equal')
        plt.tight_layout()
        plt.suptitle(f'Silhouette analysis using k = {k}',
                     fontsize=16, fontweight='semibold', y=1.05)
        plt.show();


def give_internal_eval(instance_matrix, origin_path, plot_title):
    print("#######  Results from Evaluation 3: Internal Criteria #########")

    plot_silhouettes(instance_matrix, origin_path, plot_title)
    #plot_clusters_vs_see(origin_path, plot_title)

def give_internal_eval_all_files(origin_path):

    dirs_1 = os.listdir(origin_path)

    X = pickle.load(open(origin_path + "/tweets_training_vecs600.vec", "rb"))
    instance_matrix = np.vstack(X.vectors.values)

    if ".DS_Store" in dirs_1:
        dirs_1.remove(".DS_Store")

    if "tweets_training_vecs600.vec" in dirs_1:
        dirs_1.remove("tweets_training_vecs600.vec")

    for dir_1 in dirs_1:

        origin_path_2 = origin_path + "/" + dir_1
        dirs_2 = os.listdir(origin_path_2)

        if ".DS_Store" in dirs_2:
            dirs_2.remove(".DS_Store")

        for dir_2 in dirs_2:
            dir_path = origin_path_2 + "/" + dir_2
            plot_title = f"Agreement_plot_{dir_1}_{dir_2}"
            give_internal_eval(instance_matrix, dir_path, plot_title=plot_title)


def plot_clusters_vs_see(origin_path, plot_title):
    """
    execute k-means clustering with multiple iterations of increasing k and plots the error against the
    number of k clusters


    :param X_std: standardized dataset
    :param max_range: maximum k
    :return:
    """

    list_sse = []

    files = os.listdir(f"{origin_path}")
    X = pickle.load(open(origin_path + "/tweets_training_vecs600.vec", "rb"))
    instance_matrix = np.vstack(X.vectors.values)

    if ".DS_Store" in files:
        files.remove(".DS_Store")

    if plot_title + ".png" in files:
        files.remove(plot_title + ".png")

    list_k = sorted([int(file.split("_")[0].split("=")[1]) for file in files])
    sorted_files = sorted(files, key=lambda s: int(s.split("_")[0].split("=")[1]))
    # list_k = list(range(max_range))
    # sorted_files = sorted(os.listdir("resources/small/clustering/m_1.5"), key=str.lower)

    for k, filename in zip(list_k, sorted_files):
        file = origin_path + "/" + filename

        k_means = pickle.load(open(file, "rb"))
        list_sse.append(calc_see(k_means))

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, list_sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.show()