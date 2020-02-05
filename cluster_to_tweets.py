from kmeans.k_means import K_means_multiple_times as K_means
import kmeans.k_means as k_means
import argparse
import pandas as pd
import collections
import numpy as np
import pickle
import time
from os import path
import sys
import time
import csv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def main(dest, clean, kmeans, no_write=False, histo_dest=None):
    original = pickle.load(open(clean, "rb"))
    result = k_means.from_file(kmeans)
    total_pos_count = 0
    total_instance_count = 0
    if histo_dest is not None:
        make_histogram(original, result, dest_file=histo_dest)
        print(f"Wrote histogram to {histo_dest}")
    for cluster in result.instances_by_cluster:
        instances = result.instances_by_cluster[cluster]

        pos_count = get_positive_count(original, instances)
        pos_perc = pos_count / len(instances) * 100
        total_pos_count += pos_count
        total_instance_count += len(instances)
        
        if no_write == False:
            tweets = [load_tweet(original, i) for i in instances]
            file_name = write_to_file(dest, cluster, tweets)
        print(f"Cluster {cluster} | number of instances: {len(instances)}; positive: {pos_perc}%")
    
    total_pos_perc = total_pos_count / total_instance_count * 100
    print(f"Total number of instances: {total_instance_count} | Total positive percentage {total_pos_perc}")

def load_tweet(table, index):
    return table["original"][index]

def get_positive_count(table, instances) -> int:
    pos_count = sum(int((table["label"][instance])) for instance in instances) # positive tweets have label=1, negative label=0
    return pos_count
    
def write_to_file(destination, cluster_label, tweets):
    file_name = path.join(destination, str(cluster_label)+".txt")
    text_file = open(file_name, "w", encoding="utf-8")
    for tweet in tweets:
        text_file.write(tweet+"\n")
    text_file.close()
    return file_name

def make_histogram(original, kmeans_result, dest_file):
    x = []
    num_bins = 100

    for cluster in kmeans_result.instances_by_cluster:
        instances = kmeans_result.instances_by_cluster[cluster]
        pos_count = get_positive_count(original, instances)
        pos_perc = pos_count / len(instances) * 100
        x.append(pos_perc)    
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5,  range=[0, 100])
    axes = plt.gca()
    axes.set_ylim([0, kmeans_result.k])
    axes.set(title=f"Distribution of positive tweets for k={kmeans_result.k}, init_strategy={kmeans_result.init_strategy}, m={kmeans_result.m}",
     ylabel='Number of clusters',
     xlabel='Portion of positive tweets in %')
    plt.savefig(dest_file)


def read_src_data(path):
    return pickle.load(open(path, "rb"))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Export original tweets to a directory, grouped into files by a clustering result')
    parser.add_argument(
        '--kmeans', dest='kmeans', help='path to the kmeans result file', required=True)
    parser.add_argument('--dest', dest='dest', default="resources/kmeans_tweets/",
                        help='folder the tweets will be saved')
    parser.add_argument('--clean', dest='clean', default="resources/tweets_test_clean_original.pkl",
                    help='pkl file with the mapping of original to cleaned data')
    parser.add_argument('--no_write', dest='no_write', default=False,
                    help='set to true if no files should be written; only print statistics')                
    parser.add_argument('--histogram', dest='histogram', default=None,
                    help='Create a histogram with the positive count per cluster and write it to the specified file')                
    args = parser.parse_args()
    
    main(args.dest, args.clean, args.kmeans, no_write=args.no_write, histo_dest=args.histogram)
