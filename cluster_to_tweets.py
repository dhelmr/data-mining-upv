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

def main(dest, clean, kmeans):
    original = pickle.load(open(clean, "rb"))
    result = k_means.from_file(kmeans)
    for cluster in result.instances_by_cluster:
        instances = result.instances_by_cluster[cluster]
        tweets = [load_tweet(original, i) for i in instances]
        pos_perc = get_positive_count(original, instances) / len(instances) * 100
        file_name = write_to_file(dest, cluster, tweets)
        print(f"Wrote {len(tweets)} tweets to {file_name}; positive: {pos_perc}%")

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
    args = parser.parse_args()
    
    main(args.dest, args.clean, args.kmeans)
