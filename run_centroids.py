"""
Applying clustering on unseen tweets and add corresponding cluster (nearest centroid)

"""


import glob
import pickle
import argparse
import pandas as pd
from kmeans.k_means import from_file


def get_original_text(test_index):
    #read from csv
    return 1


def load_resources(filename, k_means_path='kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result'):

    # cleaned data
    k_means_path = 'kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result'
    with open(filename, 'rb') as f:
        cleaned_data = pickle.load(f)

    # extract vectors
    vectors = cleaned_data["vectors"]

    # prepare k means
    k_means_path = 'kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result'
    kmeans = from_file(k_means_path)
    kmeans.n = len(vectors[0])

    return cleaned_data, vectors, kmeans


def write_to_file(instance_indexes):

    return None


def get_clusters(kmeans, vectors):
    """

    :param kmeans:
    :param vectors:
    :return:
    """
    clusters = list()
    distances = list()

    for element in vectors:
        cluster, distance = kmeans.predict(element)
        clusters.append(cluster)
        distances.append(distance)

    # concat series to data frame
    result = pd.DataFrame({'vector': vectors, 'cluster': clusters, 'distance': distances})
    return result


def main(filename, k_means_path):
    # TODO

    # load necessary things
    data_cleaned, vecs, kmeans = load_resources(filename, k_means_path)
    raw_data = pd.read_csv("resources/raw/tweets_test.csv")

    # separate testing instances (last 100.000 entries)
    testing_instances = vecs[-100000:]
    df = get_clusters(kmeans, testing_instances)

    # sample for each class
    clusters = df.cluster.unique()

    with open('tweet_comparison.txt', 'w') as file:
        for cluster in clusters:
            sample = df[df.cluster == cluster].sample(1)
            print(f"\n### New tweet (cluster {cluster}):  ###\n"
                  f"CLEANED: {data_cleaned['text'][sample.index]}\n"
                  f"RAW: {raw_data['text'][sample.index]}", file=file)

            instance_indexes = list(kmeans.instances_by_cluster[cluster])
            for index in instance_indexes[:3]:
                print(data_cleaned["text"][index], file=file)



    """
        for i in range(10):
        instance = testing_instances[377444+i]
        cluster, distance = kmeans.predict(instance)

        instance_indexes = list(kmeans.instances_by_cluster[cluster])

        print("##", cluster, distance)
        for index in instance_indexes[:10]:
            text = cleaned_data["text"][index]
            print(text)
       
    """

    print("### ENDED ####")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-f", "--file", default="resources/tweets_test_vecs600.vec", help="path to vectors file")
    parser.add_argument("-k", "--k_means", default="kmeans/", help="path to k_means objects")
    args = parser.parse_args()

    main(args.file, args.k_means)
