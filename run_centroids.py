"""
Applying clustering on unseen tweets and add corresponding cluster (nearest centroid)

"""

import pickle
import argparse
import pandas as pd
from kmeans.k_means import from_file


def tweet_comparison(df, k_means, tweets_txt, n_samples, result_file):
    """
    Method for the tweet comparison procedure. For each found cluster a random sample
    is pulled out of the test data. Corresponding raw and cleaned tweet text is selected
    as well as n samples from the same cluster out of the training data.
    This allows to compare if the tweets within the cluster are kind of similar to the
    newly assigned tweet from the test data.
    Results are saved to an output txt file.

    :param df: data frame containing
    :param k_means: k_means model
    :param tweets_txt: data frame containing original and cleaned tweets (index consistence must be guaranteed)
    :param n_samples: number of samples to be shown for each cluster (comparison with test-tweet from cluster)
    :param result_file: result file destination
    :return: none
    """
    # Get number of clusters
    clusters = df.cluster.unique()

    with open(result_file, 'w') as file:
        # iterating over all clusters found
        for cluster in clusters:
            # get sample of test data which was assigned to the given cluster
            sample = df[df.cluster == cluster].sample(1)

            # getting original and cleaned text to test data sample
            print(f"### TWEET FROM CLUSTER {cluster}) ###\n"
                  f"ORIGINAL:   {tweets_txt['original'][sample.index]}\n"
                  f"CLEANED:    {tweets_txt['text'][sample.index]}\n"
                  f"# Samples from corresponding cluster:", file=file)

            # get corresponding instances / indices from cluster (out of training data)
            instance_indexes = list(k_means.instances_by_cluster[cluster])

            # getting original and cleaned text samples (training data) from corresponding cluster
            for index in instance_indexes[:n_samples]:
                print(f"ORIGINAL:   {tweets_txt['original'][index]}\n"
                      f"CLEANED:    {tweets_txt['text'][index]}", file=file)
            print("\n", file=file)

    print(f"INFO: Comparison saved to file '{result_file}'")


def get_clusters(k_means, vectors):
    """
    Method to assign the newly given test instances from the test data to the given k clusters.
    Resulting in an data frame containing the instance vector, assigned cluster and distance to
    the closest centroid

    :param k_means: k_means model
    :param vectors: test instances - vectors
    :return: resulting data frame containing vector, cluster and distance to closest centroid
    """
    clusters = list()
    distances = list()

    for element in vectors:
        cluster, distance = k_means.predict(element)
        clusters.append(cluster)
        distances.append(distance)

    # concat series to data frame
    result = pd.DataFrame({'vector': vectors, 'cluster': clusters, 'distance': distances})
    return result


def main(filename, k_means_path, n_samples, result_file, save):
    print("### TEST DATA PROCESSING ####")

    # load necessary things
    print("INFO: Reading necessary data and model")
    tweets_txt = pd.read_csv("resources/clean/cleaned.csv")
    data_prepared = pickle.load(open(filename, "rb"))
    vectors = data_prepared["vectors"]

    # load and prepare k_means model
    k_means = from_file(k_means_path)
    k_means.n = len(vectors[0])

    # 100.000 (unseen) instances are used for testing
    testing_instances = vectors[-100000:]

    # apply cluster assignment process for 100.000 test instances
    print("INFO: Cluster assignment started")
    df = get_clusters(k_means, testing_instances)

    # save clustered test instances
    if save is True:
        file_path = "test_instances_clustered.csv"
        df.to_csv(file_path)
        print(f"INFO: Resulting df with clustered test instances saved to '{file_path}'")

    # start cluster / tweet comparison
    tweet_comparison(df, k_means, tweets_txt, n_samples, result_file)

    print("### TEST DATA PROCESSING ENDED ####")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEST DATA PROCESSING')
    # TODO: final project structure needs to be respected - file paths!
    parser.add_argument("-f", "--file", dest="file", default="resources/tweets_test_vecs600.vec",
                        help="path to vectors file")
    # TODO: here needs to be our 'optimal' clustering model be referenced
    parser.add_argument("-k", "--k_means", dest="kmeans", default="kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result",
                        help="path to k_means objects")
    parser.add_argument("-n", "--samples", dest="samples", default=5,
                        help="number of samples per clusters")
    parser.add_argument("-d", "--dest", dest="dest", default="tweet_comparison.txt",
                        help="file path for tweet comparison result file")
    parser.add_argument("-s", "--save", dest="save", default=False, type=bool,
                        help="save resulting data frame option on/off")
    args = parser.parse_args()

    main(args.file, args.kmeans, args.samples, args.dest, args.save)
