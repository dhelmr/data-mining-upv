"""
Applying clustering on unseen tweets and add corresponding cluster (nearest centroid)

"""


import glob
import pickle
import argparse
from kmeans.k_means import from_file


def main(filename, k_means_path):

    # tweets (pickle)
    with open(filename, 'rb') as f:
        vecs = pickle.load(f)

    # TODO glob
    kmeans = 'kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result'

    new_instances = vecs[-100000:]
    kmeans = from_file(kmeans, new_instances)

    # iterating k_means models from path
    # for filename in glob.glob(f'{k_means_path}/*.result'):

    # compare with and calculate metrics

    # save results to file / document performance

    print("### ENDED ####")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-f", "--file", default="resources/tweets_test_vecs600.vec", help="path to vectors file")
    parser.add_argument("-k", "--k_means", default="kmeans/", help="path to k_means objects")
    args = parser.parse_args()

    main(args.file, args.k_means)
