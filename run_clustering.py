from kmeans.k_means import K_means_multiple_times as K_means
import argparse
import pandas as pd
import collections
import numpy as np
import pickle
import time
from os import path
import sys
import time


def main(src, dest_folder, k, m, n_iter, max_iter):
    data = read_src_data(src)
    vecs = data["vectors"].values

    print("### Vectors loaded, start clustering...")

    def kmeans_between(k_means, cycle):
        sys.stdout.write(".")
        sys.stdout.flush()

    def between_iter_fn(i, tmp_kmeans):
        print(f"\n### {i}/{n_iter} iteration with result: ###")
        print(tmp_kmeans)

    km = K_means(k=k, m=m, max_iterations=max_iter)  # TODO more parameter
    result = km.run(n_iter, vecs, between_iter_fn=between_iter_fn,
                    after_cluster_membership=kmeans_between)
    print(f"Best result: {result}")

    # TODO more parameter in file
    dest_file = path.join(
        dest_folder, f"k_{result.k}_m_{result.m}_{time.time()}")
    result.to_file(dest_file)
    print(f"Saved to file {dest_file}")


def read_src_data(path):
    return pickle.load(open(path, "rb"))


def old():
    csv_data = pd.read_csv(path, index_col=0)
    vecs = list()
    for index, row in csv_data.iterrows():
        print(index, row)
        vec_str = row["vectors"]
        vec_str = str.replace(vec_str, "\[", "")
        vec_str = str.replace(vec_str, "\]", "")
        vec_str = str.replace(vec_str, "\n", "")
        print(vec_str)
        np_array = np.fromstring(vec_str, dtype=float)
        print(np_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run k-means clustering')
    parser.add_argument(
        '--src', dest='src', help='path of the pre-processed and clean data with its doc2vec values', default="resources/tweets_test_vecs.csv")
    parser.add_argument('--dest', dest='dest', default="resources/clustering/",
                        help='folder where to save the clustering result')
    parser.add_argument('-k', dest="k", default=10, type=int)
    parser.add_argument('-m', dest="m", default=2, type=float,
                        help="Parameter of the Minkowski-Distance (1=Manhatten distance; 2=Euclid distance)")
    parser.add_argument("--n_iter", dest="iter", default=10, type=int,
                        help="Run k-means n times. Afterwards the best result is chosen.")
    parser.add_argument("--max_iter", dest="max_iter", default=30, type=int,
                        help="Maximum number of iterations inside one k-means run. Afterwards the best result is chosen.")
    args = parser.parse_args()

    main(args.src, args.dest, args.k, args.m, args.iter, args.max_iter)
