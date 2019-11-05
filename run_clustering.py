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

def main(src, dest_folder, end_k, start_k, m_list, n_iter, max_iter, threshold, verbose, init_strategy, auto_inc, exclude_last):
    data = read_src_data(src)
    vecs = data["vectors"].values
    if exclude_last != 0:
        vecs = vecs[:-exclude_last]
        print(f"Exclude last {exclude_last} instances from clustering")
    print("### Vectors loaded, start clustering...")

    if auto_inc == False:
        start_k = end_k

    for m in m_list:
        for k in range(start_k, end_k+1):
            print(f"### k={k}, m={m}")
            km = K_means(k=k, m=m, max_iterations=max_iter, threshold = threshold, verbose = verbose, init_strategy=init_strategy)
            result = km.run(n_iter, vecs)
            print(f"### Best result for k={k}, m={m}: {result}")

            # TODO more parameter in file
            dest_file = path.join(
                dest_folder, f"k={result.k}_m={result.m}_init={result.init_strategy}_{time.time()}")
            result.to_file(dest_file)
            print(f"Saved to file {dest_file}")


def read_src_data(path):
    return pickle.load(open(path, "rb"))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run k-means clustering')
    parser.add_argument(
        '--src', dest='src', help='path of the pre-processed and clean data with its doc2vec values', default="resources/tweets_test_vecs.vec")
    parser.add_argument('--dest', dest='dest', default="resources/clustering/",
                        help='folder where to save the clustering result')
    parser.add_argument('-k', dest="k_end", default=10, type=int)
    parser.add_argument('--k-start', dest="k_start", default=2, type=int)
    parser.add_argument('-m', dest="m_list", default=2,nargs='+', type=float,
                        help="Parameter of the Minkowski-Distance (1=Manhatten distance; 2=Euclid distance). This can also be a list containing more values over which will be iterated.")
    parser.add_argument("--n_iter", dest="iter", default=10, type=int,
                        help="Run k-means n times. Afterwards the best result is chosen.")
    parser.add_argument("--max_iter", dest="max_iter", default=30, type=int,
                        help="Maximum number of iterations inside one k-means run.")
    parser.add_argument("--threshold", dest="threshold", default=0.001, type=float,
                        help="Threshold for centroid changes. A k-means run will terminate if each centroid change of the last iteration is less than this threshold value.")
    parser.add_argument("--verbose", dest="verbose", default=True, type=bool,
                        help="Verbose output on/off")
    parser.add_argument("--auto_increment", dest="auto_inc", default=False, type=bool,
                    help="Automatically increment k from k until the value specified with -k is reached")
    init_strategy_values = ", ".join( [f"{s.value}={s.name}" for s in k_means.Init_Strategy])
    parser.add_argument("--init_strategy", dest="init_strategy", default=k_means.Init_Strategy.RANDOM, type=int,
                        help="Init Strategy, one of "+init_strategy_values)
    parser.add_argument("--exclude_last_n", dest="exclude_last_n", default=0, type=int,
                    help="Excludes the last n instances from the clustering")
    args = parser.parse_args()
    
    main(args.src, args.dest, args.k_end, args.k_start, args.m_list, args.iter, args.max_iter, args.threshold, args.verbose, args.init_strategy, args.auto_inc, args.exclude_last_n)
