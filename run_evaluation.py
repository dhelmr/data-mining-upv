import argparse
from kmeans.cluster_eval import *


def main(src_path):
    print("###### CLUSTER VALIDATION ###")

    print(f"INFO: Read trained clustering models from {src_path}")

    #give_tendency_eval()

    print(f"_____ Evaluation for: {src_path} _______ ")
    external_eval_all_files(src_path)

    print("### ENDED VALIDATION ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLUSTER VALIDATION')
    parser.add_argument('--src', dest='src',
                        help='enter source path of trained clustering models')

    args = parser.parse_args()

    main(args.src)