"""
Apply trained d2v model on unseen test data / tweets
"""


import argparse
import pandas as pd
from gensim.models import Doc2Vec
import pickle


def main(model_path, src_path, dest_path):
    print("### APPLY D2V ON TEST TWEETS ###")

    print("INFO: Loading d2v model and test data")
    model_d2v = Doc2Vec.load(model_path)
    df_testing = pickle.load(open(src_path, "rb"))

    print("INFO: Apply d2v vector transformation")
    df_testing['token'] = df_testing["text"].apply(lambda x: x.split())
    df_testing["vectors"] = df_testing["token"].apply(lambda x: model_d2v.infer_vector(x))
    print("INFO: Tweet vectors created")

    if dest_path is not None:
        pickle.dump(df_testing, open(dest_path, "wb"))
        print(f"INFO: Test vectors saved to {dest_path}")

    print("### ENDED APPLY D2V ON TEST TWEETS ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D2V TRAINER')
    parser.add_argument('-m', dest='model', default='resources/models/model_d2v_dbow_600epochs.model',
                        help='enter source path of d2v model')
    parser.add_argument('--src', dest='src', default='resources/clean/tweets_test_clean_original.pkl',
                        help='enter path of test data')
    parser.add_argument('-d', dest='dest', default="resources/tweets_test_vecs.vec",
                        help='enter path where to save transformed txt data')
    args = parser.parse_args()

    main(args.model, args.src, args.dest)
