"""
Program splits original raw data into training and test csv file and saves them to resources
"""

import argparse
import pandas as pd


def main(src_path, path_train, path_test, frac):
    print("### DATA SPLITTING ####")

    print(f"INFO: Reading data from '{src_path}'")
    data_raw = pd.read_csv(src_path, names=["label", "id", "date", "query", "user", "text"], encoding="ISO-8859–1")

    print(f"INFO: Splitting data using fraction {frac}")
    data_training = data_raw.sample(frac=frac, random_state=42)
    data_testing = data_raw.drop(data_training.index)

    data_training = data_training.reset_index(drop=True, inplace=False)
    data_training.to_csv(path_train)
    print(f"INFO: Training data saved to '{path_train}'\n"
          f"INFO: Training data shape: {data_training.shape}")

    data_testing = data_testing.reset_index(drop=True, inplace=False)
    data_testing.to_csv(path_test)
    print(f"INFO: Test data saved to '{path_test}'\n"
          f"INFO: Training data shape: {data_testing.shape}")

    print("### ENDED DATA SPLITTING ####")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DATA SPLITTING TOOL')
    parser.add_argument('-src', '--src_path', default="./resources/raw/Sentiment140.csv",
                        help='enter path to raw data')
    parser.add_argument('-tr', '--path_train', default="resources/raw/tweets_train.csv",
                        help='enter path where to save tweets for training')
    parser.add_argument('-te', '--path_test', default="resources/raw/tweets_test.csv",
                        help='enter path where to save tweets for testing')
    parser.add_argument('-f', '--frac', default=0.7, type=float,
                        help='enter fraction used for training, i.e. 0.7')
    args = parser.parse_args()

    main(args.src_path, args.path_train, args.path_test, args.frac)
