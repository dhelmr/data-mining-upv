"""
Cleans a given data frame
"""

import re
import argparse
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from resources.contractions_mapping import contractions_mapping


def clean_tweets(data, stopwords_stemming=False):
    """
    This function prepares and cleans all tweets within the given data frame. Contractions handling, lowercase
    delete special signs, @s with username, https, word stemming, stopwords removal, ...

    source: https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf
    source: https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90

    :param data: pandas data frame containing raw Sentiment140 tweets
    :param stopwords_stemming: set True if stopwords removal and word stemming should be applied
    :return: cleaned/prepared pandas data frame containing tweets
    """

    tweets = []
    labels = []
    originals = []

    print("INFO: Start cleaning tweets")
    data['label'].replace([4], 1, inplace=True)
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    for i in range(len(data.text)):
        if (i + 1) % 100000 == 0:
            print(f"INFO: {i + 1} of {len(data.text)} tweets have been processed.")

        # apply different cleaning steps on each tweet
        tweet = data.text[i]
        label = data.label[i]
        original = data.text[i]
        
        tweet = tweet.lower()
        tweet = re.sub(r"@\S+", "", tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"www.\S+", "", tweet)
        tweet = " ".join([contractions_mapping.get(i, i) for i in tweet.split()])
        tweet = re.sub("[^a-zA-Z]", " ", tweet)

        # tokenize tweet
        word_tokens = word_tokenize(tweet)

        # if parameter is set: apply stopwords removal and stemming
        if stopwords_stemming is True:
            words_temp = [w for w in word_tokens if w not in stop_words]
            word_tokens = [ps.stem(w) for w in words_temp]

        #  put words back together to tweet (eliminates 'double-spaces' which might have been created)
        tweet = []
        for w in word_tokens:
            tweet.append(w)
        tweet = " ".join(tweet)

        tweets.append(tweet)
        labels.append(label)
        originals.append(original)

    # concat to data frame and drop empty entries
    print(f"INFO: {len(data)} of {len(data)} tweets have been processed.")
    data = pd.DataFrame({'label': labels, 'text': tweets, 'original': originals})
    data_clean = drop_empty_entries(data)

    return data_clean


def drop_empty_entries(data):
    """
    function to remove rows containing at least one NaN value from a data frame

    :param data: data frame containing possible NaN entries
    :return: data frame without NaN rows
    """

    temp = len(data)

    print("INFO: Drop empty tweets (NaN rows)")
    data.dropna(inplace=True)
    data = data[~data.isin(['NaN', 'NaT', 'nan', 'nat', ' ', '']).any(axis=1)]
    data.reset_index(drop=True, inplace=True)
    print(f"INFO: {temp - len(data)} NaN-Tweets were deleted\n"
          f"INFO: Final amount of tweets: {len(data)}")

    return data


def main(src_path, dest_path):
    print("### DATA CLEANING ###")

    print(f"INFO: Read data from path: {src_path}")
    data_raw = pd.read_csv(src_path, encoding="ISO-8859–1")
    data_clean = clean_tweets(data_raw, stopwords_stemming=True)
    data_clean.to_csv(dest_path, encoding="utf-8")
    print(f"INFO: Cleaned tweets saved to path: {dest_path}")

    print("### ENDED DATA CLEANING ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DATA CLEANER')
    parser.add_argument('src_path', help='enter path to raw data')
    parser.add_argument('-d', '--dest_path', default="resources/clean/tweets_cleaned.csv",
                        help='enter file path where to save cleaned tweets')
    args = parser.parse_args()

    main(args.src_path, args.dest_path)
