"""
file contains all functions (tools) for data editing and data preparing/pre processing

"""

import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from resources.contractions_mapping import contractions_mapping


def clean_tweets(data, save_to_file=False, path="resources/Sentiment140_clean.csv"):
    """
    This function prepares and cleans all tweets within the given data frame. Contractions handling, lowercase
    delete special signs, @s with username, https, ...

    TODO: stopwords removal, delete short words (both did not show significant better results, but still)

    :param data: pandas data frame containing raw Sentiment140 tweets
    :param save_to_file: option to save cleaned data
    :param path: path to save cleaned data
    :return: cleaned/prepared pandas data frame containing tweets
    """

    tweets = []
    labels = []
    data['label'].replace([4], 1, inplace=True)

    for i in range(len(data.text)):
        if (i + 1) % 100000 == 0:
            print(f"INFO:{i + 1} of {len(data.text)} tweets processed.")

        # apply different cleaning steps on each tweet
        tweet = data.text[i]
        label = data.label[i]
        tweet = tweet.lower()
        tweet = re.sub(r"@\S+", "", tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"www.\S+", "", tweet)
        tweet = " ".join([contractions_mapping.get(i, i) for i in tweet.split()])
        tweet = re.sub("[^a-zA-Z]", " ", tweet)

        # tokenize tweet and put words back together (eliminates 'double-spaces' which might have been created)
        words = nltk.word_tokenize(tweet)
        tweet = []
        for w in words:
            tweet.append(w)
        tweet = " ".join(tweet)

        tweets.append(tweet)
        labels.append(label)

    print(f"INFO:{len(data.text)} of {len(data.text)} tweets processed.")
    data = pd.DataFrame({'label': labels, 'text': tweets})

    # save csv file if demanded
    if save_to_file is True:
        data = drop_empty_entries(data)
        data.to_csv(path, encoding="utf-8")
        print(f"INFO: cleaned tweets saved to path: {path}")

    return data


def drop_empty_entries(data):
    """

    :param data:
    :return:
    """

    temp = len(data)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(f"INFO: {temp - len(data)} NaN-Tweets were deleted\n"
          f"INFO: total amount of tweets: {len(data)}")

    return data


def apply_train_test_split(data, test_size=0.2, random_state=None):
    """
    function to split data set into a training and test/validation set

    :param data: complete data set
    :param test_size: percentage to be used for testing
    :param random_state: set if reproducibility is needed
    :return: data frames containing tweets for training and testing/validation
    """

    data_train, data_test = train_test_split(data, test_size=test_size, random_state=random_state)

    print(f"Total amount of tweets: {len(data)}\n"
          f"Training data:          {len(data_train)}\n"
          f"Test/Validation data:   {len(data_test)}\n")

    return data_train, data_test
