"""
Document2Vector model training
"""

import argparse
import pandas as pd
import multiprocessing
from sklearn import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def labelling_tweets(tweets):
    """
    labels tweets with the document index (integer)
    to use this function you have to make sure, that there are no more NaN entries within the tweets

    source: https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-6-doc2vec-603f11832504

    :param tweets: tweets to be labeled
    :return: labeled tweets
    """

    labeled_tweets = []
    print(f"INFO: Start labeling tweets")

    # t:tweet; i:unique document id
    for i, t in zip(tweets.index, tweets):
        labeled_tweets.append(TaggedDocument(t.split(), ["{}".format(i)]))
    print("INFO: Tweet labeling finished")

    return labeled_tweets


def initialize_d2v_model(tweets_labeled, dm=1, negative=5, vector_size=200, min_count=2, alpha=0.065, min_alpha=0.065):
    """
    initializes an d2v model with the given parameter. Also creates a vocab based on the given tweets

    source: https://radimrehurek.com/gensim/models/doc2vec.html
    source: https://arxiv.org/pdf/1607.05368.pdf

    :param tweets_labeled: labeled tweets used to create vocab
    :param dm: 0 -> DM 'distributed memory', 1 -> DBOW 'distributed bag of words'
    :param negative:
    :param vector_size: size of vectors
    :param min_count: minimum appearance to be considered for vocab
    :param alpha: learning rate
    :param min_alpha: minimal learning rate alpha
    :return: initialized d2v model
    """

    print(f"INFO: d2v model (dm={dm}, vectorsize={vector_size}) initialization and vocab building started")

    # get amount of cpu cores & initialize d2v model
    cores = multiprocessing.cpu_count()
    print(f"SYSTEM: {cores} cores are used")

    model_d2v = Doc2Vec(dm=dm, vector_size=vector_size, negative=negative, min_count=min_count,
                    alpha=alpha, min_alpha=min_alpha)

    # load tweets into model to create vocabulary
    model_d2v.build_vocab(tweets_labeled)
    print("INFO: d2c model initialization and vocab building finished")

    return model_d2v


def train_model_d2v(model_d2v, tweets_labeled, max_epochs=20):
    """
    trains a given d2v model with the given labeled text data / tweets

    source: https://radimrehurek.com/gensim/models/doc2vec.html

    :param model_d2v: initialized (untrained) d2v model
    :param tweets_labeled: labeled tweets to use for training
    :param max_epochs: maximum amount of training epochs
    :return: trained and ready to use d2v model
    """

    print("INFO: d2c model training started")

    for epoch in range(max_epochs):
        print(f"INFO: Training epoch {epoch + 1} of {max_epochs}")

        # train model - lower training rate alpha after each epoch
        model_d2v.train(utils.shuffle(tweets_labeled), total_examples=len(tweets_labeled), epochs=1)
        model_d2v.alpha -= 0.002
        model_d2v.min_alpha = model_d2v.alpha
    print(f"INFO: d2v model training completed ({max_epochs} epochs)")

    return model_d2v


def main(src_path, dm, dest_path, epochs, vector_size):
    print("### DOC2VEC TRAINING ###")

    print(f"INFO: Read training data from {src_path}")
    data = pd.read_csv(src_path)
    tweets_training = data.text
    tweets_labeled = labelling_tweets(tweets_training)

    print(f"INFO: Start building d2v model (dm={dm})")
    model_d2v = initialize_d2v_model(tweets_labeled, dm=dm, vector_size=vector_size)

    model_d2v = train_model_d2v(model_d2v, tweets_labeled, max_epochs=epochs)

    model_d2v.save(dest_path)
    print(f"INFO: Doc2Vec model saved to {dest_path}")

    print("### ENDED DOC2VEC TRAINING ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D2V TRAINER')
    parser.add_argument('--src', dest='src',
                        help='enter source path of training data')
    parser.add_argument('--type', dest='dm', default=1, type=int,
                        help='enter d2v model type (dm=0 -> DBOW, dm=1 -> DM')
    parser.add_argument('-d', dest='dest', default="resources/models/model_d2v_new.model",
                        help='enter file path where to save d2v model')
    parser.add_argument('-e', dest='epochs', default=10, type=int,
                        help='enter wanted amount of training epochs')
    parser.add_argument('-vs', dest='vec_size', default=200, type=int,
                        help='enter wanted vector size')
    args = parser.parse_args()

    main(args.src, args.type, args.dest, args.epochs, args.vec_size)



