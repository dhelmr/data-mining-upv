"""
file contains all tools or functions in context to doc2vec

"""

import numpy as np
import multiprocessing
from sklearn import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

VECTOR_SIZE = 200


def labelling_tweets(tweets):
    """
    labels tweets with the document index (integer)
    to use this function you have to make sure, that there are no more NaN entries within the tweets

    :param tweets: tweets to be labeled
    :return: labeled tweets
    """

    labeled_tweets = []
    print("INFO: tweet labeling started")

    # t:tweet; i:unique document id
    for i, t in zip(tweets.index, tweets):
        labeled_tweets.append(TaggedDocument(t.split(), ["{}".format(i)]))
    print("INFO: tweet labeling finished")

    return labeled_tweets


def initialize_d2v_model(tweets_labeled, dm=0, negative=5, min_count=2, alpha=0.065, min_alpha=0.065):
    """
    initializes an d2v model with the given parameter. Also creates a vocab based on the given tweets

    :param tweets_labeled: labeled tweets used to create vocab
    :param dm: 0 -> DM 'distributed memory', 1 -> DBOW 'distributed bag of words'
    :param negative:
    :param min_count: minimum appearance to be considered for vocab
    :param alpha: learning rate
    :param min_alpha: minimal learning rate alpha
    :return: initialized d2v model
    """

    print(f"INFO: d2v model (dm={dm}) initialization and vocab building started")

    # get amount of cpu cores & initialize d2v model
    cores = multiprocessing.cpu_count()
    print(f"INFO: {cores} cores are used")

    model_d2v = Doc2Vec(dm=dm, vector_size=VECTOR_SIZE, negative=negative, min_count=min_count,
                    alpha=alpha, min_alpha=min_alpha)

    # load tweets into model to create vocabulary
    model_d2v.build_vocab(tweets_labeled)
    print("INFO: d2c model initialization and vocab building finished")

    return model_d2v


def train_model_d2v(model_d2v, tweets_labeled, save_model=False, path="resources/models/model_d2v.model",  max_epochs=30):
    """
    trains a given d2v model with the given labeled text data / tweets

    :param model_d2v: initialized (untrained) d2v model
    :param tweets_labeled: labeled tweets to use for training
    :param save_model: option to save trained model
    :param path: to save trained model
    :param max_epochs: maximum amount of training epochs
    :return: trained and ready to use d2v model
    """

    print("INFO: d2c model training started")

    for epoch in range(max_epochs):
        print(f"INFO: training epoch {epoch + 1} of {max_epochs}")

        # train model - lower training rate alpha after each epoch
        model_d2v.train(utils.shuffle(tweets_labeled), total_examples=len(tweets_labeled), epochs=1)
        model_d2v.alpha -= 0.002
        model_d2v.min_alpha = model_d2v.alpha
    print(f"INFO: d2v model training completed ({max_epochs} epochs)")

    # if option selected: save model to given path
    if save_model is True:
        model_d2v.save(path)
        print(f"INFO: d2v-model saved to path: {path}")

    return model_d2v


def get_vectors(model, data):
    """
    CLASS IS ONLY NEEDED FOR CLASSIFICATION (if applied)
    function to create vector representations of given data. The vectors trained by the doc2vec model are used

    :param model: d2v model
    :param data: text data which is going to transformed to vector representation
    :return: matrix containing the data as vector representation (each row represents one text file or tweet)
    """

    # delete temporary training data of model (no more updates, only querying, reduce memory usage)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # empty matrix with dimension 'amount tweets' x 200
    vectors = np.zeros((len(data), VECTOR_SIZE))

    n = 0
    for i in data.index:
        # i-ter vector from Doc2Vec in n-te row of matrix
        vectors[n] = model.docvecs[str(i)]
        n += 1

    return vectors
