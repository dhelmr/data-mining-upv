"""
TESTING - runner file for clustering project

make sure "scientific mode" is activated
"""

import pandas as pd
from gensim.models import Doc2Vec

from misc.prep_tools import clean_tweets
from misc.d2v_tools import labelling_tweets
from misc.d2v_tools import initialize_d2v_model
from misc.d2v_tools import train_model_d2v

# %% READING DATA
path = "resources/raw/Sentiment140.csv"
data = pd.read_csv(path, names=["label", "id", "date", "query", "user", "text"], encoding="ISO-8859–1")
data.head()

# %% DATA CLEANING
df = clean_tweets(data, save_to_file=True, path="resources/test_new.csv", stopwords_stemming=True)

# %% READ CLEANED DATA
df = pd.read_csv("resources/test_new.csv", index_col=0)
df.dropna(inplace=True)
df.head()

# %% TRAIN TEST SPLIT (FOR DOC2VEC MODEL TRAINING)
# build_vocab läuft in ein memory error, deshalb werden hier tweets sampled (1.6 Mio läuft irgendwie bei mir nicht
# df = df.sample(n=60000)

df_training = df.sample(frac=0.7, random_state=42)
df_training.to_csv("resources/training.csv")

df_testing = df.drop(df_training.index).reset_index(drop=True)
df_training.to_csv("resources/test.csv")

tweets_train = df_training.text
y_train = df_training.label

# %% PREPARE TRAIN TWEETS AND APPLY DOC2VEC INITIALIZATION
tweets_labeled = labelling_tweets(tweets_train)

# %% BUILD D2V MODEL (DBOW - DISTRIBUTED BAG OF WORDS, dm=0)
model_d2v_dbow = initialize_d2v_model(tweets_labeled, dm=0)
model_d2v_dbow = train_model_d2v(model_d2v_dbow, tweets_labeled,
                                 save_model=True, path="resources/models/model_d2v_dbow10.model", max_epochs=10)

# %% BUILD D2V MODEL (DM - DISTRIBUTED MEMORY, dm=1)
model_d2v_dm = initialize_d2v_model(tweets_labeled, dm=1)
model_d2v_dm = train_model_d2v(model_d2v_dm, tweets_labeled,
                               save_model=True, path="resources/models/model_d2v_dm10.model", max_epochs=10)

# %% GET VECTORS FOR CLUSTERING
# TODO: apply Doc2Vec on test data - still locking for solution to apply on whole data frame
#  (export into method/function)

model_d2v = Doc2Vec.load("resources/models/model_d2v.model")
df_testing = pd.read_csv("resources/clean/tweets_test_clean.csv", index_col=0)

df_testing['token'] = df_testing["text"].apply(lambda x: x.split())
print("INFO: tweet token created")

df_testing["vectors"] = df_testing["token"].apply(lambda x: model_d2v.infer_vector(x))
print("INFO: tweet vectors created")

# %%


