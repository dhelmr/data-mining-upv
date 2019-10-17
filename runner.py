"""
TESTING - runner file for clustering project

make sure "scientific mode" is activated
"""

import pandas as pd
from misc.prep_tools import clean_tweets
from misc.prep_tools import apply_train_test_split
from misc.d2v_tools import labelling_tweets
from misc.d2v_tools import initialize_d2v_model
from misc.d2v_tools import train_model_d2v
from misc.d2v_tools import get_vectors

# %%
path = "resources/raw/Sentiment140.csv"
data = pd.read_csv(path, names=["label", "id", "date", "query", "user", "text"], encoding="ISO-8859–1")
data.head()

# %% DATA CLEANING
df = clean_tweets(data, save_to_file=True, path="resources/test_new.csv", stopwords_stemming=True)

# %% READ CLEANED DATA
df = pd.read_csv("resources/test.csv", index_col=0)
df.dropna(inplace=True)
df.head()

# build_vocab läuft in ein memory error, deshalb werden hier tweets sampled (1.6 Mio läuft irgendwie bei mir nicht
df = df.sample(n=50000)

# %% PREPARE FOR DOC2VEC
tweets = df.text
tweets_labeled = labelling_tweets(tweets)

# %% DOC2VEC INITIALIZATION
model_d2v = initialize_d2v_model(tweets_labeled)

model_d2v = train_model_d2v(model_d2v, tweets_labeled)

# %% GET VECTORS FOR CLUSTERING

# hier werden nun wieder die ursprünglichen tweets verwendet und in entsprechende Vektoren (mittels
# des trainierten d2v models umgewandelt
# diese Tweets werden dann später für das Clustering verwendet

# hier könnte man dann eben den train / test split machen

tweets_train, tweets_eval = apply_train_test_split(tweets)

tweets_train_vec = get_vectors(model_d2v, tweets_train)
tweets_eval_vec = get_vectors(model_d2v, tweets_eval)

# %%

