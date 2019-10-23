from k_means import K_means
import csv
from gensim.models import doc2vec
from collections import namedtuple
from misc.d2v_tools import get_vectors
import numpy as np

vocabulary = set()
number_of_documents = 0
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

with open('data/small/small.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    # create vocabulary
    i=1
    for row in spamreader:
        text = row[5]
        words = text.split(" ")
        for w in words:
            vocabulary.add(w)
        docs.append(analyzedDocument(text, [i]))
        i= i+1

model = doc2vec.Doc2Vec(docs, vector_size = 10, window = 300, min_count = 1, workers = 4)

vectors = get_vectors(model, docs)
print(vectors)