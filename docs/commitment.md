---
title: "Data Mining Clustering Project - Commitment"
author: ["Daniel Helmrich", "Marius Kempf", "Simon Richebaecher"]
date: "2019-10-14"
titlepage: true
titlepage-color: "ededed"
footnotes-pretty: true
links-as-notes: true
---

# Introduction

As part of our clustering project we want to do a text mining task, analysing tweets according to their underlying sentiment. The neccesary steps from preprocessing to
the finished clustering results are implemented according to the given methodology ("2. PROYECTO: Clustering de documentos"). With the help of clustering we hope to
identify distinct sentiment groups/clusters. By testing different embedding and clustering techniques we try to find an optimal approach for initiating sentiment analysis for tweets. 

# The data set
For our project we are using the data set [Sentiment140](https://www.kaggle.com/kazanova/sentiment140). It contains 1.6 million tweets collected between April 2009 and June 2009. The tweets are binary labeled with positive and negative sentiment. Collecting and labelling the data was done by the researchers [Alec Go, Richa Bhayani and Lei Huang](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf).

Every entry or tweet contains the following variables:

1. target: the polarity of the tweet (0 = negative, 4 = positive)
2. id: the id of the tweet (2087)
3. date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
4. flag: the query (lyx). If there is no query, then this value is NO_QUERY.
5. user: the user that tweeted (robotickilldozr)
6. text: the text of the tweet (what is the answer to the ultimate question of life, the universe, and everything)

Since we are applying a clustering algorithm on text data, only the actual tweet text is used and further processed. 


# Technologies to be used

We agreed on using Python for the development of the the required programs. The final report will be written alongside the project collaboratively using the online tool [HackMD](https://hackmd.io) and later converted to PDF using [pandoc](https://pandoc.org/MANUAL.html). For the software development a private git repository is hosted on [github](https://github.com).

# Time plan

The following plan gives an overview about the necessary tasks and their respective targeted due date:

1. Pre-Processing: 1 week (due date: 21/10/2019)
    1. 1st exercise: Implementation and analysis
    2. 2nd exercise: BoW Implementation; Analysis of Sparse ARFF
    3. 3rd exercise: TF-IDF Implementation; Comparision BoW vs TF-IDF
    4. 4th exercise: Discuss Filtering
2. Clustering: 2 weeks (due date: 04/11/2019)
    1. Implementation of the cluster algorithm
    2. Quality indices, implementation of PCA
    3. Application of the inferred model
3. Finalize the report: 10/11/2019

As these tasks will depend on each other and for an efficient functioning of the group it is required that each member contributes equally to the work, we decided against asigning special roles or responsibilites to the group members. Everyone is fully responsible for each part of the project.