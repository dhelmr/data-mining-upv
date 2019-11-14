# Data Mining Clustering Project (UPV)

## Requirements

### Download data set

Download [the data set from kaggle](https://www.kaggle.com/kazanova/sentiment140) and extract the file (~250MB) to the directory `resources/raw`.

## Usage

## Pre-Processing

TODO

## Clustering

The k-means clustering is executed with the python file `run_clustering.py`. There are a lot of possible arguments:

```
% python run_clustering.py --help
usage: run_clustering.py [-h] [--src SRC] [--dest DEST] [-k K_END]
                         [--k-start K_START] [-m M_LIST [M_LIST ...]]
                         [--n_iter ITER] [--max_iter MAX_ITER]
                         [--threshold THRESHOLD] [--verbose VERBOSE]
                         [--auto_increment AUTO_INC]
                         [--init_strategy INIT_STRATEGY]
                         [--exclude_last_n EXCLUDE_LAST_N]

Run k-means clustering

optional arguments:
  -h, --help            show this help message and exit
  --src SRC             path of the pre-processed and clean data with its
                        doc2vec values
  --dest DEST           folder where to save the clustering result
  -k K_END
  --k-start K_START
  -m M_LIST [M_LIST ...]
                        Parameter of the Minkowski-Distance (1=Manhatten
                        distance; 2=Euclid distance). This can also be a list
                        containing more values over which will be iterated.
  --n_iter ITER         Run k-means n times. Afterwards the best result is
                        chosen.
  --max_iter MAX_ITER   Maximum number of iterations inside one k-means run.
  --threshold THRESHOLD
                        Threshold for centroid changes. A k-means run will
                        terminate if each centroid change of the last
                        iteration is less than this threshold value.
  --verbose VERBOSE     Verbose output on/off
  --auto_increment AUTO_INC
                        Automatically increment k from k until the value
                        specified with -k is reached
  --init_strategy INIT_STRATEGY
                        Init Strategy, one of 1=RANDOM, 2=SPACE_DIVISION,
                        3=DOUBLE_K_FIRST
  --exclude_last_n EXCLUDE_LAST_N
                        Excludes the last n instances from the clustering
```

For the exact way of functioning of the algorithm, and the meaning of the parameters, please have a look at the project report. To run the clustering multiple times for different k and m, this would be a possible call:

```
python run_clustering.py --k-start 10 -k 15 -m 1.5 2 2.5 --src resources/vecs.vec --dest resources/clustering --max_iter 20 --n_iter 2 --init_strategy 1 --threshold 0.01 --auto_increment True
```

It will iterate from k=10 until k=15 and use successively m=1.5, m=2 and m=2.5 and will initialize the centroids randomly. For each configuration of k and m, it will run the algorithm twice and choose the best result afterwards.
The k-means algorithm by itself will be terminated if 20 iterations are reached or if the change of centroids is lower that the threshold of 0.01.

The clustering results will be saved to `resources/clustering` (specified with `--dest`). Each result is stored in a separate file like `k=2_m=2.0_init=1_1573230238.2595701.result`.

## Evaluation

## Generate textfiles with the tweets for each cluster

With `cluster_to_tweets.py` it is possible to apply a clustering result on the original (raw) tweets. It generates a text file for each cluster containing the original tweets of the instances that are assigned to this cluster.

You will need:

* A clustering result, like `k=50_m=2.0_init=1_1573629301.038218.result`.
* The mapping of the cleaned data instances to the original tweets, stored in a pickle file, like `resources/tweets_test_clean_original.pkl`. This file is produced in the preprocessing step.

To generate the text files with the original tweets:

```
python cluster_to_tweets.py --clean resources/tweets_test_clean_original.pkl --kmeans k=50_m=2.0_init=1_1573629301.038218.result --dest resources/kmeans_tweets/
```

This fill the directory `resources/kmeans_tweets/` with 50 files (one for each cluster). Example content of `resources/kmeans_tweets/20.txt`:

```
Good morning! Busy day today...helping Mary teach until 2:30 then getting my hair did 
Good morning everyone. Welcome new followers! Hope you all had a nice weekend. 
morning. guess who got accepted for the midlothian orchestra thing? me 
Morning all, hope everone is well! Just want to say... I HATE THE RAIN 
good morning!! headache, headache headache... 
LaKuata: good morning! well I had an appointment but..my car won't start  I'm stuck at home
morning everyone! got a throbbing head 
@Soph4Soph  Good morning for you... goodnight for me 2:27am  sleeepy
morning: job interview. not brilliant. afternoon: pilates &amp; gym. local tv actor at gym  he's fit 
Good morning! I have class today 

...
[763 more lines]
```


