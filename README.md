# Data Mining Clustering Project (UPV)

## Requirements

### Software

Python 3.7 is needed for all parts of the software. Install required libraries with

```
pip install -r requirements.txt
```

### Download data set

Download [the data set from kaggle](https://www.kaggle.com/kazanova/sentiment140) and extract the file (~250MB) to the directory `resources/raw`.

## Usage

After downloading the data set, there are a couple of steps that need to be run:

1. Pre-Process the data
2. Create the Doc2Vec model
3. Cluster the instances with k-means.
4. Evaluate the clustering results

Each step yields one or more output files, so that they can be run independently from each other. As some of the steps will take much time, you can download example results that we provide [here](https://www.dropbox.com/sh/farrz7t1ri3p58e/AACygkgYPPqqT9hsz4AVjmfoa?dl=0).

It is also possible to use a smaller dataset, which can be found in `resources/small`. It only contains the first 1000 rows of the original data set. It can be used for quickly testing the clustering and evaluation, but it probably won't result in an usable doc2vec model.

There are also some additional steps that follow the evaluation, and can be used for visualizing the cluster results or introducing new instances. The following sections give a basic overview of how to execute each step.

### Pre-Processing

The data pre processing consists of multiple steps: data splitting (data_splitting.py) data cleaning (data_cleaning.py).

To split the original data parameters like data source, and train/test destination files need to be given as well as the used fraction for the training data.

```
python data_splitting.py -src resources/raw/Sentiment140.csv -tr resources/raw/tweets_train.csv -te resources/raw/tweets_test.csv -f 0.7
```

After splitting the data, the cleaning can be executed. You need to deploy a file path to data which should be cleaned and a target file path
Note that cleaning hugh amount of tweets takes some time. The progress is shown in the terminal.

```
python data_cleaner.py --src resources/raw/tweets_train.csv -d resources/clean/tweets_train_clean_original.pkl
```

And for the test split:


```
python data_cleaner.py --src resources/raw/tweets_test.csv -d resources/clean/tweets_test_clean_original.pkl
```

### Doc2Vec

Starting the doc2vec training requires some paraemter given. The following shows and explains which ones can be set. Furthermore the finally used parameter setting is explained in the project report. 
The scientific source for our setting is the paper from [Lau and Baldwin (2016)](https://arxiv.org/pdf/1607.05368.pdf).

```
usage: d2v_trainer.py [-h] [--src SRC] [--type DM] [-d DEST] [-e EPOCHS]
                      [-vs VEC_SIZE]

D2V TRAINER

optional arguments:
  -h, --help    show this help message and exit
  --src SRC     enter source path of training data
  --type DM     enter d2v model type (dm=0 -> DBOW, dm=1 -> DM
  -d DEST       enter file path where to save d2v model
  -e EPOCHS     enter wanted amount of training epochs
  -vs VEC_SIZE  enter wanted vector size
```

To start the d2v model training, this would be a possible call to use (note that the training can take quite some time and be aware of memory errors during vocabulary creation):

```
python d2v_trainer.py --src resources/clean/tweets_train_clean_original.pkl --type 1 -d resources/models/model_d2v.mod
el -e 100 -vs 200
```

After the d2v models is created and trained, it can be applied on new data to infer vectors from the given texts. This
is done by the file `d2v_apply.py`. To run it, the following command can be used:

```
python d2v_apply.py -m resources/models/model_d2v_dbow_600epochs.model -src resources/clean/tweets_test_clean_original.pkl -d resources/tweets_test_vecs.vec
```

### Clustering

The k-means clustering is executed with the python file `run_clustering.py`. There are a lot of possible arguments:

```
% python run_clustering.py --help
sage: run_clustering.py [-h] [--src SRC] [--dest DEST] [-k K_END]
                         [--k-start K_START] [-m M_LIST [M_LIST ...]]
                         [--n_iter ITER] [--max_iter MAX_ITER]
                         [--threshold THRESHOLD] [--verbose VERBOSE]
                         [--auto_increment AUTO_INC]
                         [--init_strategy INIT_STRATEGY]
                         [--exclude_last_n EXCLUDE_LAST_N]
                         [--double_k_result DOUBLE_K_RESULT]

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
                        Automatically increment k from --k-start until the
                        value specified with -k is reached
  --init_strategy INIT_STRATEGY
                        Init Strategy, one of 1=RANDOM, 3=DOUBLE_K_FIRST
  --exclude_last_n EXCLUDE_LAST_N
                        Excludes the last n instances from the clustering
  --double_k_result DOUBLE_K_RESULT
                        if init_strategy=3 is set, you can specify the
                        double_k result here, so that k-means don't have to be
                        run twice
```

For the exact way of functioning of the algorithm, and the meaning of the parameters, please have a look at the project report. To run the clustering multiple times for different k and m, this would be a possible call:

```
python run_clustering.py --k-start 10 -k 15 -m 1.5 2 2.5 --src resources/tweets_test_vecs600.vec --dest resources/clustering/ --max_iter 20 --n_iter 2 --init_strategy 1 --threshold 0.01 --auto_increment True
```

It will iterate from k=10 until k=15 and use successively m=1.5, m=2 and m=2.5 and will initialize the centroids randomly. For each configuration of k and m, it will run the algorithm twice and choose the best result afterwards.
The k-means algorithm by itself will be terminated if 20 iterations are reached or if the change of centroids is lower that the threshold of 0.01.

The clustering results will be saved to `resources/clustering` (specified with `--dest`). Each result is stored in a separate file like `k=2_m=2.0_init=1_1573230238.2595701.result`.

### Evaluation

The evaluation runs in one session over all trained clustering models from the given source path. Concrete evaluation indexes are printed
in the console while figures of the class vs cluster validation are saved in the respective directories of the models.
```
usage: run_evaluation.py  [--src SRC]

CLUSTER VALIDATION

optional arguments:
  -h, --help    show this help message and exit
  --src SRC     enter source path of trained clustering models

```

Note that for this step, you need to specify a directory with the following structure:
```
resources/clustering_results
├── m_15
│   └── init_1
        |   k=2_m=15.0_init=1_1573226907.528137.result
│       └── k=3_m=15.0_init=1_1573229907.912137.result
└── m_2
    ├── init_1
    │   └── k=2_m=2.0_init=1_1573226908.124172.result
    └── init_3
        └── k=2_m=2.0_init=1_1573226908.124172.result
```
The k-means results from the last step must be copied manually to the right sub-directory.

To start the evaluation process:

```
python run_evaluation.py --src resources/clustering_results
```

### Generate text files with the tweets for each cluster

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

### Testing new instances

run_centroids.py

Since the k-means clustering model is not trained with all data, we can apply it now on unseen data. 100.000 instances
were separated for this. This tool assigns the 'new' instances to the previously found clusters. Furthermore, for
each cluster a sample test-instance is compared with n instances from that cluster used for training. The results are
saved into a text file for visual and manual examination.

```
usage: run_centroids.py [-h] [--src SRC] [--k_means MODEL] [-n SAMPLES]
                        [-d DEST] [-s SAVE]

TEST DATA PROCESSING

optional arguments:
  -h, --help       show this help message and exit
  --src SRC        path to vectors file
  --kmeans MODEL  path to k_means objects
  -n SAMPLES       number of samples per clusters
  -d DEST          file path for tweet comparison result file
  -s SAVE          save resulting data frame option on/off
```

The following provides a sample command containing parameter to run the comparison of new tweet instances.
```
python run_centroids.py --src resources/tweets_test_vecs600.vec --kmeans kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result -n 5 -d tweet_comparison.txt
```

### Visualize the cluster content using word clouds

Furthermore it is possible to generate word clouds for all clusters found by the applied k-means algorithm. This
requires the .txt files created by `cluster_to_tweets.py`. The world word clouds are saved to `resources/kmeans_tweets/wordclouds/`
by default. Run the following command to start generating word clouds:

```
python word_cloud.py -src resources/kmeans_tweets/ -d resources/kmeans_tweets/wordclouds/
```

### HTML visualization

With `make-html.py` it is possible to serve a little HTML presentation that contains the generated wordclouds and original tweet files that were created in the steps above.

Example:

```
python make-html.py --texts resources/kmeans_tweets --wordclouds resources/kmeans_tweets/wordclouds --port 8101
```

Then, open a webbrowser and go to `localhost:8101`.
