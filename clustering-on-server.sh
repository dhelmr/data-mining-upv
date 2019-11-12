#!/bin/bash

# Script for running the clustering automatically on the server

mkdir -p logs

function start_clustering(){
    VECS="$1"
    K_START="$2"
    K_END="$3"
    M="$4"
    INIT_STRATEGY="$5"
    N_ITER="$6"

    BASE_VECS=$(basename $VECS)

    DEST="resources/clustering/results/$BASE_VECS/m_$M/init_$INIT_STRATEGY/"
    mkdir -p $DEST

    echo "Start clustering with K=$K_START, M=$M, INIT=$INIT_STRATEGY"
    python run_clustering.py --k-start $K_START -k $K_END -m $M --src $VECS --dest "$DEST" --max_iter 50 --n_iter $N_ITER --init_strategy $INIT_STRATEGY --auto_increment True --verbose True --exclude_last_n 100000 --threshold 0.005 > "logs/$BASE_VECS-m$M-k$K_START-kend$K_END-init$INIT-$(date)"
}

start_clustering $1 2 10 2 1 3 &
start_clustering $1 11 16 2 1 3 &
start_clustering $1 2 10 2 3 2 &
start_clustering $1 2 10 15 1 3 