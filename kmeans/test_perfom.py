from k_means import K_means
from sklearn.cluster import KMeans as scikit_KMeans
import numpy as np
import sys
import random
#
# Compares the own k_means algorithm with the scikit implementation
#

def generate_random_data(instances_n, dim=25):
    data = list()
    for i in range(0,instances_n):
        instance = np.zeros(dim)
        for d in range(0,dim):
            instance[d] = random.randint(-1000,1000)
        data.append(instance)
    return data

def run_kmeans(data, k):
    print("K",k)
    kmeans = K_means(k,m=2, max_iterations=300, verbose=False) 
    iterations = kmeans.run(data)

n = 5000
for x in range(3):
    random_data = generate_random_data(n)
    for i in range(3):
        print(i)
        run_kmeans(data=random_data, k=4)

