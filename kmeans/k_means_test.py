from k_means import K_means
from sklearn.cluster import KMeans as scikit_KMeans
from k_means_demo_test import generate_random_data
import numpy as np
import sys
#
# Compares the own k_means algorithm with the scikit implementation
#

def compare_algorithms(data, k):
    sys.stdout.write(f"Compare scikit and own implentation for k={k} on {len(data)} instances... ")
    sys.stdout.flush()
    kmeans = K_means(k, max_iterations=300, verbose=False) 
    iterations = kmeans.run(data)
    

    ndinit = np.array(kmeans.initial_centroids)
    sk = scikit_KMeans(n_clusters=k, init = ndinit, max_iter=300, tol=0, n_init=1)
    sk.fit(data)

    for i in range(0, len(data)):
        instance = data[i]
        sk_centroid = sk.cluster_centers_[sk.predict([instance])][0]
        own_centroid = kmeans.centroids[kmeans.closest_centroid(i)[0]]
        assert np.allclose(sk_centroid, own_centroid), f"FAIL, instance {instance} has different centroids! {sk_centroid} (scikit) and {own_centroid} (own algorithm)"
    print("✔")

n = 1000
random_data = generate_random_data(n)
for k in range (1,10):
    compare_algorithms(random_data, k)
for k in range (1, 1000, 100):
    compare_algorithms(random_data, k)

