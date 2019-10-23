from k_means import K_means
from sklearn.cluster import KMeans as scikit_KMeans
from k_means_demo_test import generate_random_data
import numpy as np

# Compares the own k_means algorithm with the scikit implementation

def compare_algorithms(data, k):
    print(f"Compare scikit and own implentation for k={k} on {len(data)} instances")
    kmeans = K_means(k, max_iterations=300) 
    iterations = kmeans.run(data)
    

    ndinit = np.array(kmeans.initial_centroids)
    sk = scikit_KMeans(n_clusters=k, init = ndinit, max_iter=300, tol=0)
    sk.fit(data)

    print(iterations, sk.n_iter_)


    for i in range(0, len(data)):
        instance = data[i]
        sk_centroid = sk.cluster_centers_[sk.predict([instance])][0]
        own_centroid = kmeans.centroids[kmeans.closest_centroid(i)]
        assert np.allclose(sk_centroid, own_centroid), f"FAIL, instance {instance} has different centroids! {sk_centroid} (scikit) and {own_centroid} (own algorithm)"

n = 1000
random_data = generate_random_data(n)
for k in range (1,10):
    compare_algorithms(random_data, k)
for k in range (1, 1000, 10):
    compare_algorithms(random_data, k)

