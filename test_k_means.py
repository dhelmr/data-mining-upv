from k_means import K_means
import matplotlib.pyplot as plt
import random

data = list()
for i in range(100):
    data.append([random.randint(0,100), random.randint(0,100)])
k_means = K_means(k=10)

centroids = k_means.run(data)

print(k_means.instance_map)
#plt.plot(k_means.instance_map)
colors = 10*["r", "g", "c", "b", "k", "y", "w"]
for cluster_i in k_means.instance_map:
    color = colors[cluster_i]
    centroid = k_means.centroids[cluster_i]
    plt.scatter(centroid[0], centroid[1], marker="X", s = 30)
    for i in k_means.instance_map[cluster_i]:
        instance = k_means.array[i]
        plt.scatter(instance[0], instance[1], color=color, s = 30)
plt.show()

print("hi")