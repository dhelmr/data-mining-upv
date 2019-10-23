from k_means import K_means
import matplotlib.pyplot as plt
import random
import time
import sys

if len(sys.argv) != 2:
    instances_n = 100
else:
    instances_n = int(sys.argv[1])

def generate_random_data(instances_n, filter_fn=lambda x,y: True):
    data = list()
    for i in range(instances_n):
        x = random.randint(0,instances_n)
        y = random.randint(0,instances_n)
        if filter_fn(x,y):
            data.append([x, y])
    return data

def filter(x,y):
    return not (x > instances_n*0.5 and x < instances_n*0.6) and not (y > instances_n*0.5 and y < instances_n*0.6)
data = generate_random_data(instances_n, filter)

if __name__ == "main":
    #plt.ion()
    plt.clf()

    def plot_k_means(k_means):
        plt.waitforbuttonpress()
        #plt.plot(k_means.instance_map)
        plt.clf()
        colors = 10*["r", "g", "c", "b", "k", "y"]
        
        for cluster_i in k_means.instances_by_cluster:
            color = colors[cluster_i]
            centroid = k_means.centroids[cluster_i]
            plt.scatter(centroid[0], centroid[1], marker="X", s = 50)
            for i in k_means.instances_by_cluster[cluster_i]:
                instance = k_means.instances[i]
                plt.scatter(instance[0], instance[1], color=color, s = 30)
        plt.draw()

    # execute k-means clustering
    k_means = K_means(k=4,m=1)
    def plot():
        plot_k_means(k_means)
    k_means.after_centroid_calculation = plot
    k_means.after_cluster_membership = plot
    k_means.run(data)

    plot_k_means(k_means)
    plt.show()