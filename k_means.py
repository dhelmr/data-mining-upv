from enum import Enum
import random
import numpy as np
import math


class Init_Strategy(Enum):
    RANDOM = 1,
    SPACE_DIVISION = 2,
    DOUBLE_K_FIRST = 3


class Intergroup_Distance(Enum):
    SINGLE_LINK = 1,
    COMPLETE_LINK = 2


class K_means():
    def __init__(self, k=5, m=2,
                 init_strategy=Init_Strategy.RANDOM,
                 intergroup_distance=Intergroup_Distance.SINGLE_LINK,
                 max_iterations=50000,
                 threshold=0.001):
        self.k = k
        self.m = m
        self.init_strategy = init_strategy
        self.intergroup_distance = intergroup_distance
        self.max_iterations = max_iterations
        self.treshold = threshold
        self.locked = False

    def run(self, data):
        if self.locked == True:
            raise Exception("clustering already is running")

        self.locked = True
        self.validate_data(data)
        self.array = np.array(data)

        # read dimension of feature vectors
        self.n = len(data[0])

        # stores the cluster centroids
        self.centroids = self.init_centroids(data)

        # maps each data instance index to the centroid index
        self.cluster_mapping = np.zeros(len(data))

        abort = False
        cycle = 0
        while not abort:
            clusters_changed = False
            self.clear_instance_map()
            for instance_i in range(len(data)):
                closest_centroid_i = self.closest_centroid(instance_i)
                if self.cluster_mapping[instance_i] != closest_centroid_i:
                    self.cluster_mapping[instance_i] = closest_centroid_i
                    clusters_changed = True
                self.instance_map[closest_centroid_i].add(instance_i)
            for cluster_i in self.instance_map:
                new_centroid = self.recalc_centroid(cluster_i)
                self.centroids[cluster_i] = new_centroid
                print(f"New centroid for {cluster_i}: {new_centroid}")
            if (not clusters_changed) or (cycle >= self.max_iterations):  # TODO implement treshold
                abort = True
            cycle = cycle + 1
            print(cycle, clusters_changed, self.max_iterations, self.cluster_mapping)

        self.locked = False
        return self.centroids

    def clear_instance_map(self):
        self.instance_map = dict()
        for i in range(self.k):
            self.instance_map[i] = set()

    def recalc_centroid(self, cluster_i):
        total = np.zeros(self.n)
        for instance_i in self.instance_map[cluster_i]:
            total = total + self.array[instance_i]
        return total/len(self.instance_map[cluster_i])

    def init_centroids(self, data):
        centroids = []
        if self.init_strategy == Init_Strategy.RANDOM:
            # TODO this has potentially infinite runtime, but takes less space than making a in-memory copy of the data
            while len(centroids) != self.k:
                index = random.randint(0, len(data)-1)
                random_instance = data[index]
                if random_instance not in centroids:
                    centroids.append(random_instance)
        else:
            raise Exception(f"{self.init_strategy} not supported yet")
        return centroids

    def validate_data(self, data):
        if len(data) < self.k:
            raise Exception(
                f"Cannot group {len(data)} data instances into {self.k} clusters!")

    def closest_centroid(self, instance_i):
        min_centroid_i = 0
        min_distance = self.distance(instance_i, centroid_i=0)
        for centroid_i in range(1, self.k):
            distance = self.distance(instance_i, centroid_i)
            if distance < min_distance:
                min_centroid_i = centroid_i
                min_distance = distance
        return min_centroid_i

    def distance(self, instance_i, centroid_i):
        total = 0
        for feature_i in range(self.n):
            base = abs(self.array[instance_i][feature_i] -
                       self.centroids[centroid_i][feature_i])
            total = total + math.pow(base, self.m)
        return math.pow(total, 1/self.m)  # TODO float arithemtic
