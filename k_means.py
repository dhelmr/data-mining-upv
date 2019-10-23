from enum import Enum
import random
import numpy as np
import math

class Init_Strategy(Enum):
    RANDOM = 1,
    SPACE_DIVISION = 2,
    DOUBLE_K_FIRST = 3

class K_means():
    def __init__(self, k=5, m=2,
                 init_strategy=Init_Strategy.RANDOM,
                 max_iterations=50000,
                 threshold=0.001):
        self.k = k
        self.m = m
        self.init_strategy = init_strategy
        self.max_iterations = max_iterations
        self.treshold = threshold
        self.locked = False
        self.instances_by_cluster = dict()

        # initialize functions that are called at different steps of the algorithm with 
        # a dummy function that does no do anything
        # the functions can be used as "hooks" to debug or visualize the current state of the algorithm
        def pass_fn():
            pass
        self.after_centroid_calculation = pass_fn 
        self.after_cluster_membership = pass_fn

    def run(self, data):
        if self.locked == True:
            raise Exception("clustering already is running")

        self.locked = True
        self.validate_data(data)
        self.instances = np.array(data)

        # read dimension of feature vectors
        self.n = len(data[0])

        # stores the cluster centroids
        self.centroids = self.init_centroids()

        # maps each data instance index to the centroid index
        self.cluster_mapping = np.zeros(len(data))

        abort = False
        cycle = 0
        while not abort:
            # The instance map is used to keep track which instances belong to a cluster.
            # That is needed later for calculating the centroids of the cluster.
            self.clear_instance_map()
            
            # determine cluster memeberships
            clusters_changed = False
            for instance_i in range(len(data)):
                closest_centroid_i = self.closest_centroid(instance_i)
                if self.cluster_mapping[instance_i] != closest_centroid_i:
                    self.cluster_mapping[instance_i] = closest_centroid_i
                    clusters_changed = True
                self.instances_by_cluster[closest_centroid_i].add(instance_i)
            self.after_cluster_membership()

            # calculate new centroids for each cluster
            for cluster_i in self.instances_by_cluster:
                new_centroid = self.recalc_centroid(cluster_i)
                self.centroids[cluster_i] = new_centroid
            if (not clusters_changed) or (cycle >= self.max_iterations):  # TODO implement treshold
                abort = True
            cycle = cycle + 1
            self.after_centroid_calculation()

        self.locked = False
        return self.centroids

    # clears the instance map
    def clear_instance_map(self):
        self.instances_by_cluster = dict()
        for i in range(self.k):
            self.instances_by_cluster[i] = set()

    # Calculates the centroid of an cluster by averaging all instances of that cluster
    def recalc_centroid(self, cluster_i):
        total = np.zeros(self.n)
        for instance_i in self.instances_by_cluster[cluster_i]:
            total = total + self.instances[instance_i]
        return total/len(self.instances_by_cluster[cluster_i])

    # Initializes the centroids according to the initalization strategy (self.init_strategy)
    # See the Init_Strategy enum for possible values
    def init_centroids(self):
        centroids = []
        if self.init_strategy == Init_Strategy.RANDOM:
            # TODO this has potentially infinite runtime, but takes less space than making a in-memory copy of the data
            while len(centroids) != self.k:
                index = random.randint(0, len(self.instances)-1)
                random_instance = self.instances[index].tolist()
                if (random_instance not in centroids):
                    centroids.append(random_instance)
        else:
            raise Exception(f"{self.init_strategy} not supported yet")
        return centroids

    # checks if the data is suited for running a clustering algorithm
    def validate_data(self, data):
        if len(data) < self.k:
            raise Exception(
                f"Cannot group {len(data)} data instances into {self.k} clusters!")

    # determines the closest centroid for a data instance according to the current clusters
    # the result is the index of the corresponding centroid of the self.centroids array
    def closest_centroid(self, instance_i):
        min_centroid_i = 0
        min_distance = self.distance(instance_i, centroid_i=0)
        for centroid_i in range(1, self.k):
            distance = self.distance(instance_i, centroid_i)
            if distance < min_distance:
                min_centroid_i = centroid_i
                min_distance = distance
        return min_centroid_i

    # calculates the Minkowski distance between an instance and a centroid
    # both arguments must be passed as an index of the corresponding list (self.instances, self.centroids)
    # the parameter m (or alpha) is read from self.m
    def distance(self, instance_i, centroid_i):
        total = 0
        for feature_i in range(self.n):
            base = abs(self.instances[instance_i][feature_i] -
                       self.centroids[centroid_i][feature_i])
            total = total + math.pow(base, self.m)
        return math.pow(total, 1/self.m)  # TODO float arithemtic
