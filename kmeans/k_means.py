from enum import Enum
import random
import numpy as np
import math
import pickle
import pprint


class Init_Strategy(Enum):
    RANDOM = 1,
    SPACE_DIVISION = 2,
    DOUBLE_K_FIRST = 3


class K_means():
    def __init__(self, k=5, m=2,
                 init_strategy=Init_Strategy.RANDOM,
                 max_iterations=50000,
                 threshold=0.001,
                 verbose=True):
        self.k = k
        self.m = m
        self.init_strategy = init_strategy
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.locked = False
        self.verbose = verbose

        # declare other variables that will be filled after/while run() is called
        self.total_error = -1
        self.instances_by_cluster = dict()
        self.iterations_run = -1
        self.cluster_mapping = None

    def run(self, data, after_centroid_calculation=lambda k_means, cycle: None,
            after_cluster_membership=lambda k_means, cycle: None):
        if self.locked == True:
            raise Exception("clustering already is running")

        self.locked = True
        self.validate_data(data)
        self.instances = np.array(data)

        # read dimension of feature vectors
        self.n = len(data[0])

        # stores the cluster centroids
        self.centroids = self.init_centroids()

        # store the initial centroid configuration for analysis purposes
        self.initial_centroids = self.centroids.copy()

        # maps each data instance index to the centroid index
        self.cluster_mapping = np.zeros(len(data))

        abort = False
        cycle = 0
        while not abort:
            if self.verbose == True:
                print(f">> Start cycleStart cycle {cycle+1}/{self.max_iterations}")
            # The instance map is used to keep track which instances belong to a cluster.
            # That is needed later for calculating the centroids of the cluster.
            self.clear_instance_map()

            # This variable will store the aggregated distances between the instances and its centroids
            # thus, it is a measurement how "good" or "bad" the clustering is
            self.total_error = 0

            # determine cluster memberships
            clusters_changed = 0
            for instance_i in range(len(data)):
                closest_centroid_i, distance = self.closest_centroid(
                    instance_i)
                old_centroid_i = self.cluster_mapping[instance_i]
                # check if the centroid must be updated
                if old_centroid_i != closest_centroid_i:
                    self.cluster_mapping[instance_i] = closest_centroid_i
                    clusters_changed += 1
                self.instances_by_cluster[closest_centroid_i].add(instance_i)
                self.total_error += distance
            if self.verbose == True:
                print(
                    f"{clusters_changed} cluster memberships changed, total_error={self.total_error}")
            after_cluster_membership(self, cycle)

            # This will contain the maximum distance between an old and its update centroid (used for the treshold termination criterion)
            max_centroids_change = 0

            # calculate new centroids for each cluster
            changed_centroids = 0
            for cluster_i in self.instances_by_cluster:
                new_centroid = self.calc_centroid(cluster_i)
                old_centroid = self.centroids[cluster_i]
                if np.array_equal(new_centroid, old_centroid):
                    continue

                self.centroids[cluster_i] = new_centroid
                changed_centroids += 1
                # calculate the distance of the new and old centroid (only if necessary)
                if self.threshold > 0 and self.threshold > max_centroids_change:
                    centroids_distance = self.distance(
                        old_centroid, new_centroid)
                    max_centroids_change = max(
                        max_centroids_change, centroids_distance)

            if self.verbose:
                print(
                    f"Centroid calculation completed: changed_centoids={changed_centroids}, max. changed distance (might not be accurate)={max_centroids_change}")
            after_centroid_calculation(self, cycle)

            # check if one of the abort criterions is reached
            abort_cycle = (cycle >= (self.max_iterations-1))
            abort_no_changes = (changed_centroids ==
                                0 and clusters_changed == 0)
            abort_threshold = (self.threshold > max_centroids_change)
            abort = abort_cycle or abort_no_changes or abort_threshold

            cycle = cycle + 1
            if self.verbose == True:
                print(
                    f"Finished cycle {cycle}, abort={abort}, abort_cycle={abort_cycle}, abort_no_changes={abort_no_changes}, abort_threshold={abort_threshold}")

        self.iterations_run = cycle
        self.locked = False
        return cycle

    # clears the instance map
    def clear_instance_map(self):
        self.instances_by_cluster = dict()
        for i in range(self.k):
            self.instances_by_cluster[i] = set()

    # Calculates the centroid of an cluster by averaging all instances of that cluster
    def calc_centroid(self, cluster_i):
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
    # the result is a tuple (centroid_index, distance) which contains:
    # 1.the index of the corresponding centroid of the self.centroids array
    # 2.the distance to this centroid
    def closest_centroid(self, instance_i):
        min_centroid_i = 0
        min_distance = self.centroid_instance_distance(
            instance_i, centroid_i=0)
        for centroid_i in range(1, self.k):
            distance = self.centroid_instance_distance(instance_i, centroid_i)
            if distance < min_distance:
                min_centroid_i = centroid_i
                min_distance = distance
        return (min_centroid_i, min_distance)

    # calculates the Minkowski distance between an instance and a centroid
    # both arguments must be passed as an index of the corresponding list (self.instances, self.centroids)
    # the parameter m (or alpha) is read from self.m
    def centroid_instance_distance(self, instance_i, centroid_i):
        return self.distance(self.instances[instance_i], self.centroids[centroid_i])

    # calculates the Minowski distance between two points
    # the points must be represented by arrays/lists of self.n dimension
    def distance(self, pointA, pointB):
        total = 0
        for feature_i in range(self.n):
            base = abs(pointA[feature_i] - pointB[feature_i])
            total = total + math.pow(base, self.m)
        return math.pow(total, 1/self.m)  # TODO float arithemtic

    # returns the cluster membership of an instance after run() was executed
    def get_centroid(self, instance_i):
        centroid_i = self.cluster_mapping[instance_i]
        return self.centroids[centroid_i]

    def to_file(self, file_name):
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()

    def __str__(self):
        return f"""<K-Means object>
Initialized with: k={self.k}, m={self.m}, init_strategy={self.init_strategy}
iterations run: {self.iterations_run}, total_error={self.total_error}"""


def from_file(file):
    return pickle.load(file)


class K_means_multiple_times():
    def __init__(self, k=5, m=2,
                 init_strategy=Init_Strategy.RANDOM,
                 max_iterations=50000,
                 threshold=0.001,
                 verbose=False):
        self.k = k
        self.m = m
        self.init_strategy = init_strategy
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.verbose = verbose

    def run(self, n_iterations, data,
            between_iter_fn=lambda iteration, k_means: None,
            after_centroid_calculation=lambda k_means, cycle: None,
            after_cluster_membership=lambda k_means, cycle: None):
        """
        Runs the K_means algorithm n times and returns the best-performing k_means instance
        (the one with the lowest total distance error)
        """
        best_k_means = None
        for iteration in range(0, n_iterations):
            if self.verbose == True:
                print(
                    f"### Start k-means iteration {iteration} / {n_iterations} ###")
            k_means = K_means(k=self.k,
                              init_strategy=self.init_strategy,
                              max_iterations=self.max_iterations,
                              m=self.m, threshold=self.threshold, verbose=self.verbose)
            k_means.run(data, after_centroid_calculation=after_centroid_calculation,
                        after_cluster_membership=after_cluster_membership)
            if best_k_means == None:
                best_k_means = k_means

            isBest = k_means.total_error < best_k_means.total_error
            if isBest:
                best_k_means = k_means

            between_iter_fn(iteration, k_means)

            if self.verbose == True:
                print("Finished k-means iteration with result:")
                print(k_means)
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("")

        return best_k_means
