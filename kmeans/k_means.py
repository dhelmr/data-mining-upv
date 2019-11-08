from enum import IntEnum
import random
import numpy as np
import math
import pickle
import pprint
import copy
import numba
import random

import warnings
warnings.filterwarnings('ignore')


class Init_Strategy(IntEnum):
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
        self.instances_by_cluster = dict()
        self.iterations_run = -1
        self.cluster_mapping = None

        """
        Stores the distance to the closest other centroid
        """
        self.closest_centroid_distances = np.zeros(k)

    def run(self, data, after_centroid_calculation=lambda k_means, cycle: None,
            after_cluster_membership=lambda k_means, cycle: None,
            centroid_initialization=None):
        if self.locked == True:
            raise Exception("clustering already is running")

        self.locked = True
        self._validate_data(data)
        self._set_instances(data)

        # initiaze data structures that are used for determining possible skips later
        self.upper_bound_distance_to_centroid = np.zeros(
            len(self.instances), dtype=float)
        self.lower_bound_second_closest_centroid = np.zeros(
            len(self.instances), dtype=float)
        self.last_centroid_changes = np.zeros(self.k, dtype=float)

        # initialize centroids
        if centroid_initialization != None:
            self.centroids = copy.deepcopy(centroid_initialization)
        else:
            self.centroids = self._init_centroids()

        # store the initial centroid configuration for analysis purposes
        self.initial_centroids = copy.deepcopy(self.centroids)

        # maps each data instance index to the centroid index
        self.cluster_mapping = np.zeros(len(data), dtype=int)

        if self.verbose == True:
            print(f"Start clustering with k={self.k}, m={self.m}")

        self._clear_instance_map()

        for instance_i in range(len(self.instances)):
            centroid_i = self._set_cluster_membership(instance_i)
            self.instances_by_cluster[centroid_i].add(instance_i)

        abort = False
        cycle = 0
        while not abort:
            self._update_closest_centroid_distances()
            
            clusters_changed = self._determine_cluster_memberships()
            after_cluster_membership(self, cycle)

            changed_centroids, max_centroids_change = self._update_centroids()
            after_centroid_calculation(self, cycle)

            self._update_bounds()

            # check if one of the abort criterions is reached
            abort_cycle = (cycle >= (self.max_iterations-1))
            abort_no_changes = (changed_centroids ==
                                0 and clusters_changed == 0)
            abort_threshold = (self.threshold > max_centroids_change)
            abort = abort_cycle or abort_no_changes or abort_threshold

            cycle = cycle + 1
            if self.verbose == True:
                print(
                    f"Finished cycle {cycle}/{self.max_iterations} | {clusters_changed} cluster memberships changed, changed_centoids={changed_centroids}, max. changed distance>={max_centroids_change}, abort={[abort_cycle, abort_no_changes, abort_threshold]}")

        self.iterations_run = cycle
        self.locked = False
        return cycle

    # clears the instance map
    def _clear_instance_map(self):
        self.instances_by_cluster = dict()
        for i in range(self.k):
            self.instances_by_cluster[i] = set()

    # Calculates the centroid of an cluster by averaging all instances of that cluster
    def _calc_centroid(self, cluster_i):
        centroid = jit_calc_centroid(self.n, self.instances_by_cluster[cluster_i], self.instances)
        if centroid is None:
            return self.centroids[cluster_i]
        return centroid
            
    # Initializes the centroids according to the initialization strategy (self.init_strategy)
    # See the Init_Strategy enum for possible values
    def _init_centroids(self):
        centroids = []
        if self.init_strategy == Init_Strategy.RANDOM:
            # TODO this has potentially infinite runtime, but takes less space than making a in-memory copy of the data
            while len(centroids) != self.k:
                index = random.randint(0, len(self.instances)-1)
                random_instance = self.instances[index].tolist()
                if (random_instance not in centroids):
                    centroids.append(random_instance)
        elif self.init_strategy == Init_Strategy.DOUBLE_K_FIRST:
            initializer = DoubleKInitialization(self)
            initializer.run()
            return initializer.get_centroids()
        else:
            raise Exception(f"{self.init_strategy} not supported yet")
        return centroids

    # checks if the data is suited for running a clustering algorithm
    def _validate_data(self, data):
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
        return minkowski(self.m, pointA, pointB, self.n)

    def _update_closest_centroid_distances(self):
        for centroid_i in range(len(self.centroids)):
            min_distance = None
            for centroid_j in range(len(self.centroids)):
                if centroid_i != centroid_j:
                    distance = self.distance(
                        self.centroids[centroid_i], self.centroids[centroid_j])
                    if min_distance == None or distance < min_distance:
                        min_distance = distance
            if self.closest_centroid_distances[centroid_i] != min_distance:     
                self.closest_centroid_distances[centroid_i] = min_distance

    def _determine_cluster_memberships(self):
        # determine cluster memberships
        clusters_changed = 0
        for instance_i in range(len(self.instances)):
            old_centroid_i = self.cluster_mapping[instance_i]
            s2 = self.closest_centroid_distances[old_centroid_i]/float(2)
            m = max(s2,
                    self.lower_bound_second_closest_centroid[instance_i])
            upper_bound_distance = self.upper_bound_distance_to_centroid[instance_i]
         
           # print(self.upper_bound_distance_to_centroid, self.lower_bound_second_closest_centroid, self.closest_centroid_distances)
            if upper_bound_distance > m:
                self.upper_bound_distance_to_centroid[instance_i] = self.centroid_instance_distance(
                    instance_i, old_centroid_i)
                if True or self.upper_bound_distance_to_centroid[instance_i] > m:
                    closest_centroid_i = self._set_cluster_membership(
                        instance_i)
                    # check if the centroid must be updated
                    if old_centroid_i != closest_centroid_i:
                        self.instances_by_cluster[closest_centroid_i].add(
                            instance_i)
                        self.instances_by_cluster[old_centroid_i].remove(
                            instance_i)
                        clusters_changed += 1
        return clusters_changed

    def _set_cluster_membership(self, instance_i):
        closest_centroid_i, distance = self.closest_centroid(instance_i)
        self.cluster_mapping[instance_i] = closest_centroid_i
        self._update_lower_bounds_to_second_closest_centroid(
            instance_i, closest_centroid_i)
        self.upper_bound_distance_to_centroid[instance_i] = distance
        return closest_centroid_i

    def _update_lower_bounds_to_second_closest_centroid(self, instance_i, closest_cluster_i):
        min_distance = None
        for cluster_j in range(len(self.centroids)):
            if cluster_j == closest_cluster_i:
                continue
            distance = self.centroid_instance_distance(instance_i, cluster_j)
            if min_distance == None or distance < min_distance:
                min_distance = distance
        self.lower_bound_second_closest_centroid[instance_i] = min_distance

    def _update_centroids(self):
        # This will contain the maximum distance between an old and its update centroid (used for the treshold termination criterion)
        max_centroids_change = 0

        # calculate new centroids for each cluster
        changed_centroids = 0
        for cluster_i in self.instances_by_cluster:
            new_centroid = self._calc_centroid(cluster_i)
            old_centroid = self.centroids[cluster_i]
            if np.array_equal(new_centroid, old_centroid):
                continue
            self.centroids[cluster_i] = new_centroid
            changed_centroids += 1
            centroids_distance = self.distance(
                old_centroid, new_centroid)
            max_centroids_change = max(
                max_centroids_change, centroids_distance)
            self.last_centroid_changes[cluster_i] = centroids_distance

        return (changed_centroids, max_centroids_change)

    def _update_bounds(self):
        r = None
        r2 = None
        for cluster_i in range(len(self.last_centroid_changes)):
            last_change = self.last_centroid_changes[cluster_i]
            if r == None or last_change > self.last_centroid_changes[r]:
                r2 = r
                r = cluster_i
            elif r2 == None or last_change > self.last_centroid_changes[r2]:
                r2 = cluster_i

        for instance_i in range(len(self.instances)):
            cluster_i = self.cluster_mapping[instance_i]
            upper_bound = self.upper_bound_distance_to_centroid[instance_i]
            lower_bound = self.lower_bound_second_closest_centroid[instance_i]
            self.upper_bound_distance_to_centroid[instance_i] = upper_bound + \
                self.last_centroid_changes[cluster_i]
            if r == cluster_i:
                self.lower_bound_second_closest_centroid[instance_i] = lower_bound - \
                    self.last_centroid_changes[r2]
            else:
                self.lower_bound_second_closest_centroid[instance_i] = lower_bound - \
                    self.last_centroid_changes[r]

    def get_centroid(self, instance_i):
        """
        returns the corresponding centroid coordinates of an instance after run() was executed
        """
        centroid_i = self.cluster_mapping[instance_i]
        return self.centroids[centroid_i]

    def to_file(self, file_name):
        """
        Stores the kmeans object as a pickle file that can later be read with from_file()
        """
        file = open(file_name, 'wb')
        pickle.dump(self, file)
        file.close()

    def __str__(self):
        return f"""<K-Means object>
Initialized with: k={self.k}, m={self.m}, threshold={self.threshold}, init_strategy={self.init_strategy}
iterations run: {self.iterations_run}"""

    def copy(self, new_k):
        return K_means(k=new_k, m=self.m, init_strategy=self.init_strategy,
                       max_iterations=self.max_iterations, threshold=self.threshold, verbose=self.verbose)

    def calc_SSE(self):
        """
        Calculates the squared summed squared error of all instance-centroid distances
        """
        total = 0
        for instance_i in range(len(self.instances)):
            centroid_i = self.cluster_mapping[instance_i]
            distance = self.centroid_instance_distance(instance_i=instance_i, centroid_i=centroid_i)
            total+=pow(distance, 2)
        return total

    def result_to_file(self, file):
        """
        Stores the result of the clustering (after run()), without datastructures only used for the algorithm 
        and without the actual instances
        """
        copy = self.copy(new_k= self.k)
        copy.cluster_mapping = self.cluster_mapping
        copy.centroids = self.centroids
        copy.instances_by_cluster = self.instances_by_cluster
        copy.first_instances = self.instances[0:20]
        copy.initial_centroids = self.initial_centroids
        pickle.dump(copy, open(file, "wb"))

    def _set_instances(self, data):
        self.instances = np.array(data)
        # read dimension of feature vectors
        self.n = len(data[0])


def from_file(file, data=[]):
    kmeans = pickle.load(open(file, "rb"))
    if data != []:
        kmeans._set_instances(data)
    return kmeans

@numba.jit(nopython=True)
def minkowski(m, pointA, pointB, n_features):
    total = 0
    for feature_i in range(n_features):
        base = abs(pointA[feature_i] - pointB[feature_i])
        total = total + math.pow(base, m)
    return math.pow(total, 1/m)

@numba.autojit(nopython=True)
def jit_calc_centroid(n, cluster_instances_i, instances):
    total = np.zeros(n)
    for instance_i in cluster_instances_i:
        total = total + instances[instance_i]
    n_instances = len(cluster_instances_i)
    if n_instances == 0:
        return None
    return total/n_instances

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
        best_k_means, best_sse = None, None
        for iteration in range(0, n_iterations):
            if self.verbose == True:
                print(
                    f"### Start k-means iteration {iteration+1} / {n_iterations} ###")
            k_means = K_means(k=self.k,
                              init_strategy=self.init_strategy,
                              max_iterations=self.max_iterations,
                              m=self.m, threshold=self.threshold, verbose=self.verbose)
            k_means.run(data, after_centroid_calculation=after_centroid_calculation,
                        after_cluster_membership=after_cluster_membership)

            sse = k_means.calc_SSE()
            if best_sse == None or sse < best_sse:
                best_k_means = k_means
                best_sse = sse

            between_iter_fn(iteration, k_means)

            if self.verbose == True:
                print("Finished k-means iteration with result:")
                print(k_means)
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                print("")

        return best_k_means


class DoubleKInitialization():
    def __init__(self, orig_k_means):
        self.orig_k_means = orig_k_means
        new_k = min(2*orig_k_means.k, len(orig_k_means.instances))
        k_means = orig_k_means.copy(new_k)
        k_means.init_strategy = Init_Strategy.RANDOM
        self.k_means = k_means

    def run(self):
        if (self.orig_k_means.verbose == True):
            print("Start k-means run with 2k for initialization")
        self.k_means.run(self.orig_k_means.instances)

    def get_centroids(self):
        sorted_centroid_indexes = list(range(0, self.k_means.k))
        sorted_centroid_indexes.sort(
            key=lambda cluster_i: self.intra_cluster_distance(cluster_i))
        return list(map(lambda cluster_i: self.k_means.centroids[cluster_i], sorted_centroid_indexes[:self.orig_k_means.k]))

    def intra_cluster_distance(self, cluster_i):
        """
        Calculates the summed squared distance of all instances of a cluster and divided by the number of instances
        """
        total = float(0)
        instances = self.k_means.instances_by_cluster[cluster_i]
        if len(instances) == 0:
            return float('nan')
        for instance_i in instances:
            distance = self.k_means.centroid_instance_distance(
                instance_i, cluster_i)
            total += pow(distance, 2)
        return total/len(instances)