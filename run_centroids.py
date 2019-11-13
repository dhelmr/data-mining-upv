"""
Applying clustering on unseen tweets and add corresponding cluster (nearest centroid)

"""


import glob
import pickle
import argparse
from kmeans.k_means import from_file

def get_original_text(test_index):
    #read from csv
    return 1

def main(filename, k_means_path):
    # TODO glob
    kmeans = 'kmeans/models/k=2_m=2.0_init=1_1573230238.2595701.result'
    # tweets (pickle)
    with open(filename, 'rb') as f:
        cleaned_data = pickle.load(f)

    

    vecs = cleaned_data["vectors"]
    
    kmeans = from_file(k_means_path)
    kmeans.n = len(vecs[0])
    testing_instances = vecs[-100000:]
    print(testing_instances)

    for i in range(10):
        instance = testing_instances[377444+i]
        cluster, distance = kmeans.predict(instance)
        instance_indexes = list(kmeans.instances_by_cluster[cluster])
        print("##", cluster, distance)
        for index in instance_indexes[:10]:
            text = cleaned_data["text"][index]
            print(text)
       

    # iterating k_means models from path
    # for filename in glob.glob(f'{k_means_path}/*.result'):

    # compare with and calculate metrics

    # save results to file / document performance

    print("### ENDED ####")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-f", "--file", default="resources/tweets_test_vecs600.vec", help="path to vectors file")
    parser.add_argument("-k", "--k_means", default="kmeans/", help="path to k_means objects")
    args = parser.parse_args()

    main(args.file, args.k_means)
