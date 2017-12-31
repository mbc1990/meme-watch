import os
import numpy as np
from shutil import copyfile
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
import json

# Number of entries in each feature vector
layer_lengths = {
    "conv1": 2904000,
    "conv2": 1866240,
    "conv3": 648960,
    "conv4": 648960,
    "conv5": 432640,
    "fc6": 40960,
    "fc7": 40960,
    "fc8": 10000,
    "norm1": 699840,
    "norm2": 432640,
    "pool1": 699840,
    "pool2": 432640,
    "pool5": 92160
}

def kmeans(input):
    """
    Args:
        input nparray data to cluster
    """
    print "Clustering..."
    km = KMeans(n_clusters=4)

    # Run the algorithm
    km.fit(input)
    return km.labels_


def prepare_data(layers):
    """
    Loads data from disk, weights it, and puts it in a numpy array 
    
    Args:
        layers list of tuples ("layer_name", <layer weight float>) 

    Returns: nparray, indices (a map of index in nparray -> filename at that index)
    """
    files = os.listdir("saved_memes/")

    # num_inputs = len(files)
    num_to_cluster = 10 
    
    # Since we're pre-allocating the np array below, we need to know how
    # many features each layer output has
    num_features = 0
    for layer, weight in layers:
        num_features += layer_lengths[layer]
    
    # Every pca_sample_rate vectors, add one to the matrix to be used for PCA
    pca_sample_rate = 2 
    pca_input = np.zeros([num_to_cluster/pca_sample_rate, num_features])
    pca_counter = 0

    input = np.zeros([num_to_cluster, num_features])
    counter = 0
    indices = []
    for idx, file in enumerate(files):
        # Short circuit early bc not enough memory
        if counter == num_to_cluster:
            # Fit on a sampling of input data
            pca = PCA()
            pca.fit(pca_input)

            # But transform the whole input matrix
            # input = pca.transform(input)
            return input, indices

        file_input = []
        # TODO: This should know each output layer length so it can correctly index into the pre-allocated nparray
        for layer, weight in layers:
            path = "layer_data/" + file + "." + layer + ".json"
            data = json.load(open(path))
            file_input += data
        input[counter] = np.array(file_input)
        indices.append(file)
        
        # If we're going to use the vector in PCA, add it to that array
        if idx % pca_sample_rate == 0:
            pca_input[pca_counter] = np.array(file_input)
            pca_counter += 1

        counter += 1
        print str(counter)


def hierarchical(input, num_clusters):
    """
    Does hierarchical agglomerative clustering
    """
    print "Clustering..."
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity="cosine", linkage="average")
    ac.fit(input)
    return ac.labels_


def main():
    data, indices = prepare_data([("fc6", 0.75),
                                  ("fc7", 0.85),
                                  ("fc8", 1.0), 
                                  ("norm1", 0.25), 
                                  ("norm2", 0.35), 
                                  ("pool1", 0.25), 
                                  ("pool2", 0.35), 
                                  ("pool5", 0.85)])

    classes = hierarchical(data, 8)

    clusters = {}
    for idx, fname in enumerate(indices):
        cluster = classes[idx]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(fname)
    
    # N.B. You must delete the contents of cluster_output before running
    os.makedirs("cluster_output")
    for key in clusters.keys():
        os.makedirs("cluster_output/"+str(key))

        # Copy the image files to the cluster directory
        for fname in clusters[key]:
            copyfile("saved_memes/"+fname, "cluster_output/"+str(key)+"/"+fname)

    print clusters

if __name__ == "__main__":
    main()
