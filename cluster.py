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
    
    # Since we're pre-allocating the np array below, we need to know how
    # many features each layer output has
    num_features = 0
    for layer, weight in layers:
        num_features += layer_lengths[layer]

    # N.B. This should be adjusted down if the full data set isn't being used
    # num_inputs = len(files) 
    num_inputs = 11

    input = np.zeros([num_inputs, num_features])
    counter = 0
    indices = []
    for idx, file in enumerate(files):
        file_input = []
        # TODO: This should know each output layer length so it can correctly index into the pre-allocated nparray
        for layer, weight in layers:
            path = "layer_data/" + file + "." + layer + ".json"
            data = json.load(open(path))
            file_input += data

        input[counter] = np.array(file_input)
        indices.append(file)

        # Short circuit early bc not enough memory
        if counter == 10:
            # N.B. When passed a matrix of all values, this is really slow
            # TODO: PCA on sampling of data
            # pca = PCA()
            # res = pca.fit_transform(np.array(input))
            return input, indices
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
