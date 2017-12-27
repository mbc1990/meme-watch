import os
import numpy as np
from shutil import copyfile
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity
import json

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
    files = os.listdir("saved_memes/")
    
    # N.B. THESE MUST BE UPDATED TO MATCH THE INPUT DATA
    num_features = 2449040
    num_inputs = 201 

    input = np.zeros([num_inputs, num_features])
    counter = 0
    indices = []
    for idx, file in enumerate(files):
        try:
            file_input = []
            # TODO: This should know each output layer length so it can correctly index into the pre-allocated nparray
            for layer in layers:
                path = "layer_data/" + file + "." + layer + ".json"
                data = json.load(open(path))
                file_input += data
        except:
            continue
        input[counter] = np.array(file_input)

        # input.append(np.array(file_input))
        indices.append(file)
        if counter == 200:
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

    # classes = kmeans(data)
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
