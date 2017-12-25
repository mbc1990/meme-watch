import os
import numpy as np
from shutil import copyfile
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json

def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    return cosine_similarity(X,Y)

def kmeans(input):
    """
    Args:
        input nparray data to cluster
    """
    print "Clustering..."
    km = KMeans(n_clusters=8)
    # Run the algorithm
    km.fit(input)
    return km.labels_


def prepare_data():
    files = os.listdir("layer_data/")
    input = []
    counter = 0
    indices = []
    for idx, file in enumerate(files):
        data = json.load(open("layer_data/" + file))
        input.append(np.array(data))
        indices.append(file.split(".")[0])
        # Initially test with ~1/100th of the data set
        if counter == 10:
            return np.array(input), indices
        counter += 1
        print str(counter)


def main():
    data, indices = prepare_data()
    classes = kmeans(data)
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
