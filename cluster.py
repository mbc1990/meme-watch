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
    input = []
    counter = 0
    indices = []
    for file in files:
        try:
            file_input = []
            for layer in layers:
                path = "layer_data/" + file + "." + layer + ".json"
                data = json.load(open(path))
                file_input += data
        except:
            continue

        input.append(np.array(file_input))
        indices.append(file)
        if counter == 50:
            # N.B. When passed a matrix of all values, this is really slow
            # pca = PCA()
            # res = pca.fit_transform(np.array(input))
            return np.array(input), indices
        counter += 1
        print str(counter)


def hierarchical(input, num_clusters):
    """
    Does hierarchical agglomerative clustering
    """
    print "Clustering..."
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity="cosine", linkage="complete")
    ac.fit(input)
    return ac.labels_


def main():
    data, indices = prepare_data(["conv1", "conv2", "conv3", "conv4", "conv5","fc6", "fc7", "fc8", "norm1", "norm2", "pool1", "pool2", "pool5"])
    classes = hierarchical(data, 25)

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
