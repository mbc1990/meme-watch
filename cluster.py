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
    km = KMeans(n_clusters=8)

    # Run the algorithm
    km.fit(input)
    return km.labels_


def prepare_data():
    files = os.listdir("layer_data_conv1/")
    input = []
    counter = 0
    indices = []
    for idx, file in enumerate(files):
        data = json.load(open("layer_data_conv1/" + file))

        # TODO: Dimensionality reduction? 
        arr = np.array(data)

        input.append(arr)
        indices.append(file.split(".")[0])
        if counter == 100:
            # N.B. When passed a matrix of all values, this is really slow
            # pca = PCA()
            # res = pca.fit_transform(np.array(input))
            return np.array(input), indices
        counter += 1
        print str(counter)

def hierarchical(input):
    """
    Does hierarchical agglomerative clustering
    """
    print "Clustering..."
    # TODO: Try complete/maximum linkage
    # ac = AgglomerativeClustering(n_clusters=8, affinity=cos_sim_affinity, linkage="average")
    ac = AgglomerativeClustering(n_clusters=8, affinity="cosine", linkage="average")
    ac.fit(input)
    return ac.labels_


def main():
    data, indices = prepare_data()
    classes = hierarchical(data)

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
