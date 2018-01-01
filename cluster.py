import os
import json
from sets import Set
import numpy as np
from shutil import copyfile
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tesserocr import PyTessBaseAPI

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
    km = KMeans(n_clusters=10)

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

    num_to_cluster = len(files)
    
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
            input = pca.transform(input)
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


def multi_cluster():
    """
    Run clustering separately on each layer output, then... what?
    """
    pass


def extract_text():
    """
    Runs OCR on input images and saves the results
    """

    files = os.listdir("saved_memes/")
    with PyTessBaseAPI() as api:
        for idx, file in enumerate(files):
            if not os.path.exists("text_data/" + file + ".text.json"):
                print idx
                api.SetImageFile("saved_memes/" + file)
                text = api.GetUTF8Text()
                output = {
                        "text": text
                }
                with open("text_data/" + file + ".text.json", 'w') as outfile:
                    json.dump(text, outfile)


def text_cluster(clusters):
    """
    Cluster based on extracted text from images
    """
    files = os.listdir("saved_memes/")
    # files = files[:26]

    num_to_cluster = len(files)

    # num_to_cluster = 26
    
    seen_tokens = Set()

    # Get unordered word vector for each image
    extracted_texts = []
    with PyTessBaseAPI() as api:
        for idx, file in enumerate(files):
            print idx

            # Get text from disk, or run OCR if we dont have it
            text = ""
            if os.path.exists("text_data/" + file + ".text.json"):
                text = json.load(open("text_data/" + file + ".text.json"))
            else:
                api.SetImageFile("saved_memes/" + file)
                text = api.GetUTF8Text()
                # TODO: save to disk if we do this

            # Clean up text
            text = text.lower()
            text = text.replace("\n", "")

            words = text.split(" ")
            # Strip empty tokens
            words = [word for word in words if word != ""]

            # TODO: Remove stopwords
            # TODO: TF-IDF

            bag_of_words = {} 

            # Throw out images we couldn't extract text from
            if len(words) < 5:
                num_to_cluster -= 1
                continue

            for token in words:
                seen_tokens.add(token)
                if token not in bag_of_words:
                    bag_of_words[token] = 0
                bag_of_words[token] += 1
            extracted_texts.append({"filename": file, "bag_of_words": bag_of_words})

    print "Clustering " + str(len(extracted_texts)) + " images"
    
    # word -> index in input vector
    word_index = {}
    counter = 0
    for token in seen_tokens:
        word_index[token] = counter
        counter += 1

    indices = {}

    input = np.zeros([num_to_cluster, len(seen_tokens)])
    counter = 0
    for bow in extracted_texts:
        indices[bow["filename"]] = counter
        for key in bow["bag_of_words"].keys():
            input[counter][word_index[key]] = bow["bag_of_words"][key]

        # Normalize word vector 
        norm = np.linalg.norm(input[counter], ord=1)            
        print "Norm: " + str(norm)
        input[counter] /= norm

        counter += 1

    classes = hierarchical(input, clusters)
    return classes, indices



def main():
    '''
    data, indices = prepare_data([("conv1", 0.5),
                                  ("conv2", 0.75),
                                  ("fc7", 0.85),
                                  ("fc8", 1.0)])
    classes = hierarchical(data, 40)
    '''

    classes, indices = text_cluster(25)

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
