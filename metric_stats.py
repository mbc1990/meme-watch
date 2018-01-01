import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import json

class Image:
    def __init__(self, fname):
        self.fname = fname

    def __eq__(self, obj):
        return self.fname == obj.fname

    def get_data_for_layer(self, layer):
        data = json.load(open("layer_data/" + self.fname + "."+layer+".json"))
        return np.array(data)


def avg_distance_per_image():
    """
    For each image, prints the average cosine_similarity for each layer, for each other file
    """
    files = os.listdir("saved_memes")
    images = []
    for f in files:
        img = Image(f)
        images.append(img)
    
    # TODO: Testing, remove
    images = images[:10]
    
    layers_to_compare = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "fc6",
        "fc7",
        "fc8",
        "norm1",
        "norm2",
        "pool1",
        "pool2",
        "pool5"
    ]

    for img_a in images:
        print img_a.fname
        print "===================="
        for layer in layers_to_compare:
            total = 0.0
            for img_b in images:
                # Don't compare to yourself
                if img_a == img_b:
                    continue
                a = img_a.get_data_for_layer(layer)
                b = img_b.get_data_for_layer(layer)
                dist = cosine(a, b)
                total += dist
            avg = total / float(len(images) - 1)
            print "    " + layer + " :" + str(avg)

def text_metrics():
    """
    Look into what kind of text we could extract
    """
    files = os.listdir("text_data/")

    # Total length in words of all extracted text (for average)
    total_len = 0.0

    # Documents with any extracted text
    with_text = 0

    for file in files:
        data = json.load(open("text_data/" + file))
        # Strip newline characters
        data = data.replace("\n", "")

        words = data.split(" ")
        # Strip empty tokens
        words = [word for word in words if word != ""]
        if len(words) > 10:
            total_len += len(words)
            print words
            print "---"
            with_text += 1

    avg = total_len / len(files)
    print "Average length: " + str(avg) + " words"
    print str(float(with_text)/len(files) * 100) + "% of files with text ("+str(with_text) + "/" + str(len(files))+")"


def most_similar(img, top_n=10):
    """
    Returns the n most similar images to the input

    Args:
        img Image to find similar images for
        top_n int number of results to return
    """
    pass


def get_layer_sizes():
    """
    Prints the lengths of each layer output
    """
    test_file = "1f2f6d647a7ddb8a4629448ae00e655f"
    layers = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "fc6",
        "fc7",
        "fc8",
        "norm1",
        "norm2",
        "pool1",
        "pool2",
        "pool5"
    ]
    img = Image(test_file)
    for layer in layers:
        data = img.get_data_for_layer(layer)
        print layer + ": " + str(len(data))


def main():
    # avg_distance_per_image()    
    # get_layer_sizes()
    text_metrics()
    

if __name__ == "__main__":
    main()
