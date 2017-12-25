from flask import Flask
from flask import request
import caffe
import numpy as np
import sys
import json
from scipy.spatial import distance


app = Flask(__name__)

@app.route('/store_layer_outputs')
def store_layer_outputs():
    path_to_image = request.args.get("filename")
    image_id = request.args.get("image_id")
    layer_data_dir = "layer_data/"
    caffe_root = "/home/malcolm/Projects/caffe/"
    sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_cpu()
    model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout) TODO: Whats this

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)
    
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    image = caffe.io.load_image(path_to_image)
    transformed_image = transformer.preprocess('data', image)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    layers = net.blobs.keys()
    for layer in layers:
        data = net.blobs.get(layer).data.flatten()
        fname = layer_data_dir + str(image_id) + "." + layer + ".json"
        with open(fname, 'w') as outfile:
            json.dump(data.tolist(), outfile)

if __name__ == '__main__':
    app.run(debug=True, port=4446)
