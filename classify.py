from imageio import imread
import tensorflow as tf
import time
import numpy as np
import cv2
from scipy import misc
import random
from helper import mask_hood, map_labels
from sklearn.metrics import f1_score

color = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [0, 255, 0],
}

def read_image(image_file, im_size):
    im_height, im_width, channels = im_size
    image = imread(image_file)

    vertical_start = 169
    vertical_end = vertical_start + im_height
    image = image[vertical_start:vertical_end, :im_width, :]
    return image

def restore_image(image_vec, im_size):
    image_vec = [color[elem] for elem in image_vec]
    image_vec = np.array(image_vec)
    image = image_vec.reshape(im_size)
    return image

# Settings
frozen_graph = './inference/frozen_fcn_vgg16_80e-adam-batchsize.pb'
vertical_start = 169
im_height = 352
im_height_orig = 600
lower_height = im_height_orig - im_height - vertical_start
im_width = 800
batch_size = 8
im_size = (352, 800, 3)
num_classes = 3

pixels = im_height * im_width
vertical_end = vertical_start + im_height
mask = mask_hood()

# Open inference graph file
with tf.gfile.GFile(frozen_graph, "rb") as file:
    graph_def = tf.GraphDef()    
    graph_def.ParseFromString(file.read())

G = tf.Graph()

with tf.Session(graph=G) as sess:

    logits, = tf.import_graph_def(graph_def, return_elements=['logits:0'])
    input_image = G.get_tensor_by_name("import/nn_input:0")
    keep_prob = G.get_tensor_by_name("import/keep_prob:0")

    for i in range(5):
        randnr = random.randint(0, 1000)
        # Image is expanded to 4 dims - 1st dim batch size (=1)
        image_file = './Train/CameraRGB/' + str(randnr) + '.png'
        label_file = './Train/CameraSeg/' + str(randnr) + '.png'
        image = read_image(image_file, im_size)
        image_4d = np.expand_dims(image, axis=0)
        image_4d = (image_4d / 255.0) - 0.5
        label = imread(label_file)
        # Labels are stored in the R channel of RGB (channel 0)
        label = label[:,:,0]
        # Transforms hood pixels. Before: vehicle (class 2), after: class 0. Mask is 0 for all hood pixels, 1 otherwise
        label = np.multiply(label, mask)
        # Crops image below hood and above horizon where all pixels are of class 0 and therefore not of interest. Speeds up training.
        label = label[vertical_start:vertical_end, :im_width]

        label_shape = label.shape
        label = (np.array(label)).reshape(-1)

        label = np.eye(13)[label] # Convert label vector to one-hot
        label = map_labels(label) # Convert labels to 0,1 or 2

        pred = sess.run(logits, feed_dict={input_image:image_4d, keep_prob:1.0})
        y_pred = np.argmax(pred, axis = 1)
        label = np.argmax(label, axis = 1)

        binary_car_result = np.where(y_pred == 2, 1, 0).astype('uint8')
        binary_road_result = np.where(y_pred == 1, 1, 0).astype('uint8')

        car_labels = np.where(label == 2, 1, 0).astype('uint8')
        road_labels = np.where(y_pred == 1, 1, 0).astype('uint8')

        label_image = restore_image(label, im_size)
        pred_image = restore_image(y_pred, im_size)

        car_f1 = f1_score(car_labels, binary_car_result)
        road_f1 = f1_score(road_labels, binary_road_result)

        print("Car F1: {0}, Road F1: {1}".format(car_f1, road_f1))

        imlbl = np.concatenate((label_image, pred_image), axis=0)
        misc.imshow(imlbl)

