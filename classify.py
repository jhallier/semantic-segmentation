from imageio import imread
import tensorflow as tf
import time
import numpy as np
import cv2
from scipy import misc
import random

color = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [0, 255, 0],
}

def read_image(image_file, im_size):
    im_height, im_width, channels = im_size
    image = imread(image_file)
    image = image[:im_height, :im_width, :]
    return image

def restore_image(image_vec, im_size):
    image_vec = [color[elem] for elem in image_vec]
    image_vec = np.array(image_vec)
    image = image_vec.reshape(im_size)
    return image
    
model_path = './model/model-10'
graph_path = './model/model-10.meta'

im_size = (576, 800, 3)
#images = tf.placeholder(tf.float32, shape=(1, im_size[0], im_size[1], 3))

start_time = time.time()
# Runs inference on one image on the loaded graph
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(graph_path)
    saver.restore(sess, model_path)

    graph = tf.get_default_graph()

    logits = graph.get_tensor_by_name("logits:0")
    input_image = graph.get_tensor_by_name("nn_input:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    for i in range(5):
        randnr = random.randint(0, 1000)
        # Image is expanded to 4 dims - 1st dim batch size (=1)
        image_file = './Train/CameraRGB/' + str(randnr) + '.png'
        image = read_image(image_file, im_size)
        image_4d = np.expand_dims(image, axis=0)

        pred = sess.run(logits, feed_dict={input_image:image_4d, keep_prob:1.0})
        softmax = sess.run(tf.nn.softmax(pred))
        index = np.argmax(softmax, axis = 1)
        label_image = restore_image(index, im_size)
        imlbl = np.concatenate((image, label_image), axis=0)
        misc.imshow(imlbl)

end_time = time.time()
#print('Inference time: {0} seconds', end_time-start_time)
