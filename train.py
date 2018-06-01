from cv2 import imshow, cvtColor, waitKey, COLOR_RGB2BGR
import os.path
import numpy as np
import tensorflow as tf
import matplotlib as plt
from glob import glob
import random
import time
from models import vgg16
from helper import gen_batch_function, map_labels
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

def restore_image(image_vec, im_size = (352,800,3)):
    image_vec = [color[elem] for elem in image_vec]
    image_vec = np.array(image_vec)
    image = image_vec.reshape(im_size)
    return image

def restore_images(image_vec, im_size = (8,352,800,3)):
    image_vec = [color[elem] for elem in image_vec]
    image_vec = np.array(image_vec)
    image = image_vec.reshape(im_size)
    return image

def optimize(nn_output, correct_label, learning_rate, num_classes):

    logits = tf.reshape(nn_output, (-1, num_classes), name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes))

    #precision = tf.metrics.precision(labels = labels, predictions = logits)
    #recall = tf.metrics.recall(labels = labels, predictions = logits)

    #f_score = tf.divide(tf.multiply(precision, recall), tf.add(precision, recall))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, nr_train_images, batch_size, num_classes, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    lr = 1e-3
    kp = 0.5
    save_trained_model = True
    save_model_path = "./model/"

    print("Training")

    # Save graph
    #if save_trained_model:
    #    tf.train.write_graph(sess.graph_def, save_model_path, "graph.pb")

    for i in range(epochs):
        nr_images = 0
        if ((i == 30) | (i == 40)):
            lr = lr * 0.1
        print("Epoch: {0}. Learning rate: {1}".format(i+1, lr))

        for images, labels in get_batches_fn(batch_size, num_classes):
            #time_start = time.time()

            # Get random image for testing on current minibatch
            #test_images = random.choice(images)
            #test_image_4d = np.expand_dims(test_image, axis=0)

            #### Train #####
            t_op, curr_loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: images, correct_label: labels, keep_prob: kp, learning_rate: lr})

            #time_end = time.time()
            #print("Time for training operation: {0}s".format(time_end-time_start))

            nr_images += len(images)
            print("Total images: {0} - Loss: {1}".format(nr_images, curr_loss))

        if save_trained_model:
            # Save model checkpoint
            saver = tf.train.Saver().save(sess, save_model_path + 'model', global_step = epochs)
            tf.train.write_graph(sess.graph_def, save_model_path, "graph.pb", as_text = False)


if __name__=='__main__':
    images_folder = './Train/CameraRGB/'
    labels_folder = './Train/CameraSeg/'
    image_shape = (352, 800)
    num_classes = 3 # Vehicle, road, other
    epochs = 10
    batch_size = 8


    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:

        nr_train_images = len(glob(os.path.join(images_folder, '*.png')))

        get_batches_fn = gen_batch_function(images_folder, labels_folder, image_shape)

        correct_label = tf.placeholder(tf.int32, shape=(None, image_shape[0], image_shape[1], num_classes))
        #Input size: m, 352, 800, 3
        
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        nn_input, nn_output = vgg16(keep_prob, num_classes, image_shape)

        file_writer = tf.summary.FileWriter('./log', sess.graph)

        logits, train_op, cross_entropy_loss = optimize(nn_output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        train_nn(sess, epochs, nr_train_images, batch_size, num_classes, get_batches_fn, train_op, cross_entropy_loss, nn_input, correct_label, keep_prob, learning_rate)

'''
train_file = './Train/lyftdata.tfrecord'

with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    
    file_queue = tf.train.string_input_producer([train_file], num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['train/image'], tf.float32)
    label = tf.cast(features['train/label'], tf.int32)

    image = tf.reshape(image, [576, 800, 3])

    images, labels = tf.train.shuffle_batch([image, label], batch_size=8, capacity=30, num_threads=2, min_after_dequeue=10)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    img, lbl = sess.run([images, labels])

    for j in range(8):
        plt.subplot(2, 4, j+1)
        plt.imshow(img[j, ...])
    
    plt.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()
'''