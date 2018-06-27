from cv2 import imshow, cvtColor, waitKey, COLOR_RGB2BGR
import os.path
import numpy as np
import tensorflow as tf
import matplotlib as plt
from glob import glob
import random
import time
from models import *
from helper import gen_batch_function, map_labels
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_image(image_file, im_size):
    im_height, im_width, channels = im_size
    image = imread(image_file)

    vertical_start = 169
    vertical_end = vertical_start + im_height
    image = image[vertical_start:vertical_end, :im_width, :]
    return image

def optimize(nn_output, correct_label, learning_rate, num_classes):

    logits = tf.reshape(nn_output, (-1, num_classes), name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes))

    #precision = tf.metrics.precision(labels = labels, predictions = logits)
    #recall = tf.metrics.recall(labels = labels, predictions = logits)

    #f_score = tf.divide(tf.multiply(precision, recall), tf.add(precision, recall))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    #cross_entropy_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits, pos_weight=2))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.7, beta2=0.85).minimize(cross_entropy_loss)

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

    lr = 0.5e-3
    kp = 0.5
    save_trained_model = True
    save_model_path = "./model/"

    print("Training")


    for i in range(epochs):
        nr_images = 0
        if i == 20:
            lr = lr * 0.5
        if i == 30:
            lr = lr * 0.5
        if i == 40:
            lr = lr * 0.5

        print("Epoch: {0}. Learning rate: {1}".format(i+1, lr))

        for images, labels in get_batches_fn(batch_size, num_classes):

            #### Train #####
            t_op, curr_loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: images, correct_label: labels, keep_prob: kp, learning_rate: lr})

            nr_images += len(images)
            print("Total images: {0} - Loss: {1}".format(nr_images, curr_loss))

        if save_trained_model:
            # Save model checkpoint
            saver = tf.train.Saver().save(sess, save_model_path + 'model', global_step = epochs)
            # Save graph
            tf.train.write_graph(sess.graph_def, save_model_path, "graph.pb", as_text = False)


if __name__=='__main__':
    images_folder = './Train/CameraRGB/'
    labels_folder = './Train/CameraSeg/'
    image_shape = (352, 800)
    num_classes = 3 # Vehicle, road, other
    epochs = 50
    batch_size = 8
    train_model = 'vgg16'


    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:

        nr_train_images = len(glob(os.path.join(images_folder, '*.png')))

        get_batches_fn = gen_batch_function(images_folder, labels_folder, image_shape)

        correct_label = tf.placeholder(tf.int32, shape=(None, image_shape[0], image_shape[1], num_classes))
        
        learning_rate = tf.placeholder(tf.float32)
        
        if train_model == 'vgg16':
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            nn_input, nn_output = vgg16(keep_prob, num_classes, image_shape)
        
        elif train_model == 'vgg16_pretrained':
            vgg_path = './vgg'
            nn_input, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)
            nn_output = layers(l3, l4, l7, num_classes)
        
        elif train_model == 'resnet50':
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            nn_input, nn_output = resnet50(keep_prob, num_classes, image_shape)

        file_writer = tf.summary.FileWriter('./log', sess.graph)

        logits, train_op, cross_entropy_loss = optimize(nn_output, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        train_nn(sess, epochs, nr_train_images, batch_size, num_classes, get_batches_fn, train_op, cross_entropy_loss, nn_input, correct_label, keep_prob, learning_rate)
